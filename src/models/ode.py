import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from models.attention import flash_attention
import torch.cuda.amp as amp
from utils.misc import auto_grad_checkpoint
from timm.layers import DropPath
from models.autoencoder import Encoder, Decoder


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast(device_type='cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast(device_type='cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


@torch.amp.autocast(device_type='cuda', enabled=False)
def inv_freq(dims, base=10000):
    inv_freq = [1.0 / (base ** (torch.arange(0, dim, 2).to(dtype=torch.float64) / dim)) for dim in dims]
    return inv_freq


@torch.amp.autocast(device_type='cuda', enabled=False)
def rope_cos_sin(inv_freqs, position_ids):
    """
    inv_freqs = List[torch.Tensor]
    """
    freqs_cis_batch = []
    for i in range(len(position_ids)):
        freqs_cis = []
        for idx, inv_freq in enumerate(inv_freqs):
            freqs = torch.outer(position_ids[i, :, idx], inv_freq)
            freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs))
        freqs_cis = torch.cat(freqs_cis, dim=-1) # seq_len, dim
        freqs_cis_batch.append(freqs_cis)
    freqs_cis_batch = torch.stack(freqs_cis_batch) # bs, seq_len, dim
    return freqs_cis_batch


@torch.amp.autocast(device_type='cuda', enabled=False)
def mmrope_apply(x, freqs):
    """
    x: (bs, seq_len, n, dim)
    freqs: (seq_len, dim),
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2)) # (bs, seq_len, n, dim/2)
    x_out = torch.view_as_real(x * freqs[:, :, None, :]).flatten(3)
    return x_out.to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class LayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens=None, grid_sizes=None, freqs=None, position=True):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        if position:
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)
        
        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CrossAttention(SelfAttention):
    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class SelfAttention2(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens=None, grid_sizes=None, freqs=None, position=True):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        if position:
            q = mmrope_apply(q, freqs)
            k = mmrope_apply(k, freqs)
        
        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CrossAttention2(SelfAttention):
    def forward(self, x, context, context_lens, current_freqs, future_freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        q = mmrope_apply(q, future_freqs)
        k = mmrope_apply(k, current_freqs)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class DecoupledFFN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        lora_rank: int = 16
    ):
        super().__init__()
        self.ffn_cn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, embed_dim)
        )

        if lora_rank != 0:
            self.ffn_ad = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(approximate='tanh'),
                nn.Linear(ffn_dim, embed_dim)
            )
            # self.ffn_ad_meta_encoder = nn.Sequential(
            #     nn.Linear(6, 128),
            #     nn.GELU(approximate='tanh'),
            #     nn.Linear(128, 1)
            # )

    # def forward(self, x, alpha=None):
    #     cn_path = self.ffn_cn(x) # bs, l, d
    #     if hasattr(self, 'ffn_ad') or hasattr(self, 'ffn_ad_lora'):
    #         meta = torch.sigmoid(self.ffn_ad_meta_encoder(alpha))
    #         alpha, _ = torch.split(alpha, [1, 5], dim=1)
    #         alpha[alpha > 0.1] = 1.0
    #         alpha = alpha.unsqueeze(-1) * meta.unsqueeze(-1)
    #         ad_path = self.ffn_ad(x)
    #         return cn_path + alpha * ad_path
    #     return cn_path 

    def forward(self, x, alpha=None):
        cn_path = self.ffn_cn(x) # bs, l, d
        if hasattr(self, 'ffn_ad') or hasattr(self, 'ffn_ad_lora'):
            if isinstance(alpha, torch.Tensor) and alpha.dim() == 1:
                alpha = alpha.unsqueeze(-1).unsqueeze(-1) # bs, 1, 1
            if hasattr(self, 'ffn_ad'):
                ad_path = self.ffn_ad(x)
            elif hasattr(self, 'ffn_ad_lora'):
                ad_path = self.ffn_ad_lora(x)
            return cn_path + alpha * ad_path
        return cn_path


class DecoupledFFN1(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        lora_rank: int = 16
    ):
        super().__init__()
        self.ffn_cn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, embed_dim)
        )

        if lora_rank != 0:
            self.ffn_ad = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(approximate='tanh'),
                nn.Linear(ffn_dim, embed_dim)
            )
            # self.ffn_ad_meta_encoder = nn.Sequential(
            #     nn.Linear(6, 128),
            #     nn.GELU(approximate='tanh'),
            #     nn.Linear(128, 1)
            # )

    # def forward(self, x, alpha=None):
    #     cn_path = self.ffn_cn(x) # bs, l, d
    #     if hasattr(self, 'ffn_ad') or hasattr(self, 'ffn_ad_lora'):
    #         meta = torch.sigmoid(self.ffn_ad_meta_encoder(alpha))
    #         alpha, _ = torch.split(alpha, [1, 5], dim=1)
    #         alpha[alpha > 0.1] = 1.0
    #         alpha = alpha.unsqueeze(-1) * meta.unsqueeze(-1)
    #         ad_path = self.ffn_ad(x)
    #         return cn_path + alpha * ad_path
    #     return cn_path 

    def forward(self, x, alpha=None):
        cn_path = self.ffn_cn(x) # bs, l, d
        if hasattr(self, 'ffn_ad') or hasattr(self, 'ffn_ad_lora'):
            if isinstance(alpha, torch.Tensor) and alpha.dim() == 1:
                alpha = alpha.unsqueeze(-1).unsqueeze(-1) # bs, 1, 1
            if hasattr(self, 'ffn_ad'):
                ad_path = self.ffn_ad(x)
            elif hasattr(self, 'ffn_ad_lora'):
                ad_path = self.ffn_ad_lora(x)
            return cn_path + alpha * ad_path
        return cn_path

class RABA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm = LayerNorm(embed_dim, eps=1e-6)
        self.macro_attn = SelfAttention(embed_dim, num_heads)
    
    def forward(self, x, M):
        B, L, D = x.shape # L patch number
        B, _L, R = M.shape # _L patch number, R: region number

        region_tokens = torch.einsum('bld,blr->brd', x, M)
        region_tokens = torch.bmm(M.transpose(1, 2), x) # brl,bld -> brd

        region_affiliation_sums = M.sum(dim=1, keepdim=True).transpose(1, 2) # br1

        region_tokens = region_tokens / region_affiliation_sums.clamp(min=1e-6) # brl

        updated_region_tokens = self.macro_attn(self.norm(region_tokens), position=False)

        broadcasted_tokens = torch.bmm(M, updated_region_tokens) # blr,brd->bld
        return broadcasted_tokens
    

class FusedAttnDiTBlock2(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        lora_rank=32
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # stage1 standard attention
        self.norm1_t = LayerNorm(dim, eps=eps)
        self.self_attn_t = SelfAttention2(dim, num_heads, window_size, qk_norm, eps)

        # self.raba_attn_t = RABA(dim, num_heads)
        self.norm3 = LayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = CrossAttention(dim, num_heads, (-1, -1), qk_norm, eps) 

        # stage4 ffn
        self.norm2_t = LayerNorm(dim, eps=eps)
        self.ffn_t = DecoupledFFN(dim, ffn_dim, lora_rank)

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    
    def forward(self, tokens_t, seq_lens, cond_embed, M, grid_sizes, delta_freqs, alpha, context):
        assert cond_embed.dtype == torch.float32
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            e = (self.modulation + cond_embed).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # stage1 standard attention
        y_t = self.self_attn_t(self.norm1_t(tokens_t).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes, delta_freqs)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            tokens_t = tokens_t + y_t * e[2]

        # stage2 raba attention
        tokens_t = tokens_t + self.cross_attn(self.norm3(tokens_t), context, context_lens=None)

        # stage4 ffn
        y_t = self.ffn_t(self.norm2_t(tokens_t).float() * (1 + e[4]) + e[3], alpha)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            tokens_t = tokens_t + y_t * e[5]
        return tokens_t


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        assert e.dtype == torch.float32
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, dim, img_size, patch_size):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # pad to make sure the input size is divisible by patch size
        b, c, d, h, w = x.shape
        pad_d = (self.patch_size[0] - d % self.patch_size[0]) % self.patch_size[0]
        pad_h = (self.patch_size[1] - h % self.patch_size[1]) % self.patch_size[1]
        pad_w = (self.patch_size[2] - w % self.patch_size[2]) % self.patch_size[2]
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
        return self.proj(x).flatten(2).transpose(1, 2)
    
    @property
    def seq_len(self):
        return math.prod(
            [math.ceil(i / j) for i, j in zip(self.img_size, self.patch_size)]
        )
    
    @property
    def seq_shape(self):
        return [math.ceil(i / j) for i, j in zip(self.img_size, self.patch_size)]


class ConditionEmbedder(nn.Module):
    def __init__(self, cond_dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.fake_latent = nn.Parameter(torch.randn(1, hidden_size) / hidden_size**0.5)
    
    def forward(self, c, dropout_ids):
        # 因为还有其他condition，所以传dropout_ids
        # dropout_ids [True, False, True, False]
        # c [N, dim]
        c_emb = self.mlp(c)
        dropout_ids = dropout_ids.unsqueeze(1).float().to(c_emb.device) # N,1
        c_emb = dropout_ids * self.fake_latent + (1 - dropout_ids) * c_emb
        return c_emb


class MMDiTModel2(nn.Module):
    _no_split_modules = ['FusedAttnDiTBlock']
    def __init__(
        self,
        img_size=(15, 18, 15),
        patch_size=(1, 1, 1),
        in_dim=3,
        dim=768,
        ffn_dim=1536,
        freq_dim=256,
        out_dim=3,
        num_heads=12,
        num_layers=12,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        condition_dim=5,
        class_dropout_prob=0.1,
        lora_rank=32,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.condition_dim = condition_dim
        self.class_dropout_prob = class_dropout_prob
        self.lora_rank = lora_rank

        self.target_img_size = tuple((math.ceil(i / j) * j for i, j in zip(img_size, patch_size)))
        self.patch_embedding_c = PatchEmbedding(in_dim, dim, self.target_img_size, patch_size)

        self.condition_meta_embedding = nn.Sequential(nn.Linear(condition_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cond_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList([
            FusedAttnDiTBlock2(
                dim,
                ffn_dim,
                num_heads,
                window_size,
                qk_norm,
                cross_attn_norm,
                eps,
                lora_rank
            ) for _ in range(num_layers)
        ])

        self.head = Head(dim, out_dim, patch_size, eps)
        
        d = dim // num_heads
        self.inv_freqs = inv_freq([d - 4 * (d // 6), d // 6 * 2, d // 6 * 2])
        
        self.init_weights()

    def forward(self, x, condition_meta, condition_img, alpha, M, context, **kwargs):
        # params
        device = self.patch_embedding_c.proj.weight.device
        bs = x.size(0)

        if self.inv_freqs[0].device != device:
            self.inv_freqs = [inv_freq.to(device) for inv_freq in self.inv_freqs]
        
        # position ids
        current_age, current_dx, future_dx, current_mmse, sex, future_age = condition_meta.split([1, 1, 1, 1, 1, 1], dim=1)

        condition_meta = torch.cat([current_dx, future_dx, current_mmse, sex, (future_age - current_age) * 100.0], dim=1) # bs,3

        input_d, input_h, input_w = self.patch_embedding_c.seq_shape
        d_coords, h_coords, w_coords = torch.meshgrid(torch.arange(input_d, device=device).float(), torch.arange(input_h, device=device).float(), torch.arange(input_w, device=device).float(), indexing='ij')
        delta_position_ids = torch.stack([d_coords, h_coords, w_coords], dim=-1).reshape(1, -1, 3).repeat(bs, 1, 1)

        freqs = rope_cos_sin(self.inv_freqs, delta_position_ids) # bs,seqlen,dim

        x_c = self.patch_embedding_c(condition_img)
        # x_c = self.condition_img_mlpproj(x_c)
        grid_sizes = torch.stack(
            [torch.tensor([self.target_img_size[0] // self.patch_size[0], self.target_img_size[1] // self.patch_size[1], self.target_img_size[2] // self.patch_size[2]], dtype=torch.long) for u in x_c]) # [b, 3]
        
        seq_lens = torch.tensor([u.size(0) for u in x_c], dtype=torch.long) # [b]

        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            e = self.condition_meta_embedding(condition_meta)
            e0 = self.cond_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
        
        kwargs = dict(
            seq_lens=seq_lens,
            cond_embed=e0,
            M=M,
            grid_sizes=grid_sizes,
            delta_freqs=freqs,
            alpha=alpha,
            context=context
        )
        # forward
        for block in self.blocks:
            if self.training:
                x_c = auto_grad_checkpoint(block, x_c, **kwargs)
            else:
                x_c = block(x_c, **kwargs)

        # output
        x = self.head(x_c, e)
        x = self.unpatchify(x, grid_sizes)
        return x

    def unpatchify(self, x, grid_sizes):
        T_p, H_p, W_p = self.patch_size
        N_t, N_h, N_w = grid_sizes[0].unbind(dim=0)
        R_t, R_h, R_w = self.img_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_dim,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and 'lora' not in name:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding_c.proj.weight.flatten(1))


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]] # 正确的，https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997/11，神人设计

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)


class UniModel(nn.Module):
    def __init__(
        self,
        stage2=False
    ):
        super().__init__()
        self.encoder = Encoder(
            spatial_dims=3,
            in_channels=1,
            num_channels=(32, 64, 64, 64),
            out_channels=128,
            num_res_blocks=(1, 2, 2, 2),
            norm_num_groups=16,
            norm_eps=1e-6,
            attention_levels=(False, False, False, False),
            with_nonlocal_attn=False,
        )

        self.ana_encoder = Encoder(
            spatial_dims=3,
            in_channels=1,
            num_channels=(16, 32, 32, 32),
            out_channels=32,
            num_res_blocks=(1, 1, 1, 1),
            norm_num_groups=16,
            norm_eps=1e-6,
            attention_levels=(False, False, False, False),
            with_nonlocal_attn=False,
        )

        self.dit = MMDiTModel2(
            img_size=(15, 18, 15),
            patch_size=(2, 2, 2),
            in_dim=128,
            dim=768,
            ffn_dim=3072,
            freq_dim=256,
            out_dim=128,
            num_heads=12,
            num_layers=12,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
            condition_dim=5,
            class_dropout_prob=0.1,
            lora_rank=32,
        )
        self.decoder = Decoder(
            spatial_dims=3,
            num_channels=(32, 64, 64, 64),
            in_channels=128,
            out_channels=3, # svf
            num_res_blocks=(1, 2, 2, 2),
            norm_num_groups=16,
            norm_eps=1e-6,
            attention_levels=(False, False, False, False),
            with_nonlocal_attn=False,
        )
        self.stn = SpatialTransformer(
            size=(120, 144, 120)
        )
        self.mlp = nn.Sequential(
            nn.Linear(32, 768),
            nn.GELU(),
            nn.Linear(768, 768),
        )
        nn.init.xavier_normal_(self.mlp[0].weight)
        nn.init.xavier_normal_(self.mlp[2].weight)
        if stage2:
            for name, param in self.named_parameters():
                param.requires_grad = False

                if 'ffn_ad' in name:
                    param.requires_grad = True
            
                if 'condition_meta_embedding' in name:
                    param.requires_grad = True
            
                if 'cond_projection' in name:
                    param.requires_grad = True
            
                if 'modulation' in name:
                    param.requires_grad = True
            

    def forward(
        self,
        x_dummy, # zero
        condition_meta, # condition
        condition_img, # moving
        alpha,
        M,
        segm,
        **kwargs
    ):
        src_img = condition_img.clone()
        condition_img = self.encoder(condition_img)
        x = torch.ones_like(condition_img) * 0.01
        _, context = self.ana_encoder(segm.float(), [2, -1])

        context = torch.cat([c.reshape(c.shape[0], c.shape[1], -1).permute(0, 2, 1) for c in context], dim=1)
        context = self.mlp(context)
        x = self.dit(x, condition_meta, condition_img, alpha, M, context)
        pred_vec = self.decoder(x)
        output = self.stn(src_img, pred_vec)
        return output, pred_vec


if __name__ == '__main__':
    model = UniModel().cuda()
    from utils.misc import set_grad_checkpoint
    set_grad_checkpoint(model, False)
    x = torch.randn(2, 3, 15, 18, 15).cuda()
    condition_meta = torch.randn(2, 6).cuda()
    condition_img = torch.randn(2, 1, 120, 144, 120).cuda()
    segm = torch.randn(2, 1, 120, 144, 120).cuda()
    alpha = torch.randn(2).cuda()
    M = torch.randn(2, 576, 34).cuda()
    print(f'number of parameters {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    import time
    for i in range(100):
        start = time.time()
        out, out2 = model(x, condition_meta, condition_img, alpha, M, segm)
        loss = out.mean() + out2.mean()
        loss.backward()
        end = time.time()
        print(end - start)
    print(out.shape)
