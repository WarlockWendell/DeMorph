TORCH_DISTRIBUTED_DEBUG=DETAIL OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,2 ./scripts/my_torchrun --standalone --nnodes=1 --nproc_per_node=2 train2.py \
    --cfg ../config/ode.yaml \
    --batch_size 2 \
    --epochs 50 \
    --data_list ../data/train.csv \
    --data_path /home/guest/projects/code/ode/src/ADNI_NII_NORM_1.pt \
    --mpath ../data/cache_M_16_16_16 \
    --output_dir ../output/ode5 \
    --log_dir ../output/ode5 \
    --grad_checkpointing \
    --grad_clip true \
    --weight_decay 0.00001 \
    --lr 0.001 \
    --lr_schedule cosine \
    --warmup_epochs 5 \
    --ema_rate 0.99


TORCH_DISTRIBUTED_DEBUG=DETAIL OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,2 ./scripts/my_torchrun --standalone --nnodes=1 --nproc_per_node=2 train2.py \
    --cfg ../config/ode2.yaml \
    --batch_size 2 \
    --epochs 25 \
    --data_list ../data/train.csv \
    --data_path /home/guest/projects/code/ode/src/ADNI_NII_NORM_1.pt \
    --mpath ../data/cache_M_16_16_16 \
    --output_dir ../output/ode6 \
    --log_dir ../output/ode6 \
    --grad_checkpointing \
    --grad_clip true \
    --weight_decay 0.00001 \
    --lr 0.001 \
    --lr_schedule cosine \
    --warmup_epochs 5 \
    --ema_rate 0.99 \
    --resume ../output/ode5/checkpoint-0049.pth \
    --stage2