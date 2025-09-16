__all__ = [
    "SpatialTransformer",
    "IntegrateVelocityField",
    "ResizeDisplacementField",
]

# Standard library imports
from typing import Tuple, Union, Optional

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as nnf

# Custom imports
import neurite as ne


class SpatialTransformer(nn.Module):
    """
    N-D Spatial transformation according to a deformation field.

    Uses a deformation field to transform the moving image.

    References
    ----------
    If you find this helpful, please cite the following paper:

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019.
    """

    def __init__(
        self,
        size: Tuple[int],
        interpolation_mode: str = "bilinear",
        align_corners: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Initialize `SpatialTransformer`.

        Parameters
        ----------
        size : tuple[int]
            Expected size of `moving_image` (input image to be warped) for the forward pass.
        interpolation_mode : str
            Algorithm used for interpolating the warped image. Default is  'bilinear'. Options are:
            'bilinear' | 'nearest' | 'bicubic'.
        align_corners : bool
            Map the corner points of the moving image to the corner points of the warped image.
        device : str
            Device to construct and hold the identity grid.
        """
        super().__init__()

        self.size = size
        self.device = device
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners

        # Make identity grid (the grid to later warp with deformation field) and register as a
        # buffer (without saving to `state_dict`: persistent=False)
        self.register_buffer(
            name='identity_grid',
            tensor=ne.utils.utils.grid(size=size, device=device),
            persistent=False  # Don't save to this module's state dict!
        )

    def forward(
        self,
        moving_image: torch.Tensor,
        deformation_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of `SpatialTransformer`

        Parameters
        ----------
        moving_image : torch.Tensor
            Tensor to be spatially transformed by `deformation_field`
        deformation_field : torch.Tensor
            Field causing the spatial transformation of `moving_image`.

        Returns
        -------
        torch.Tensor
            Warped `moving_image` according to the `deformation_field`.
        """

        # Validate the dimensions of the input
        if moving_image.dim() < 4 or deformation_field.dim() != moving_image.dim():
            raise ValueError(
                f"Expected `moving_image` to have at least 4 dimensions and for `flow field` to "
                f"match `moving_image` dimensions, got moving_image.dim()={moving_image.dim()}, "
                f"deformation_field.dim()={deformation_field.dim()}"
            )

        # Wow, this is legacy! Neither Adrian nor I know why the dims need to be permuted...
        # Well, at least that's what he said in his code
        deformation_field = deformation_field.moveaxis(1, -1).contiguous()

        # Warp the identity grid with the deformation field
        warped_grid = self.identity_grid + deformation_field

        # Normalize the axes so the range does not exceed the interval [-1, 1]
        warped_grid = self._normalize_warped_grid(warped_grid)

        # Sample grid
        warped_image = nnf.grid_sample(
            input=moving_image,
            grid=warped_grid,
            mode=self.interpolation_mode,
            align_corners=self.align_corners,
            padding_mode="border"
        )

        return warped_image

    def _normalize_warped_grid(
        self,
        warped_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize a warped grid to make PyTorch `grid_sample()` happy!

        PyTorch's `grid_sample()` requires coordinates in the range [-1, 1].
        This function scales and shifts the warped grid accordingly.

        Parameters
        ----------
        warped_grid : torch.Tensor
            The resultant of the identity grid and the deformation field.

        Returns
        -------
        torch.Tensor
            The warped grid rescaled to the range [-1, 1] for each spatial axis
        """

        for i, dim in enumerate(self.size):

            # Rescale each dimension individually
            warped_grid[..., i] = 2 * (warped_grid[..., i] / (dim - 1) - 0.5)

        return warped_grid


class IntegrateVelocityField(nn.Module):
    """
    Integrates a velocity field over multiple steps using the scaling and squaring method.

    This module ensures that transformations caused by a velocity field is diffeomorphic by
    compounding small, intermediate transformations (by recursive scaling and squaring). This
    ensures the resultant is both smooth and invertable.

    Attributes
    ----------
    steps : int
        The number of squaring steps used for integration.
    scale : float
        Scaling factor for the initial velocity field, determined as `1 / (2^steps)`.
    transformer : nn.Module
        A spatial transformer module used to iteratively warp the vector field.

    Examples
    -------
    ### Integrate a 2D velocity field over multiple steps:
    >>> shape = (128, 128)  # 2D spatial grid
    >>> integrator = IntegrateVelocityField(shape, steps=256)
    >>> velocity_field = torch.randn(1, 2, 128, 128)  # (B, C, H, W)
    >>> displacement_field = integrator(velocity_field)
    >>> displacement_field.shape
    torch.Size([1, 2, 128, 128])

    ### Perform integration on a 3D velocity field with a single scaling step:
    >>> shape = (64, 64, 64)  # 3D spatial grid
    >>> integrator = IntegrateVelocityField(shape, steps=1)
    >>> velocity_field = torch.randn(1, 3, 64, 64, 64)  # (B, C, D, H, W)
    >>> displacement_field = integrator(velocity_field)
    >>> displacement_field.shape
    torch.Size([1, 3, 64, 64, 64])
    """

    def __init__(
        self, shape: tuple,
        steps: int = 1,
        interpolation_mode: str = "bilinear",
        align_corners: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize `IntegrateVelocityField`

        Parameters
        ----------
        shape : tuple
            Shape of the input velocity field (excluding batch and channel dimensions).
        steps : int, optional
            Number of integration steps. A higher value leads to a more smooth and accurate
            integration at the cost of higher/longer computation. Default is 1.
        interpolation_mode : str
            Algorithm used for interpolating the warped image. Default is  'bilinear'. Options are:
            'bilinear' | 'nearest' | 'bicubic'.
        align_corners : bool
            Map the corner points of the moving image to the corner points of the warped image.
        device : str
            Device to construct and hold the identity grid.
        """

        super().__init__()

        if steps < 0:
            raise ValueError(f"steps should be >= 0, found: {steps}")

        self.steps = steps
        self.scale = 1.0 / (2 ** self.steps)  # Initial downscaling factor

        # Make the transformer which will perform the warping operation
        self.transformer = SpatialTransformer(shape, interpolation_mode, align_corners, device)

    def forward(self, velocity_field: torch.Tensor) -> torch.Tensor:
        """
        Integrates the input velocity field using scaling and squaring.

        Parameters
        ----------
        vector_field : torch.Tensor
            A velocity field of shape (B, C, *spatial_dims), where B is batch size,
            C is the number of vector components (typically spatial dimensions),
            and `spatial_dims` represent the grid dimensions.

        Returns
        -------
        torch.Tensor
            The integrated displacement field with the same shape as the input.
        """

        # Apply initial scaling to the velocity field
        velocity_field = velocity_field * self.scale

        # Integration loop
        for _ in range(self.steps):

            # Recursive integration step
            velocity_field = velocity_field + self.transformer(velocity_field, velocity_field)

        return velocity_field


class ResizeDisplacementField(nn.Module):
    """
    Resize and rescale a displacement field.

    Resizd a displacement field both spatially (via interpolation) and in magnitude (via scaling).

    Examples
    -------
    ### Resize a 2D displacement field
    >>> resize_field = ResizeDisplacementField(scale_factor=2.0, interpolation_mode="bilinear")
    >>> displacement_field = torch.rand(1, 2, 16, 16)  # Example displacement field in 2d
    >>> resized_displacement_field = resize_field(displacement_field)
    >>> print(resized_displacement_field.shape)  # Should be larger if scale_factor > 1
    torch.Size([1, 2, 32, 32])
    """

    def __init__(
        self,
        scale_factor: Optional[Union[float, int, ne.samplers.Sampler]] = 1.0,
        interpolation_mode: str = "bilinear",
        align_corners: bool = True,
    ):
        """
        Instantiate the `ResizeDisplacementField` module.

        Parameters
        ----------
        scale_factor : Optional[Union[float, int, Sampler]], optional
            Factor by which to stretch or shrink the spatial dimensions of the displacement field.
            Values of `scale_factor` > 1 stretch/expand the field, and values < 1 shrink it. By
            default None.
        interpolation_mode : str
            Algorithm used for interpolating the warped image. Default is  'bilinear'. Options are:
            'bilinear' | 'nearest' | 'bicubic', 'trilinear'.
        align_corners : bool
            Map the corner points of the moving image to the corner points of the warped image.
        """
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.scale_factor = ne.samplers.Fixed.make(scale_factor)

    def forward(self, displacement_field: torch.Tensor) -> torch.Tensor:
        """
        Instantiate the `ResizeDisplacementField` object.

        Parameters
        ----------
        displacement_field : torch.Tensor
            Vector field of shape (B, C, H, W) representing a displacement field, where C represents
            each spatial component of the vector field.

        Returns
        -------
        torch.Tensor
            Resized displacement field.
        """

        # Sample from the scaling sampler. If type Fixed, just get the fixed value!
        scale_factor = self.scale_factor()

        resized_displacement_field = nnf.interpolate(
            displacement_field * scale_factor,  # Scale the magnitudes of the displacement field
            scale_factor=scale_factor,
            mode=self.interpolation_mode,
            align_corners=self.align_corners,
        )

        return resized_displacement_field