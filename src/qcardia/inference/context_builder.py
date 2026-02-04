"""
Context builder for extracting spatial and temporal neighbors.

This module handles context window extraction for transformer models that
require neighboring slices and frames as input.
"""

from typing import List, Tuple
import torch
import numpy as np


class ContextBuilder:
    """
    Builds context windows for transformer models.
    
    Extracts spatial (neighboring slices) and temporal (neighboring frames)
    context around a target position for context-aware inference.
    
    Example:
        >>> builder = ContextBuilder(spatial_window=2, temporal_window=4)
        >>> context = builder.build_context(image_tensor, z=5, t=10)
        >>> print(context.shape)  # (B, C, H, W, Q) where Q = 1+2+4 = 7
    """
    
    def __init__(self, spatial_window: int = 0, temporal_window: int = 0, 
                 spatial_stride: int = 1, temporal_stride: int = 1):
        """
        Initialize context builder.
        
        Args:
            spatial_window: Number of spatial neighbors (slices)
            temporal_window: Number of temporal neighbors (frames)
            spatial_stride: Stride between spatial neighbors
            temporal_stride: Stride between temporal neighbors
        """
        self.spatial_window = spatial_window
        self.temporal_window = temporal_window
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
    
    def build_context(self, image_tensor: torch.Tensor, z: int, t: int) -> torch.Tensor:
        """
        Build context tensor for position (z, t).
        
        Context structure: [target] + [spatial neighbors] + [temporal neighbors]
        
        Args:
            image_tensor: Input tensor, shape (B, C, H, W, Z, T)
            z: Target slice index
            t: Target time index
            
        Returns:
            context_tensor: Shape (B, C, H, W, Q) 
                          where Q = 1 + spatial_window + temporal_window
        """
        # Validate input shape
        if len(image_tensor.shape) == 5:
            # 5D tensor: (B, C, H, W, T) - assume single slice
            image_tensor = image_tensor.unsqueeze(4)  # Add Z dimension
            B, C, H, W, T_new, _ = image_tensor.shape
            image_tensor = image_tensor.squeeze(-1)  # Remove extra dim
            # Rearrange to (B, C, H, W, Z=1, T)
            image_tensor = image_tensor.unsqueeze(4).transpose(4, 5)
            
        B, C, H, W, Z, T = image_tensor.shape
        
        # Extract target image
        target = image_tensor[..., z, t].unsqueeze(-1)  # (B, C, H, W, 1)
        
        context_parts = [target]
        
        # Spatial context (neighboring slices at same time t)
        if self.spatial_window > 0:
            neighbor_z = self.get_neighbor_indices(z, self.spatial_window, 
                                                   self.spatial_stride, Z)
            if len(neighbor_z) > 0:
                spatial_context = image_tensor[..., neighbor_z, t]  # (B, C, H, W, n_spatial)
                context_parts.append(spatial_context)
        
        # Temporal context (neighboring frames at same slice z)
        if self.temporal_window > 0:
            neighbor_t = self.get_neighbor_indices(t, self.temporal_window,
                                                   self.temporal_stride, T)
            if len(neighbor_t) > 0:
                temporal_context = image_tensor[..., z, neighbor_t]  # (B, C, H, W, n_temporal)
                context_parts.append(temporal_context)
        
        # Concatenate along Q dimension
        context_tensor = torch.cat(context_parts, dim=-1)
        
        return context_tensor
    
    def get_neighbor_indices(self, center_idx: int, window: int, 
                            stride: int, max_range: int) -> List[int]:
        """
        Get neighbor indices with wrapping.
        
        Ported from inference_engine/shared/utils.py
        
        Args:
            center_idx: Center index
            window: Size of the context window
            stride: Stride between indices
            max_range: Maximum valid index range
            
        Returns:
            List of neighbor indices (excluding center)
            
        Example:
            >>> builder = ContextBuilder()
            >>> indices = builder.get_neighbor_indices(5, 2, 1, 10)
            >>> print(indices)  # [4, 6] (2 neighbors around 5)
        """
        indices = []
        start = center_idx - (window * stride) // 2
        i = 0
        while len(indices) < window:
            idx = (start + i * stride) % max_range
            if idx != center_idx:
                indices.append(idx)
            i += 1
        return indices
    
    def get_context_shape(self, image_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Calculate output context shape.
        
        Args:
            image_shape: Input image shape (B, C, H, W, Z, T)
            
        Returns:
            Output shape (B, C, H, W, Q)
        """
        if len(image_shape) == 6:
            B, C, H, W, Z, T = image_shape
        elif len(image_shape) == 5:
            B, C, H, W, T = image_shape
            Z = 1
        else:
            raise ValueError(f"Expected 5D or 6D tensor, got {len(image_shape)}D")
        
        Q = 1 + self.spatial_window + self.temporal_window
        return (B, C, H, W, Q)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ContextBuilder(spatial={self.spatial_window}, "
                f"temporal={self.temporal_window}, "
                f"spatial_stride={self.spatial_stride}, "
                f"temporal_stride={self.temporal_stride})")
