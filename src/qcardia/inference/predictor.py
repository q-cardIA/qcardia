"""
Predictor module for orchestrating inference.

Handles both simple per-image inference (UNet) and context-aware
inference (Transformer) with spatial/temporal context windows.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .config_handler import InferenceConfig
from .context_builder import ContextBuilder


class InferencePredictor:
    """
    Main prediction orchestrator for both simple and context-aware models.
    
    This class handles:
    - Simple per-image inference (UNet models)
    - Context-aware inference (Transformer models with spatial/temporal context)
    - Batch processing
    - Different tensor dimensionalities (4D, 5D, 6D)
    
    Example:
        >>> predictor = InferencePredictor(model, config, device='cpu')
        >>> predictions = predictor.predict(preprocessed_tensor)
    """
    
    def __init__(self, model: nn.Module, config: InferenceConfig, 
                 device: str = None, batch_size: int = 50):
        """
        Initialize predictor.
        
        Args:
            model: Loaded PyTorch model
            config: Inference configuration
            device: Device for inference ('cpu' or 'cuda'). If None, auto-detect.
            batch_size: Batch size for simple inference
        """
        # Auto-detect device if not specified
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Windows/Linux
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")  # MacOS
        else:
            self.device = torch.device("cpu")
            print("No GPU available; using CPU")
            
        self.model = model.to(self.device)
        self.config = config
        self.batch_size = batch_size
        
        # Determine if model needs context
        self.is_transformer = config.is_transformer()
        self.needs_context = config.needs_context()
        
        # Create context builder if needed
        if self.needs_context:
            self.context_builder = ContextBuilder(
                spatial_window=config.spatial_window,
                temporal_window=config.temporal_window,
                spatial_stride=config.spatial_stride,
                temporal_stride=config.temporal_stride
            )
        else:
            self.context_builder = None
    
    def predict(self, preprocessed_tensor: torch.Tensor) -> torch.Tensor:
        """
        Main prediction method.
        
        Automatically selects appropriate inference path based on model type.
        
        Args:
            preprocessed_tensor: Preprocessed input tensor
                Shape varies by dimensionality:
                - Simple: (N, C, H, W) - batch of images
                - 2D+T: (N, C, H, W, T) - with time
                - 2D+Z: (N, C, H, W, Z) - with slices
                - 4D: (N, C, H, W, Z, T) - full 4D
        
        Returns:
            predictions: Tensor with class predictions
                Shape: (N, num_classes, H_target, W_target, ...)
        """
        # Detect tensor dimensionality
        tensor_dims = len(preprocessed_tensor.shape)
        
        if not self.needs_context:
            # Simple per-image inference (original behavior)
            return self._predict_simple(preprocessed_tensor)
        else:
            # Context-aware inference
            return self._predict_context_aware(preprocessed_tensor)
    
    def _predict_simple(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Simple batch-wise inference for UNet models.
        
        This is the original qcardia inference behavior.
        
        Args:
            tensor: Shape (N, C, H, W) or (N, C, H, W, Z, T) flattened
            
        Returns:
            predictions: Shape (N, num_classes, H_target, W_target)
        """
        # Reshape to 4D if needed
        original_shape = tensor.shape
        if len(tensor.shape) > 4:
            # Flatten extra dimensions into batch
            N = np.prod(tensor.shape[:-3]) if len(tensor.shape) == 5 else np.prod(tensor.shape[:-3])
            tensor = tensor.reshape(-1, tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])
        
        N, C, H, W = tensor.shape
        
        # Initialize output
        model_output = torch.zeros(
            N,
            self.config.nr_classes,
            self.config.target_size[0],
            self.config.target_size[1]
        )
        
        # Run inference in batches
        self.model.eval()
        with torch.no_grad():
            # Create progress bar for batches
            total_batches = (N + self.batch_size - 1) // self.batch_size
            pbar = tqdm(range(0, N, self.batch_size), 
                       desc="Inference (simple)", 
                       total=total_batches)
            
            for i in pbar:
                end_idx = min(i + self.batch_size, N)
                batch = tensor[i:end_idx]
                
                if batch.shape[0] > 0:
                    output = self.model(batch.to(self.device))
                    
                    # Handle different output formats
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    
                    model_output[i:end_idx] = output.cpu()
                    pbar.set_postfix({'images': f'{end_idx}/{N}'})
        
        return model_output
    
    def _predict_context_aware(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Context-aware inference for Transformer models.
        
        Extracts spatial and temporal context for each position and runs
        inference with context windows.
        
        Args:
            tensor: Shape (N, C, H, W, Z, T) or similar
            
        Returns:
            predictions: Shape (N, num_classes, H_target, W_target, Z, T)
        """
        # Ensure 6D tensor: (B, C, H, W, Z, T)
        tensor_6d = self._ensure_6d(tensor)
        B, C, H, W, Z, T = tensor_6d.shape
        
        # Initialize output
        output = torch.zeros(
            B, 
            self.config.nr_classes,
            self.config.target_size[0],
            self.config.target_size[1],
            Z, 
            T
        )
        
        self.model.eval()
        with torch.no_grad():
            # Iterate over all positions with progress bar
            total_positions = Z * T
            pbar = tqdm(total=total_positions, 
                       desc="Inference (context-aware)")
            
            for z in range(Z):
                for t in range(T):
                    # Build context for this position
                    context = self.context_builder.build_context(tensor_6d, z, t)
                    
                    # Process batch elements one at a time to minimize GPU memory
                    # This is critical for context-aware models with large context windows
                    batch_preds = []
                    for b_idx in range(B):
                        # Forward pass with single batch element
                        pred = self.model(context[b_idx:b_idx+1].to(self.device))
                        
                        # Handle different output formats
                        if isinstance(pred, (tuple, list)):
                            pred = pred[0]
                            if isinstance(pred, (tuple, list)):
                                pred = pred[0]
                        
                        # Handle decode_all_context mode
                        if self.config.decode_all_context and len(pred.shape) > 4:
                            pred = pred[:, 0, ...]  # Take first output
                        
                        batch_preds.append(pred.cpu())
                    
                    # Concatenate batch predictions
                    output[..., z, t] = torch.cat(batch_preds, dim=0)
                    pbar.update(1)
                    pbar.set_postfix({'slice': z+1, 'frame': t+1})
            
            pbar.close()
        
        return output
    
    def _ensure_6d(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensure tensor is 6D: (B, C, H, W, Z, T).
        
        Args:
            tensor: Input tensor of various shapes
            
        Returns:
            6D tensor
        """
        if len(tensor.shape) == 4:
            # Check if this is flattened (N_total, C, H, W) where N_total = Z * T
            # In this case, we need additional information to reshape properly
            # For now, assume (B, C, H, W) -> (B, C, H, W, 1, 1) for single slice/frame
            N, C, H, W = tensor.shape
            
            # Heuristic: if N is very large (>50), it's likely flattened multi-slice data
            # We'll need to reshape it. Try to infer Z and T from config or assume square-ish
            if hasattr(self.config, 'n_slices') and hasattr(self.config, 'n_frames'):
                Z = self.config.n_slices
                T = self.config.n_frames
                if N == Z * T:
                    # Reshape (Z*T, C, H, W) -> (1, C, H, W, Z, T)
                    return tensor.view(Z, T, C, H, W).permute(2, 3, 4, 0, 1).unsqueeze(0)
            
            # Default: treat as independent batch samples
            return tensor.unsqueeze(-1).unsqueeze(-1)
            
        elif len(tensor.shape) == 5:
            # (B, C, H, W, X) -> need to determine if X is Z or T
            # Assume it's T (time) and Z=1
            return tensor.unsqueeze(4)  # (B, C, H, W, Z=1, T)
        elif len(tensor.shape) == 6:
            return tensor
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
    def _reshape_output(self, output: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Reshape output to match target shape.
        
        Args:
            output: Model output
            target_shape: Desired output shape
            
        Returns:
            Reshaped output
        """
        # Implementation depends on specific requirements
        return output
    
    def get_prediction_info(self) -> dict:
        """
        Get information about the predictor configuration.
        
        Returns:
            Dictionary with predictor info
        """
        return {
            "model_type": "transformer" if self.is_transformer else "unet",
            "needs_context": self.needs_context,
            "batch_size": self.batch_size,
            "device": self.device,
            "spatial_window": self.config.spatial_window if self.needs_context else 0,
            "temporal_window": self.config.temporal_window if self.needs_context else 0,
            "target_size": self.config.target_size,
            "nr_classes": self.config.nr_classes
        }
    
    def __repr__(self) -> str:
        """String representation."""
        context_str = ""
        if self.needs_context:
            context_str = f", context=({self.config.spatial_window}S+{self.config.temporal_window}T)"
        return (f"InferencePredictor(model={'Transformer' if self.is_transformer else 'UNet'}"
                f"{context_str}, device={self.device})")
