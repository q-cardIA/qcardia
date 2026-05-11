"""
Utility functions for running cardiac inference pipelines.

This module provides high-level functions for:
- Model configuration and path validation
- Data directory discovery
- Multi-model, multi-view pipeline execution
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import numpy as np
from matplotlib import pyplot as plt


def get_data_directory(chamber_dir: Path) -> Optional[Path]:
    """
    Find the data directory within a chamber directory.
    
    Args:
        chamber_dir: Path to chamber directory (e.g., CINE_SAX)
        
    Returns:
        Path to data directory, or None if not found
        
    Notes:
        - For SAX views: DICOM files are directly in chamber_dir
        - For LA views: DICOM files are in a subdirectory
    """
    # Check if data is directly in chamber_dir (SAX case)
    dcm_files = list(chamber_dir.glob("*.dcm"))
    
    if dcm_files:
        return chamber_dir
    
    # Look for subdirectory with DICOM data.
    # Prefer known primary-series names; fall back to the subdirectory with the most DICOMs.
    _PREFERRED = {"sa stack", "2ch", "3ch", "4ch", "scar_sa", "scar_2ch"}
    subdirs = [
        d for d in chamber_dir.iterdir()
        if d.is_dir()
        and "segmentation" not in d.name.lower()
        and "result" not in d.name.lower()
        and not d.name.startswith(".")
    ]

    if not subdirs:
        return None

    # Prefer a subdir whose name matches a known primary series
    for preferred in _PREFERRED:
        for d in subdirs:
            if d.name.lower() == preferred:
                return d

    # Otherwise return the subdir containing the most DICOM files
    def _dcm_count(d: Path) -> int:
        return sum(1 for f in d.iterdir() if f.suffix.lower() == ".dcm" and not f.name.startswith("."))

    return max(subdirs, key=_dcm_count)
    
    return None


def determine_chamber_type(chamber_dir: Path) -> str:
    """
    Determine chamber type from directory name.
    
    Args:
        chamber_dir: Path to chamber directory
        
    Returns:
        Chamber type: "SAX", "2CH", "3CH", "4CH", or "UNKNOWN"
    """
    name = chamber_dir.name.upper()
    
    if "SAX" in name or "SA" in name:
        return "SAX"
    elif "2CH" in name:
        return "2CH"
    elif "3CH" in name:
        return "3CH"
    elif "4CH" in name:
        return "4CH"
    else:
        return "UNKNOWN"


def verify_model_paths(model_paths: Dict[str, Path]) -> bool:
    """
    Verify that all model paths exist.
    
    Args:
        model_paths: Dictionary mapping model names to paths
        
    Returns:
        True if all paths valid, False otherwise
    """
    all_valid = True
    
    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            print(f"  ✗ {model_name} path does not exist: {model_path}")
            all_valid = False
        else:
            print(f"  ✓ {model_name}: {model_path}")
    
    return all_valid


def get_middle_slice_and_frame(cine_seq, cine_segmentation) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Extract middle slice and frame for visualization.
    
    Args:
        cine_seq: CineSeries object
        cine_segmentation: Segmentation array
        
    Returns:
        Tuple of (mid_slice_idx, mid_frame_idx, pred_slice, input_image)
    """
    seg_shape = cine_segmentation.shape
    
    # Handle different dimensionalities
    if len(seg_shape) == 4:
        # Check format: (slices, frames, H, W) or (slices, H, W, frames)
        if seg_shape[1] == cine_seq.number_of_temporal_positions:
            # Format: (slices, frames, H, W)
            mid_slice_idx = seg_shape[0] // 2 if seg_shape[0] > 1 else 0
            mid_frame_idx = seg_shape[1] // 2
            pred_slice = cine_segmentation[mid_slice_idx, mid_frame_idx, :, :]
        else:
            # Format: (slices, H, W, frames)
            mid_slice_idx = seg_shape[0] // 2 if seg_shape[0] > 1 else 0
            mid_frame_idx = seg_shape[3] // 2
            pred_slice = cine_segmentation[mid_slice_idx, :, :, mid_frame_idx]
    elif len(seg_shape) == 3:
        # (slices, H, W) - single frame or (frames, H, W)
        mid_slice_idx = 0
        mid_frame_idx = seg_shape[0] // 2 if seg_shape[0] == cine_seq.number_of_temporal_positions else 0
        pred_slice = cine_segmentation[mid_frame_idx] if seg_shape[0] == cine_seq.number_of_temporal_positions else cine_segmentation[0]
    else:
        pred_slice = cine_segmentation
        mid_slice_idx = 0
        mid_frame_idx = 0
    
    # Get corresponding input image
    slice_keys = sorted(cine_seq.slice_data.keys())
    slice_key = slice_keys[mid_slice_idx] if mid_slice_idx < len(slice_keys) else slice_keys[0]
    
    # Get the pixel array
    pixel_array = cine_seq.slice_data[slice_key]["pixel_array"]
    
    # Handle pixel array - it might be a list of frames or numpy array
    if isinstance(pixel_array, list):
        input_image = pixel_array[mid_frame_idx] if mid_frame_idx < len(pixel_array) else pixel_array[0]
    elif isinstance(pixel_array, np.ndarray):
        if len(pixel_array.shape) > 2:
            input_image = pixel_array[mid_frame_idx] if mid_frame_idx < pixel_array.shape[0] else pixel_array[0]
        else:
            input_image = pixel_array
    else:
        input_image = np.array(pixel_array)
    
    return mid_slice_idx, mid_frame_idx, pred_slice, input_image


def create_data_check_visualization(cine_seq, chamber_type, output_path: Path):
    """
    Create a simple visualization to verify data loaded correctly.
    
    Args:
        cine_seq: CineSeries object
        chamber_type: Name of chamber
        output_path: Path to save visualization
    """
    first_slice_key = list(cine_seq.slice_data.keys())[0]
    first_frame = cine_seq.slice_data[first_slice_key]["pixel_array"][0]
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(first_frame, cmap='gray')
    ax.set_title(f"{chamber_type} - First frame")
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()


def print_pipeline_header():
    """Print pipeline header."""
    print("="*80)
    print("CARDIAC INFERENCE PIPELINE")
    print("="*80)


def print_results_summary(results_summary: List[Dict]):
    """
    Print summary of pipeline results.
    
    Args:
        results_summary: List of result dictionaries
    """
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - RESULTS SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results_summary if r["status"] == "SUCCESS")
    failed_count = sum(1 for r in results_summary if r["status"] == "FAILED")
    
    print(f"\nTotal: {len(results_summary)} tasks")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {failed_count}")
    
    if success_count > 0:
        print("\nSuccessful runs:")
        for result in results_summary:
            if result["status"] == "SUCCESS":
                print(f"  ✓ {result.get('model', 'N/A')} → {result.get('chamber', 'N/A')}")
                if "output" in result:
                    print(f"    Output: {result['output']}")
    
    if failed_count > 0:
        print("\nFailed runs:")
        for result in results_summary:
            if result["status"] == "FAILED":
                print(f"  ✗ {result.get('model', 'N/A')} → {result.get('chamber', 'N/A')}")
                if "error" in result:
                    print(f"    Error: {result['error']}")
    
    print("\n" + "="*80)
