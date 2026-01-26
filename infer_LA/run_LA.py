"""Multi-model inference pipeline for long-axis cine segmentation.

Compares segmentation predictions from multiple models (UNet-LA, UNet-SA) on
2-chamber, 3-chamber, and 4-chamber cardiac cine MRI sequences.

Outputs:
    - Segmentation masks (NIfTI format)
    - Static and animated visualizations
    - Volume curves and ejection fraction metrics
    - Anatomical marker point visualizations

Usage:
    python run_LA.py
"""

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
import sys

# Add qcardia source to path
qcardia_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(qcardia_path))
from qcardia.series import CineSeries

# Import visualization utilities
import utils_viz

# Configuration - Multiple models for comparison
MODELS = {
    "UNet-LA": Path("/home/20203531/msc-thesis/msc-marcus/wandb/run-20250702_165723-sg792w9e_UNet-LA"),
    "UNet-SA": Path("/home/20203531/msc-thesis/msc-marcus/wandb/run-20251026_173443-bi8ls2x5_nnUNet-baseline2")
}
DATA_PATH = Path("/home/20203531/msc-thesis/data/CardiSorted_QLGE04_")

# Chamber types to process
CHAMBER_TYPES = ["CINE_2CH", "CINE_3CH", "CINE_4CH"]

print("Multi-model LA inference pipeline")
print(f"Models: {', '.join(MODELS.keys())}")

# Verify paths
for model_name, model_path in MODELS.items():
    if not model_path.exists():
        print(f"ERROR: {model_name} path does not exist: {model_path}")
        sys.exit(1)

if not DATA_PATH.exists():
    print(f"ERROR: Data path does not exist: {DATA_PATH}")
    sys.exit(1)

print("✓ Paths verified")

# Process chambers
for chamber_type in CHAMBER_TYPES:
    chamber_dir = DATA_PATH / chamber_type
    print(f"\n{'='*80}")
    print(f"Processing {chamber_type}")
    print(f"{'='*80}")
    
    if not chamber_dir.exists():
        print(f"  Skipping {chamber_type} (not found)")
        continue
    
    # Find data directory (exclude segmentation/result directories)
    subdirs = [d for d in chamber_dir.iterdir() 
               if d.is_dir() 
               and 'segmentation' not in d.name.lower() 
               and 'result' not in d.name.lower()]
    if not subdirs:
        print(f"  Skipping {chamber_type} (no data subdirectories)")
        continue
    
    cine_dir = subdirs[0]
    
    # Process with each model
    for model_name, wandb_run_path in MODELS.items():
        print(f"\n  {model_name}:")
        
        OUTPUT_PATH = Path(f"{cine_dir}_results_{model_name}")
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load data
            cine_seq = CineSeries(cine_dir, batch_size=50)
            
            # Save data check visualization (first model only)
            if model_name == list(MODELS.keys())[0]:
                first_slice_key = list(cine_seq.slice_data.keys())[0]
                first_frame = cine_seq.slice_data[first_slice_key]["pixel_array"][0]
                
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(first_frame, cmap='gray')
                ax.set_title(f"{chamber_type} - First frame")
                ax.axis('off')
                vis_path = OUTPUT_PATH / f"{chamber_type}_data_check.png"
                plt.savefig(vis_path, bbox_inches='tight', dpi=100)
                plt.close()
            
            # Run prediction
            print(f"    Predicting...")
            cine_segmentation = cine_seq.predict_segmentation(wandb_run_path)
            
            if cine_segmentation.max() == 0:
                print(f"    WARNING: Empty prediction")
            
            # Prepare visualization
            seg_shape = cine_segmentation.shape
            
            # Handle different dimensionalities
            # Expected format: (slices, height, width, frames) or (slices, frames, height, width)
            if len(seg_shape) == 4:
                # Check if it's (slices, frames, H, W) or (slices, H, W, frames)
                if seg_shape[1] == cine_seq.number_of_temporal_positions:
                    # Format: (slices, frames, H, W)
                    mid_slice_idx = 0  # Only 1 slice for LA views
                    mid_frame_idx = seg_shape[1] // 2
                    pred_slice = cine_segmentation[mid_slice_idx, mid_frame_idx, :, :]
                else:
                    # Format: (slices, H, W, frames)
                    mid_slice_idx = 0
                    mid_frame_idx = seg_shape[3] // 2
                    pred_slice = cine_segmentation[mid_slice_idx, :, :, mid_frame_idx]
            elif len(seg_shape) == 3:
                # (slices, H, W) - single frame
                mid_slice_idx = 0
                mid_frame_idx = 0
                pred_slice = cine_segmentation[mid_slice_idx]
            else:
                pred_slice = cine_segmentation
                mid_frame_idx = 0
            
            # Get corresponding input image
            slice_keys = list(cine_seq.slice_data.keys())
            slice_key = slice_keys[0] if slice_keys else "slice01"
            
            # Get the pixel array
            pixel_array = cine_seq.slice_data[slice_key]["pixel_array"]
            
            # Handle pixel array - it might be a list of frames
            if isinstance(pixel_array, list):
                if mid_frame_idx < len(pixel_array):
                    input_image = pixel_array[mid_frame_idx]
                else:
                    input_image = pixel_array[0]
            elif isinstance(pixel_array, np.ndarray):
                if len(pixel_array.shape) > 2:
                    if mid_frame_idx < pixel_array.shape[0]:
                        input_image = pixel_array[mid_frame_idx]
                    else:
                        input_image = pixel_array[0]
                else:
                    input_image = pixel_array
            else:
                input_image = np.array(pixel_array)
            
            # Create visualizations
            print(f"    Generating visualizations...")
            vis_path = OUTPUT_PATH / f"segmentation_frame{mid_frame_idx}.png"
            utils_viz.create_static_segmentation_plot(
                input_image, pred_slice, f"{chamber_type} ({model_name})", mid_frame_idx, vis_path
            )
            
            gif_path = OUTPUT_PATH / f"segmentation_animation.gif"
            utils_viz.create_segmentation_gif(
                cine_segmentation, pixel_array, f"{chamber_type} ({model_name})", 
                cine_seq.number_of_temporal_positions, gif_path
            )
            # ====================================================================
            print(f"\n  [6] Saving predictions for {chamber_type} ({model_name})...")
            output_seg_path = Path(f"{cine_dir}_segmentation_{model_name}")
            cine_seq.save_predictions(output_seg_path)
            print(f"    ✓ Predictions saved to: {output_seg_path}")
            
            # Compute volume curves
            print(f"    Computing volumes...")
            try:
                lv_vol_curve = cine_seq.compute_volume_curve(structure="lv")
                myo_vol_curve = cine_seq.compute_volume_curve(structure="myo")
                rv_vol_curve = cine_seq.compute_volume_curve(structure="rv")
                
                lv_ef = utils_viz.compute_ejection_fraction(lv_vol_curve)
                rv_ef = utils_viz.compute_ejection_fraction(rv_vol_curve)
                print(f"    LV EF: {lv_ef:.1f}%, RV EF: {rv_ef:.1f}%")
                
                vis_path = OUTPUT_PATH / f"volume_curves.png"
                utils_viz.plot_volume_curves(
                    lv_vol_curve, myo_vol_curve, rv_vol_curve, lv_ef, rv_ef, f"{chamber_type} ({model_name})", vis_path
                )
                    
            except Exception as e:
                print(f"    ERROR: {e}")
            
            # Visualize marker points
            try:
                if cine_seq.number_of_slices == 1:
                    cine_seq.base_slice_num = 0
                    cine_seq.mid_slice_num = 0
                    cine_seq.apex_slice_num = 0
                
                cine_seq._compute_marker_points()
                lv_centers = cine_seq.get_lv_center_points()
                rv_centers = cine_seq.get_rv_center_points()
                rv_insertions = cine_seq.get_rv_insertion_points()
                
                pred_slice_uint8 = pred_slice.astype(np.uint8)
                pred_masked = np.ma.masked_where(pred_slice_uint8 == 0, pred_slice_uint8)
                cmap_custom = utils_viz.get_custom_colormap()
                
                vis_path = OUTPUT_PATH / f"marker_points.png"
                utils_viz.plot_marker_points(
                    input_image, pred_masked, lv_centers, rv_centers, rv_insertions,
                    mid_frame_idx, f"{chamber_type} ({model_name})", cmap_custom, vis_path
                )
                
            except Exception as e:
                print(f"    WARNING: Marker points failed - {e}")
            
            print(f"    ✓ Complete")
            
        except Exception as e:
            print(f"    ERROR processing {chamber_type} with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

print("\n" + "="*80)
print("PIPELINE COMPLETE")
print("Results saved to chamber and model-specific directories:")
for chamber in CHAMBER_TYPES:
    chamber_name = chamber.replace('CINE_', '')  # e.g., "2CH"
    for model_name in MODELS.keys():
        result_dir = DATA_PATH / chamber / f"{chamber_name}_results_{model_name}"
        if result_dir.exists():
            print(f"  - {result_dir}")
print("="*80)



