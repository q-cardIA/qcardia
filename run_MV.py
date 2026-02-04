"""Multi-model, multi-view inference pipeline for cardiac cine segmentation.

Runs inference with different model types on appropriate chamber views:
- UNet-LA: Long-axis views (2CH, 3CH, 4CH)
- Context-aware Transformer: Short-axis views (SAX)
- Standard UNet: Short-axis views (SAX)

Outputs:
    - Segmentation masks (NIfTI format)
    - Static and animated visualizations
    - Volume curves and ejection fraction metrics
    - Anatomical marker point visualizations

Usage:
    python run_MV.py
"""

from pathlib import Path
import sys
import traceback
from matplotlib import pyplot as plt
from natsort import natsorted
import numpy as np

# Add qcardia source to path
qcardia_path = Path(__file__).parent / "src"
sys.path.insert(0, str(qcardia_path))

from qcardia.series import CineSeries
from qcardia import pipeline_utils
from qcardia.vis import (
    create_static_segmentation_plot,
    create_segmentation_gif,
    plot_volume_curves,
    plot_marker_points,
    compute_ejection_fraction,
    plot_all_sax_visualizations
)

# Configuration - Models mapped to their compatible chamber types
MODEL_CONFIG = {
    "UNet-LA": {
        "path": Path("/home/20203531/msc-thesis/msc-marcus/wandb/run-20250702_165723-sg792w9e_UNet-LA"),
        "views": ["CINE_2CH", "CINE_3CH", "CINE_4CH"],
        "type": "simple",
        "description": "Long-axis UNet model"
    },
    "Transformer-Context": {
        "path": Path("/home/20203531/msc-thesis/msc-marcus/wandb/run-context"),
        "views": ["CINE_SAX"],
        "type": "context-aware",
        "description": "Context-aware transformer for SAX"
    },
    "UNet-Baseline": {
        "path": Path("/home/20203531/msc-thesis/msc-marcus/wandb/run-20251022_002710-s4unsx34_nnUNet-baseline"),
        "views": ["CINE_SAX"],
        "type": "simple",
        "description": "Standard UNet baseline for SAX"
    }
}

DATA_PATH = Path("/home/20203531/msc-thesis/data/CardiSorted_QLGE04_")

print("="*80)
print("CARDIAC INFERENCE PIPELINE")
print("="*80)
print("\nModels configured:")
for model_name, config in MODEL_CONFIG.items():
    print(f"  {model_name}:")
    print(f"    Type: {config['type']}")
    print(f"    View(s): {', '.join(config['views'])}")
    print(f"    Description: {config['description']}")

# Verify paths
print("\nVerifying paths...")
all_paths_valid = True

for model_name, config in MODEL_CONFIG.items():
    model_path = config["path"]
    if not model_path.exists():
        print(f"  ✗ {model_name} path does not exist: {model_path}")
        all_paths_valid = False
    else:
        print(f"  {model_name}: {model_path}")

if not DATA_PATH.exists():
    print(f"  ✗ Data path does not exist: {DATA_PATH}")
    all_paths_valid = False
else:
    print(f"  Data path: {DATA_PATH}")

if not all_paths_valid:
    print("\nERROR: Some paths are invalid. Exiting.")
    sys.exit(1)

print("\nAll paths verified")

# Get all available views from data directory
available_views = [d.name for d in DATA_PATH.iterdir() 
                     if d.is_dir() and d.name.startswith("CINE_")]
print(f"\nAvailable views: {', '.join(available_views)}")

# Create model-chamber combinations to process
processing_queue = []
for model_name, config in MODEL_CONFIG.items():
    for chamber_type in config["views"]:
        if chamber_type in available_views:
            processing_queue.append({
                "model_name": model_name,
                "model_path": config["path"],
                "chamber_type": chamber_type,
                "model_type": config["type"]
            })

print(f"\nProcessing {len(processing_queue)} model-chamber combinations...")

# Process each combination
results_summary = []

for idx, task in enumerate(processing_queue, 1):
    model_name = task["model_name"]
    model_path = task["model_path"]
    chamber_type = task["chamber_type"]
    model_type = task["model_type"]
    
    print(f"\n{'='*80}")
    print(f"[{idx}/{len(processing_queue)}] {model_name} → {chamber_type}")
    print(f"{'='*80}")
    
    chamber_dir = DATA_PATH / chamber_type
    cine_dir = pipeline_utils.get_data_directory(chamber_dir)
    
    if cine_dir is None:
        print(f"  Skipping {chamber_type} (no data found)")
        continue
    print(f"  Data directory: {cine_dir}")
    print(f"  Model type: {model_type}")
    
    # Create output directory
    OUTPUT_PATH = Path(f"{cine_dir}_results_{model_name}")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        print(f"  [1/7] Loading {chamber_type} data...")
        cine_seq = CineSeries(cine_dir, batch_size=50)
        print(f"    Slices: {cine_seq.number_of_slices}")
        print(f"    Frames: {cine_seq.number_of_temporal_positions}")
        
        # Save data check visualization
        print(f"  [2/7] Creating data check visualization...")
        vis_path = OUTPUT_PATH / f"{chamber_type}_data_check.png"
        pipeline_utils.create_data_check_visualization(cine_seq, chamber_type, vis_path)
        print(f"    Saved: {vis_path.name}")
        
        # Run prediction
        print(f"  [3/7] Running inference...")
        cine_segmentation = cine_seq.predict_segmentation(model_path)
        
        print(f"    Output shape: {cine_segmentation.shape}")
        print(f"    Unique classes: {np.unique(cine_segmentation)}")
        
        if cine_segmentation.max() == 0:
            print(f"    WARNING: Empty prediction (all zeros)")
        
        # Prepare visualization - extract middle slice/frame
        print(f"  [4/7] Preparing visualizations...")
        mid_slice_idx, mid_frame_idx, pred_slice, input_image = pipeline_utils.get_middle_slice_and_frame(
            cine_seq, cine_segmentation
        )
        
        # Create static visualization
        print(f"  [5/7] Creating static segmentation plot...")
        vis_path = OUTPUT_PATH / f"segmentation_frame{mid_frame_idx}.png"
        create_static_segmentation_plot(
            input_image, pred_slice, 
            f"{chamber_type} ({model_name})", 
            mid_frame_idx, vis_path
        )
        print(f"    Saved: {vis_path.name}")
        
        # Create animation
        print(f"  [6/7] Creating segmentation animation...")
        seg_shape = cine_segmentation.shape
        
        # For SAX data, create one GIF per slice
        if chamber_type == "CINE_SAX" and len(seg_shape) == 4:
            print(f"    Creating {seg_shape[0]} GIFs (one per slice)...")
            for slice_idx in range(seg_shape[0]):
                gif_path = OUTPUT_PATH / f"segmentation_animation_slice{slice_idx:02d}.gif"
                try:
                    create_segmentation_gif(
                        cine_segmentation, cine_seq.slice_data, 
                        f"{chamber_type} ({model_name})", 
                        cine_seq.number_of_temporal_positions, gif_path,
                        slice_idx=slice_idx
                    )
                except Exception as e:
                    print(f"    Animation creation failed for slice {slice_idx}: {e}")
            print(f"    Saved: {seg_shape[0]} GIF animations (slice00-{seg_shape[0]-1:02d})")
        else:
            # Single GIF for LA views
            gif_path = OUTPUT_PATH / f"segmentation_animation.gif"
            try:
                create_segmentation_gif(
                    cine_segmentation, cine_seq.slice_data, 
                    f"{chamber_type} ({model_name})", 
                    cine_seq.number_of_temporal_positions, gif_path
                )
                print(f"    Saved: {gif_path.name}")
            except Exception as e:
                print(f"    Animation creation failed: {e}")
        
        # Save predictions
        print(f"  [7/7] Saving segmentation predictions...")
        output_seg_path = Path(f"{cine_dir}_segmentation_{model_name}")
        cine_seq.save_predictions(output_seg_path)
        print(f"    Saved to: {output_seg_path}")
        
        # SAX-specific comprehensive visualizations
        if chamber_type == "CINE_SAX" and len(seg_shape) == 4:
            print(f"  [Extra] Creating SAX-specific visualizations...")
            try:
                plot_all_sax_visualizations(
                    cine_segmentation, chamber_type, model_name, OUTPUT_PATH, fps=10
                )
            except Exception as e:
                print(f"    SAX visualization failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Compute volume curves (primarily for LA views)
        elif chamber_type in ["CINE_2CH", "CINE_3CH", "CINE_4CH"]:
            print(f"  [Extra] Computing volume curves...")
            try:
                lv_vol_curve = cine_seq.compute_volume_curve(structure="lv")
                myo_vol_curve = cine_seq.compute_volume_curve(structure="myo")
                rv_vol_curve = cine_seq.compute_volume_curve(structure="rv")
                
                lv_ef = compute_ejection_fraction(lv_vol_curve)
                rv_ef = compute_ejection_fraction(rv_vol_curve)
                print(f"    LV EF: {lv_ef:.1f}%, RV EF: {rv_ef:.1f}%")
                
                vis_path = OUTPUT_PATH / f"volume_curves.png"
                plot_volume_curves(
                    lv_vol_curve, myo_vol_curve, rv_vol_curve, 
                    lv_ef, rv_ef, f"{chamber_type} ({model_name})", vis_path
                )
                print(f"    Saved: {vis_path.name}")
            except Exception as e:
                print(f"    Volume computation failed: {e}")
        
        # Visualize marker points
        print(f"  [Extra] Computing marker points...")
        try:
            cine_seq._compute_marker_points()
            lv_centers = cine_seq.get_lv_center_points()
            rv_centers = cine_seq.get_rv_center_points()
            rv_insertions = cine_seq.get_rv_insertion_points()
            
            vis_path = OUTPUT_PATH / f"marker_points_frame{mid_frame_idx}.png"
            plot_marker_points(
                input_image, pred_slice,
                lv_centers, rv_centers, rv_insertions,
                f"{chamber_type} ({model_name})", mid_frame_idx, vis_path
            )
            print(f"    Saved: {vis_path.name}")
        except Exception as e:
            print(f"    Marker point visualization failed: {e}")
        
        print(f"\n  ✅ SUCCESS: {model_name} on {chamber_type}")
        results_summary.append({
            "model": model_name,
            "chamber": chamber_type,
            "status": "SUCCESS",
            "output": OUTPUT_PATH
        })
        
    except Exception as e:
        print(f"\n  ❌ ERROR processing {chamber_type} with {model_name}:")
        print(f"     {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        results_summary.append({
            "model": model_name,
            "chamber": chamber_type,
            "status": "FAILED",
            "error": str(e)
        })
        continue

# Print summary
print("\n" + "="*80)
print("PIPELINE COMPLETE - RESULTS SUMMARY")
print("="*80)

success_count = sum(1 for r in results_summary if r["status"] == "SUCCESS")
failed_count = sum(1 for r in results_summary if r["status"] == "FAILED")

print(f"\nTotal: {len(results_summary)} tasks")
print(f"✅ Success: {success_count}")
print(f"❌ Failed: {failed_count}")

print("\nSuccessful runs:")
for result in results_summary:
    if result["status"] == "SUCCESS":
        print(f"  {result['model']} → {result['chamber']}")
        print(f"    Output: {result['output']}")

if failed_count > 0:
    print("\nFailed runs:")
    for result in results_summary:
        if result["status"] == "FAILED":
            print(f"  ✗ {result['model']} → {result['chamber']}")
            print(f"    Error: {result['error']}")

print("\n" + "="*80)
print("All results saved to chamber and model-specific directories")
print("="*80)
