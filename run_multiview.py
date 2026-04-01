"""Multi-view cardiac cine segmentation inference pipeline.

Runs one or more models on the corresponding cardiac views and saves segmentation
masks, visualisations, volume curves, and anatomical marker plots.

Expected data directory structure (e.g. produced by cardisort):

    <data-dir>/
        CINE_SAX/           # Short-axis cine stack
            sa stack/       # DICOM files (subfolder name may vary)
        CINE_2CH/           # Two-chamber long-axis view
        CINE_3CH/           # Three-chamber long-axis view
        CINE_4CH/           # Four-chamber long-axis view

If your folders use different names (e.g. produced by a different sorter),
edit the LA_CHAMBERS and SAX_CHAMBERS lists below to match.

Model paths should point to a WandB run directory that contains:

    files/
        config.yaml
        best_model.pt   (or last_model.pt)

Usage:
    python run_multiview.py --data-dir /path/to/subject \\
        --la-model    /path/to/la/wandb/run \\
        --sax-model   /path/to/sax/wandb/run \\
        [--context-model /path/to/context/wandb/run] \\
        [--output-dir /path/to/output]

At least one of --la-model, --sax-model, or --context-model must be supplied.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

# Add qcardia source to path when running directly from this directory
qcardia_path = Path(__file__).parent / "src"
sys.path.insert(0, str(qcardia_path))

from qcardia.series import CineSeries
from qcardia import pipeline_utils
from qcardia.vis import (
    compute_ejection_fraction,
    create_segmentation_gif,
    create_static_segmentation_plot,
    plot_all_sax_visualizations,
    plot_marker_points,
    plot_volume_curves,
)

# ---------------------------------------------------------------------------
# Chamber folder name patterns
# ---------------------------------------------------------------------------
# These match the output produced by cardisort. Edit if your data uses
# different folder naming conventions.
LA_CHAMBERS = ["CINE_2CH", "CINE_3CH", "CINE_4CH"]
SAX_CHAMBERS = ["CINE_SAX"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-view cardiac cine segmentation inference pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        metavar="DIR",
        help=(
            "Path to the sorted subject directory containing CINE_SAX/, "
            "CINE_2CH/, CINE_3CH/, CINE_4CH/ subfolders."
        ),
    )
    parser.add_argument(
        "--la-model",
        metavar="DIR",
        help="WandB run directory for the long-axis (2CH/3CH/4CH) UNet model.",
    )
    parser.add_argument(
        "--sax-model",
        metavar="DIR",
        help="WandB run directory for the short-axis baseline UNet model.",
    )
    parser.add_argument(
        "--context-model",
        metavar="DIR",
        help=(
            "WandB run directory for the context-aware Transformer SAX model. "
            "Requires qcardia-models-dev (see README)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help=(
            "Directory to write all results. "
            "Defaults to <data-dir>_results/ next to the subject folder."
        ),
    )
    return parser.parse_args()


def build_model_config(args):
    """Assemble MODEL_CONFIG dict from parsed CLI arguments."""
    config = {}
    if args.la_model:
        config["UNet-LA"] = {
            "path": Path(args.la_model),
            "views": LA_CHAMBERS,
            "description": "Long-axis UNet model",
        }
    if args.sax_model:
        config["UNet-SAX"] = {
            "path": Path(args.sax_model),
            "views": SAX_CHAMBERS,
            "description": "Standard UNet baseline for SAX",
        }
    if args.context_model:
        config["Transformer-Context"] = {
            "path": Path(args.context_model),
            "views": SAX_CHAMBERS,
            "description": "Context-aware transformer for SAX",
        }
    return config


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    DATA_PATH = Path(args.data_dir)
    MODEL_CONFIG = build_model_config(args)

    if not MODEL_CONFIG:
        print("ERROR: Specify at least one of --la-model, --sax-model, or --context-model.")
        sys.exit(1)

    OUTPUT_BASE = (
        Path(args.output_dir)
        if args.output_dir
        else DATA_PATH.parent / f"{DATA_PATH.name}_results"
    )

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("CARDIAC INFERENCE PIPELINE")
    print("=" * 80)
    print(f"\nData:    {DATA_PATH}")
    print(f"Output:  {OUTPUT_BASE}")
    print("\nModels configured:")
    for model_name, cfg in MODEL_CONFIG.items():
        print(f"  {model_name} ({cfg['description']})")
        print(f"    Views : {', '.join(cfg['views'])}")
        print(f"    Path  : {cfg['path']}")

    # -----------------------------------------------------------------------
    # Validate paths
    # -----------------------------------------------------------------------
    print("\nVerifying paths...")
    all_valid = True

    if not DATA_PATH.exists():
        print(f"  ERROR: data-dir does not exist: {DATA_PATH}")
        all_valid = False
    else:
        print(f"  OK  data-dir: {DATA_PATH}")

    for model_name, cfg in MODEL_CONFIG.items():
        if not cfg["path"].exists():
            print(f"  ERROR: {model_name} path does not exist: {cfg['path']}")
            all_valid = False
        else:
            print(f"  OK  {model_name}: {cfg['path']}")

    if not all_valid:
        print("\nAborting: one or more paths are invalid.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Discover available views
    # -----------------------------------------------------------------------
    available_views = [d.name for d in DATA_PATH.iterdir() if d.is_dir()]
    cine_views = [v for v in available_views if v.startswith("CINE_")]
    print(f"\nAvailable views: {', '.join(cine_views) or '(none found)'}")

    # -----------------------------------------------------------------------
    # Build processing queue
    # -----------------------------------------------------------------------
    processing_queue = []
    for model_name, cfg in MODEL_CONFIG.items():
        for chamber_type in cfg["views"]:
            if chamber_type in available_views:
                processing_queue.append(
                    {
                        "model_name": model_name,
                        "model_path": cfg["path"],
                        "chamber_type": chamber_type,
                    }
                )
            else:
                print(f"  Skipping {model_name}/{chamber_type} (folder not found)")

    if not processing_queue:
        print("\nNothing to process. Check that your folder names match LA_CHAMBERS / SAX_CHAMBERS.")
        sys.exit(0)

    print(f"\nProcessing {len(processing_queue)} model-chamber combinations...")

    # -----------------------------------------------------------------------
    # Run inference for each combination
    # -----------------------------------------------------------------------
    results_summary = []

    for idx, task in enumerate(processing_queue, 1):
        model_name = task["model_name"]
        model_path = task["model_path"]
        chamber_type = task["chamber_type"]

        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(processing_queue)}] {model_name} -> {chamber_type}")
        print(f"{'=' * 80}")

        chamber_dir = DATA_PATH / chamber_type
        cine_dir = pipeline_utils.get_data_directory(chamber_dir)

        if cine_dir is None:
            print(f"  Skipping {chamber_type}: no DICOM data found inside {chamber_dir}")
            continue

        print(f"  Data directory : {cine_dir}")

        # Output directories for this model / chamber combination
        OUTPUT_PATH = OUTPUT_BASE / model_name / chamber_type
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        seg_output_path = OUTPUT_BASE / model_name / f"{chamber_type}_segmentation"

        try:
            # 1 / 7  Load data
            print(f"  [1/7] Loading {chamber_type} data...")
            cine_seq = CineSeries(cine_dir, batch_size=50)
            print(f"    Slices: {cine_seq.number_of_slices}")
            print(f"    Frames: {cine_seq.number_of_temporal_positions}")

            # 2 / 7  Data check visualisation
            print(f"  [2/7] Creating data check visualisation...")
            vis_path = OUTPUT_PATH / f"{chamber_type}_data_check.png"
            pipeline_utils.create_data_check_visualization(cine_seq, chamber_type, vis_path)
            print(f"    Saved: {vis_path.name}")

            # 3 / 7  Inference
            print(f"  [3/7] Running inference...")
            cine_segmentation = cine_seq.predict_segmentation(model_path)
            print(f"    Output shape   : {cine_segmentation.shape}")
            print(f"    Unique classes : {np.unique(cine_segmentation)}")
            if cine_segmentation.max() == 0:
                print(f"    WARNING: Prediction is all zeros — check model path and weights.")

            # 4 / 7  Prepare visualisation helpers
            print(f"  [4/7] Preparing visualisations...")
            mid_slice_idx, mid_frame_idx, pred_slice, input_image = (
                pipeline_utils.get_middle_slice_and_frame(cine_seq, cine_segmentation)
            )

            # 5 / 7  Static segmentation plot
            print(f"  [5/7] Creating static segmentation plot...")
            vis_path = OUTPUT_PATH / f"segmentation_frame{mid_frame_idx}.png"
            create_static_segmentation_plot(
                input_image,
                pred_slice,
                f"{chamber_type} ({model_name})",
                mid_frame_idx,
                vis_path,
            )
            print(f"    Saved: {vis_path.name}")

            # 6 / 7  Animated GIFs
            print(f"  [6/7] Creating segmentation animation...")
            seg_shape = cine_segmentation.shape

            if chamber_type in SAX_CHAMBERS and len(seg_shape) == 4:
                n_slices = seg_shape[0]
                print(f"    Creating {n_slices} GIFs (one per slice)...")
                for slice_idx in range(n_slices):
                    gif_path = OUTPUT_PATH / f"segmentation_animation_slice{slice_idx:02d}.gif"
                    try:
                        create_segmentation_gif(
                            cine_segmentation,
                            cine_seq.slice_data,
                            f"{chamber_type} ({model_name})",
                            cine_seq.number_of_temporal_positions,
                            gif_path,
                            slice_idx=slice_idx,
                        )
                    except Exception as e:
                        print(f"    Animation failed for slice {slice_idx}: {e}")
                print(f"    Saved: {n_slices} GIFs (slice00-{n_slices - 1:02d})")
            else:
                gif_path = OUTPUT_PATH / "segmentation_animation.gif"
                try:
                    create_segmentation_gif(
                        cine_segmentation,
                        cine_seq.slice_data,
                        f"{chamber_type} ({model_name})",
                        cine_seq.number_of_temporal_positions,
                        gif_path,
                    )
                    print(f"    Saved: {gif_path.name}")
                except Exception as e:
                    print(f"    Animation failed: {e}")

            # 7 / 7  Save NIfTI predictions
            print(f"  [7/7] Saving segmentation masks (NIfTI)...")
            cine_seq.save_predictions(seg_output_path)
            print(f"    Saved to: {seg_output_path}")

            # Extra: SAX volume heatmaps and 3-D animation
            if chamber_type in SAX_CHAMBERS and len(seg_shape) == 4:
                print(f"  [Extra] SAX-specific visualisations...")
                try:
                    plot_all_sax_visualizations(
                        cine_segmentation, chamber_type, model_name, OUTPUT_PATH, fps=10
                    )
                except Exception as e:
                    print(f"    SAX visualisation failed: {e}")
                    traceback.print_exc()

            # Extra: volume curves for LA views
            elif chamber_type in LA_CHAMBERS:
                print(f"  [Extra] Computing volume curves...")
                try:
                    lv_vol_curve = cine_seq.compute_volume_curve(structure="lv")
                    myo_vol_curve = cine_seq.compute_volume_curve(structure="myo")
                    rv_vol_curve = cine_seq.compute_volume_curve(structure="rv")

                    lv_ef = compute_ejection_fraction(lv_vol_curve)
                    rv_ef = compute_ejection_fraction(rv_vol_curve)
                    print(f"    LV EF: {lv_ef:.1f}%  RV EF: {rv_ef:.1f}%")

                    vis_path = OUTPUT_PATH / "volume_curves.png"
                    plot_volume_curves(
                        lv_vol_curve,
                        myo_vol_curve,
                        rv_vol_curve,
                        lv_ef,
                        rv_ef,
                        f"{chamber_type} ({model_name})",
                        vis_path,
                    )
                    print(f"    Saved: {vis_path.name}")
                except Exception as e:
                    print(f"    Volume computation failed: {e}")

            # Extra: anatomical marker points
            print(f"  [Extra] Computing anatomical marker points...")
            try:
                cine_seq._compute_marker_points()
                lv_centers = cine_seq.get_lv_center_points()
                rv_centers = cine_seq.get_rv_center_points()
                rv_insertions = cine_seq.get_rv_insertion_points()

                vis_path = OUTPUT_PATH / f"marker_points_frame{mid_frame_idx}.png"
                plot_marker_points(
                    input_image,
                    pred_slice,
                    lv_centers,
                    rv_centers,
                    rv_insertions,
                    f"{chamber_type} ({model_name})",
                    mid_frame_idx,
                    vis_path,
                )
                print(f"    Saved: {vis_path.name}")
            except Exception as e:
                print(f"    Marker point visualisation failed: {e}")

            print(f"\n  SUCCESS: {model_name} on {chamber_type}")
            results_summary.append(
                {
                    "model": model_name,
                    "chamber": chamber_type,
                    "status": "SUCCESS",
                    "output": OUTPUT_PATH,
                }
            )

        except Exception as e:
            print(f"\n  ERROR processing {chamber_type} with {model_name}:")
            print(f"    {e}")
            traceback.print_exc()
            results_summary.append(
                {
                    "model": model_name,
                    "chamber": chamber_type,
                    "status": "FAILED",
                    "error": str(e),
                }
            )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    success = [r for r in results_summary if r["status"] == "SUCCESS"]
    failed = [r for r in results_summary if r["status"] == "FAILED"]

    print(f"\nTotal: {len(results_summary)}  |  OK: {len(success)}  |  Failed: {len(failed)}")

    for result in success:
        print(f"  OK   {result['model']} / {result['chamber']}  ->  {result['output']}")

    for result in failed:
        print(f"  FAIL {result['model']} / {result['chamber']}  :  {result['error']}")

    print(f"\nAll results written under: {OUTPUT_BASE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
