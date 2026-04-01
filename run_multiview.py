"""Cardiac cine segmentation inference pipeline.

Runs a single model on one or more cardiac views and saves segmentation masks,
visualisations, volume curves, and anatomical marker plots.

Expected data directory structure (e.g. produced by cardisort):

    <data-dir>/
        CINE_SAX/           # Short-axis cine stack
            sa stack/       # DICOM files (subfolder name may vary)
        CINE_2CH/           # Two-chamber long-axis view
        CINE_3CH/           # Three-chamber long-axis view
        CINE_4CH/           # Four-chamber long-axis view

If your folders use different names, pass them explicitly with --views.

Model path should point to a WandB run directory that contains:

    files/
        config.yaml
        best_model.pt   (or last_model.pt)

Usage:
    # Run on all available CINE_* views found in data-dir:
    python run_multiview.py --model /path/to/wandb/run --data-dir /path/to/subject

    # Run on specific views only:
    python run_multiview.py --model /path/to/wandb/run --data-dir /path/to/subject \\
        --views CINE_SAX CINE_2CH

    # Custom output location and model label:
    python run_multiview.py --model /path/to/wandb/run --data-dir /path/to/subject \\
        --name MyModel --output-dir /path/to/results
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

# Chamber types that are treated as long-axis views (volume curves, single GIF)
LA_CHAMBERS = ["CINE_2CH", "CINE_3CH", "CINE_4CH"]
# Chamber types that are treated as short-axis views (per-slice GIFs, heatmaps)
SAX_CHAMBERS = ["CINE_SAX"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cardiac cine segmentation inference pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="DIR",
        help="WandB run directory containing files/config.yaml and files/best_model.pt.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        metavar="DIR",
        help="Subject directory containing CINE_SAX/, CINE_2CH/, etc. subfolders.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        metavar="VIEW",
        default=None,
        help=(
            "Which chamber folders to process, e.g. --views CINE_SAX CINE_2CH. "
            "Defaults to all CINE_* folders found in data-dir."
        ),
    )
    parser.add_argument(
        "--name",
        metavar="LABEL",
        default=None,
        help=(
            "Label used for output folder naming. "
            "Defaults to the model directory name."
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


def main():
    args = parse_args()

    MODEL_PATH = Path(args.model)
    DATA_PATH = Path(args.data_dir)
    MODEL_NAME = args.name or MODEL_PATH.name
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
    print(f"\nModel  : {MODEL_PATH}  [{MODEL_NAME}]")
    print(f"Data   : {DATA_PATH}")
    print(f"Output : {OUTPUT_BASE}")

    # -----------------------------------------------------------------------
    # Validate paths
    # -----------------------------------------------------------------------
    print("\nVerifying paths...")
    ok = True
    for label, path in [("model", MODEL_PATH), ("data-dir", DATA_PATH)]:
        if not path.exists():
            print(f"  ERROR: {label} path does not exist: {path}")
            ok = False
        else:
            print(f"  OK  {label}: {path}")
    if not ok:
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Discover views
    # -----------------------------------------------------------------------
    if args.views:
        views_to_run = args.views
    else:
        views_to_run = [d.name for d in DATA_PATH.iterdir()
                        if d.is_dir() and d.name.startswith("CINE_")]

    if not views_to_run:
        print("\nNo CINE_* folders found. Check --data-dir or pass --views explicitly.")
        sys.exit(0)

    print(f"\nViews to process: {', '.join(views_to_run)}")

    # -----------------------------------------------------------------------
    # Run inference for each view
    # -----------------------------------------------------------------------
    results_summary = []

    for idx, chamber_type in enumerate(views_to_run, 1):
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(views_to_run)}] {MODEL_NAME} -> {chamber_type}")
        print(f"{'=' * 80}")

        chamber_dir = DATA_PATH / chamber_type
        if not chamber_dir.exists():
            print(f"  Skipping: folder {chamber_dir} does not exist.")
            continue

        cine_dir = pipeline_utils.get_data_directory(chamber_dir)
        if cine_dir is None:
            print(f"  Skipping: no DICOM data found inside {chamber_dir}")
            continue

        print(f"  Data directory : {cine_dir}")

        OUTPUT_PATH = OUTPUT_BASE / MODEL_NAME / chamber_type
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        seg_output_path = OUTPUT_BASE / MODEL_NAME / f"{chamber_type}_segmentation"

        try:
            # 1 / 7  Load data
            print(f"  [1/7] Loading {chamber_type} data...")
            cine_seq = CineSeries(cine_dir, batch_size=50)
            print(f"    Slices: {cine_seq.number_of_slices}  Frames: {cine_seq.number_of_temporal_positions}")

            # 2 / 7  Data check visualisation
            print(f"  [2/7] Creating data check visualisation...")
            vis_path = OUTPUT_PATH / f"{chamber_type}_data_check.png"
            pipeline_utils.create_data_check_visualization(cine_seq, chamber_type, vis_path)
            print(f"    Saved: {vis_path.name}")

            # 3 / 7  Inference
            print(f"  [3/7] Running inference...")
            cine_segmentation = cine_seq.predict_segmentation(MODEL_PATH)
            print(f"    Output shape: {cine_segmentation.shape}  Classes: {np.unique(cine_segmentation)}")
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
                input_image, pred_slice,
                f"{chamber_type} ({MODEL_NAME})",
                mid_frame_idx, vis_path,
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
                            cine_segmentation, cine_seq.slice_data,
                            f"{chamber_type} ({MODEL_NAME})",
                            cine_seq.number_of_temporal_positions, gif_path,
                            slice_idx=slice_idx,
                        )
                    except Exception as e:
                        print(f"    Animation failed for slice {slice_idx}: {e}")
                print(f"    Saved: {n_slices} GIFs (slice00-{n_slices - 1:02d})")
            else:
                gif_path = OUTPUT_PATH / "segmentation_animation.gif"
                try:
                    create_segmentation_gif(
                        cine_segmentation, cine_seq.slice_data,
                        f"{chamber_type} ({MODEL_NAME})",
                        cine_seq.number_of_temporal_positions, gif_path,
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
                        cine_segmentation, chamber_type, MODEL_NAME, OUTPUT_PATH, fps=10
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
                        lv_vol_curve, myo_vol_curve, rv_vol_curve,
                        lv_ef, rv_ef, f"{chamber_type} ({MODEL_NAME})", vis_path,
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
                    input_image, pred_slice,
                    lv_centers, rv_centers, rv_insertions,
                    f"{chamber_type} ({MODEL_NAME})", mid_frame_idx, vis_path,
                )
                print(f"    Saved: {vis_path.name}")
            except Exception as e:
                print(f"    Marker point visualisation failed: {e}")

            print(f"\n  SUCCESS: {MODEL_NAME} on {chamber_type}")
            results_summary.append({"chamber": chamber_type, "status": "SUCCESS", "output": OUTPUT_PATH})

        except Exception as e:
            print(f"\n  ERROR processing {chamber_type}:")
            print(f"    {e}")
            traceback.print_exc()
            results_summary.append({"chamber": chamber_type, "status": "FAILED", "error": str(e)})

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    success = [r for r in results_summary if r["status"] == "SUCCESS"]
    failed = [r for r in results_summary if r["status"] == "FAILED"]

    print(f"\nTotal: {len(results_summary)}  |  OK: {len(success)}  |  Failed: {len(failed)}")
    for r in success:
        print(f"  OK   {r['chamber']}  ->  {r['output']}")
    for r in failed:
        print(f"  FAIL {r['chamber']}  :  {r['error']}")

    print(f"\nAll results written under: {OUTPUT_BASE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
