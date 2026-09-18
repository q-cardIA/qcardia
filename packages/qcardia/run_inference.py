"""Segment cine DICOM views with a trained model.

Wraps CineSeries for use on a single study directory laid out as
<study>/CINE_SAX, <study>/CINE_4CH, ... Writes one NIfTI per view plus the
voxel spacing needed to turn masks into volumes.

    python run_inference.py --model wandb/my-run --data-dir study --views CINE_SAX

LA-conditioned models additionally need the long-axis view they were
conditioned on:

    python run_inference.py --model wandb/wla-run --data-dir study \
        --views CINE_SAX --lax-dir study/CINE_4CH
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

from qcardia.series import CineSeries
from qcardia.utils import get_data_directory

LA_CHAMBERS = ("CINE_2CH", "CINE_3CH", "CINE_4CH")
SAX_CHAMBERS = ("CINE_SAX",)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path,
                        help="Run directory holding the weights and config.")
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Study directory containing the view folders.")
    parser.add_argument("--views", nargs="+", default=None,
                        help="View folders to segment. Defaults to every CINE_*.")
    parser.add_argument("--as-chamber", default=None,
                        help="Treat --views as this chamber kind. Needed when a "
                             "view folder is a raw series name that does not "
                             "follow the CINE_* convention.")
    parser.add_argument("--name", default=None,
                        help="Output subdirectory name. Defaults to the model's.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--lax-dir", type=Path, default=None,
                        help="Long-axis DICOM directory for LA-conditioned models.")
    parser.add_argument("--lax-model", type=Path, default=None,
                        help="Weights used to pre-segment the long-axis view. "
                             "Defaults to the conditioned model's weights_path.")
    parser.add_argument("--lvis", action="store_true",
                        help="Also segment without LA conditioning, to measure "
                             "how much the LA vectors change the result.")
    parser.add_argument("--batch-size", type=int, default=50)
    return parser.parse_args()


def segment_view(args, chamber: str, output_base: Path) -> Path:
    """Segment one view and write its masks. Returns the output directory."""
    chamber_dir = args.data_dir / chamber
    if not chamber_dir.exists():
        raise FileNotFoundError(f"View folder does not exist: {chamber_dir}")

    dicom_dir = get_data_directory(chamber_dir)
    if dicom_dir is None:
        raise FileNotFoundError(f"No DICOM data inside {chamber_dir}")

    # A manual path override makes --views a raw series name, which carries no
    # information about whether it is short or long axis; --as-chamber does.
    is_sax = (args.as_chamber or chamber) in SAX_CHAMBERS

    print(f"  data     : {dicom_dir}")
    series = CineSeries(dicom_dir, batch_size=args.batch_size)
    print(f"  slices   : {series.number_of_slices}  "
          f"frames: {series.number_of_temporal_positions}")

    segmentation = series.predict_segmentation(
        args.model,
        lax_dicom_dir=args.lax_dir if is_sax else None,
        lax_model_path=args.lax_model if is_sax else None,
    )
    if segmentation.max() == 0:
        print("  WARNING  : segmentation is empty for every class")

    seg_dir = output_base / f"{chamber}_segmentation"
    series.save_predictions(seg_dir)

    pixdims = series._get_pixel_spacing().tolist()
    (seg_dir / "spacing.json").write_text(json.dumps({
        "pixel_spacing_mm": pixdims[:2],
        "slice_thickness_mm": pixdims[2],
        "n_slices": series.number_of_slices,
        "n_frames": series.number_of_temporal_positions,
    }, indent=2))
    print(f"  masks    : {seg_dir}")

    if args.lvis and is_sax and series._la_vectors is not None:
        print("  lvis     : re-segmenting with LA conditioning withheld")
        series.predict_segmentation(args.model, withhold_la_conditioning=True)
        series.save_predictions(output_base / f"{chamber}_no_la_segmentation")
        series._segmentation_prediction = segmentation

    return seg_dir


def main():
    args = parse_args()
    model_name = args.name or args.model.name
    output_base = args.output_dir / model_name

    views = args.views or sorted(
        d.name for d in args.data_dir.iterdir()
        if d.is_dir() and d.name.startswith("CINE_")
    )
    if not views:
        print(f"No CINE_* folders in {args.data_dir}; pass --views explicitly.")
        return 1

    failed = []
    for chamber in views:
        print(f"\n{model_name} -> {chamber}")
        try:
            segment_view(args, chamber, output_base)
        except Exception as error:
            traceback.print_exc()
            print(f"  FAILED   : {error}")
            failed.append(chamber)

    print(f"\n{len(views) - len(failed)}/{len(views)} views segmented "
          f"into {output_base}")
    if failed:
        print(f"failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
