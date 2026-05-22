import argparse
import os
import runpy
import sys
from pathlib import Path


def resolve_path(path):
    return Path(path).expanduser().resolve()


def run_script(script_path, script_args, cwd):
    print("\nRunning:")
    print("in-process " + " ".join([str(script_path), *[str(part) for part in script_args]]))
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        sys.argv = [str(script_path), *[str(part) for part in script_args]]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the sketch extrusion debug pipeline, then recover rank-0 cap endpoints "
            "in 3D from the generated debug output."
        )
    )
    parser.add_argument("image", nargs="?", default="Sketch_Test7.png")
    parser.add_argument("--output", default="result.png")
    parser.add_argument("--debug-dir", default="debug")
    parser.add_argument(
        "--calibration",
        default="blender_axonometric_dev_dataset/dev_camera_anchor_calibrationInput.json",
    )
    parser.add_argument(
        "--obj",
        default="scene_50.obj",
        help="OBJ file used by recover_rank_cap_endpoints_3d.py for cap-plane assignment.",
    )
    parser.add_argument("--trace-min-pixels", type=int, default=3)
    parser.add_argument("--straightness", type=float, default=0.65)
    parser.add_argument("--min-stroke-length", type=float, default=25)
    parser.add_argument("--parallel-angle-thresh", type=float, default=15)
    parser.add_argument(
        "--copy-side-iou-compare-percent",
        type=float,
        default=None,
        help=(
            "Forwarded to the extrusion script. Percent of longest side strokes to compare "
            "by cap-sweep IoU; count is ceil(n * percent / 100), minimum 2 when enabled. "
            "Omit to keep the extrusion script default."
        ),
    )
    parser.add_argument("--cap-loop-endpoint-tol", type=float, default=50)
    parser.add_argument("--split-corner-angle", type=float, default=30)
    parser.add_argument("--post-split-merge-gap", type=float, default=3)
    parser.add_argument("--post-split-merge-angle", type=float, default=12)
    parser.add_argument("--cap-loop-max-subset-size", type=int, default=15)
    parser.add_argument("--same-loop-endpoint-tol", type=float, default=5)
    parser.add_argument("--min-cap-total-arc", type=float, default=50)
    parser.add_argument("--split-segment-arc", type=float, default=30)
    parser.add_argument("--split-peak-min-distance", type=float, default=10)
    parser.add_argument("--split-optimize-max-iters", type=int, default=5)
    parser.add_argument(
        "--split-segment-arc30",
        action="store_const",
        const=30,
        dest="split_segment_arc",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--force-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward --force-parallel to the extrusion script by default.",
    )
    parser.add_argument(
        "--extrusion-script",
        default="extrusion_debug_caploop_percluster_capviz_subsetskip_fixed2_bbox_caps.py",
    )
    parser.add_argument(
        "--recover-script",
        default="recover_rank_cap_endpoints_3d.py",
    )
    parser.add_argument(
        "--skip-extrusion",
        action="store_true",
        help="Skip the first sketch/debug generation step.",
    )
    parser.add_argument(
        "--skip-recover",
        action="store_true",
        help="Skip the second 3D recovery step.",
    )
    parser.add_argument(
        "--reconstruct-blender",
        action="store_true",
        help="Pass --reconstruct-blender to recover_rank_cap_endpoints_3d.py.",
    )
    parser.add_argument(
        "--support-plane-debug-dir",
        default="debug/support_plane_fallback",
        help="Pass support-plane fallback debug output directory to recover_rank_cap_endpoints_3d.py.",
    )
    parser.add_argument(
        "--support-plane-polygon-tol",
        type=float,
        default=15.0,
        help="Pass support-plane fallback polygon tolerance to recover_rank_cap_endpoints_3d.py.",
    )
    args = parser.parse_args()

    cwd = Path(__file__).resolve().parent
    image = resolve_path(args.image)
    output = resolve_path(args.output)
    debug_dir = resolve_path(args.debug_dir)
    calibration = resolve_path(args.calibration)
    obj = resolve_path(args.obj)
    extrusion_script = resolve_path(args.extrusion_script)
    recover_script = resolve_path(args.recover_script)
    support_plane_debug_dir = resolve_path(args.support_plane_debug_dir)

    if not args.skip_extrusion:
        extrusion_args = [
            str(image),
            "--output",
            str(output),
            "--debug-dir",
            str(debug_dir),
            "--trace-min-pixels",
            str(args.trace_min_pixels),
            "--straightness",
            str(args.straightness),
            "--min-stroke-length",
            str(args.min_stroke_length),
            "--parallel-angle-thresh",
            str(args.parallel_angle_thresh),
            "--cap-loop-endpoint-tol",
            str(args.cap_loop_endpoint_tol),
            "--split-corner-angle",
            str(args.split_corner_angle),
            "--post-split-merge-gap",
            str(args.post_split_merge_gap),
            "--post-split-merge-angle",
            str(args.post_split_merge_angle),
            "--cap-loop-max-subset-size",
            str(args.cap_loop_max_subset_size),
            "--same-loop-endpoint-tol",
            str(args.same_loop_endpoint_tol),
            "--min-cap-total-arc",
            str(args.min_cap_total_arc),
            "--split-segment-arc",
            str(args.split_segment_arc),
            "--split-peak-min-distance",
            str(args.split_peak_min_distance),
            "--split-optimize-max-iters",
            str(args.split_optimize_max_iters),
        ]
        if args.copy_side_iou_compare_percent is not None:
            extrusion_args.extend([
                "--copy-side-iou-compare-percent",
                str(args.copy_side_iou_compare_percent),
            ])
        if args.force_parallel:
            extrusion_args.append("--force-parallel")
        run_script(extrusion_script, extrusion_args, cwd)

    if not args.skip_recover:
        recover_args = [
            "--debug-dir",
            str(debug_dir),
            "--calibration",
            str(calibration),
            "--obj",
            str(obj),
            "--support-plane-debug-dir",
            str(support_plane_debug_dir),
            "--support-plane-polygon-tol",
            str(args.support_plane_polygon_tol),
            "--copy-side-angle-thresh",
            str(args.parallel_angle_thresh),
        ]
        if args.reconstruct_blender:
            recover_args.append("--reconstruct-blender")
        run_script(recover_script, recover_args, cwd)


if __name__ == "__main__":
    main()
