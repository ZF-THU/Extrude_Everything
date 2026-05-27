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
    parser.add_argument(
        "--side-straightness",
        type=float,
        default=0.85,
        help="Forwarded to the extrusion script. Hard prefilter for side-stroke clustering before PCA direction grouping.",
    )
    parser.add_argument(
        "--side-min-chord-px",
        type=float,
        default=25.0,
        help="Forwarded to the extrusion script. Minimum endpoint chord length for side-stroke clustering.",
    )
    parser.add_argument(
        "--side-line-p90-error-px",
        type=float,
        default=4.0,
        help="Forwarded to the extrusion script. Absolute p90 distance-to-PCA-line limit for side-stroke clustering.",
    )
    parser.add_argument(
        "--side-line-p90-error-ratio",
        type=float,
        default=0.035,
        help="Forwarded to the extrusion script. Relative p90 distance-to-PCA-line limit, multiplied by chord length.",
    )
    parser.add_argument(
        "--side-line-rms-error-px",
        type=float,
        default=2.5,
        help="Forwarded to the extrusion script. Absolute RMS distance-to-PCA-line limit for side-stroke clustering.",
    )
    parser.add_argument(
        "--side-line-rms-error-ratio",
        type=float,
        default=0.025,
        help="Forwarded to the extrusion script. Relative RMS distance-to-PCA-line limit, multiplied by chord length.",
    )
    parser.add_argument(
        "--side-chord-dev-ratio-max",
        type=float,
        default=0.08,
        help="Forwarded to the extrusion script. Maximum p90 endpoint-chord deviation divided by chord length for side-stroke clustering.",
    )
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
    parser.add_argument("--post-split-merge-protect-junction-radius", type=float, default=3)
    parser.add_argument("--cap-loop-max-subset-size", type=int, default=15)
    parser.add_argument("--same-loop-endpoint-tol", type=float, default=5)
    parser.add_argument(
        "--min-cap-bbox-area",
        type=int,
        default=0,
        help="Forwarded to the extrusion script. Reject cap candidates whose bbox area is below this threshold.",
    )
    parser.add_argument(
        "--cap-sweep-iou-stop-thresh",
        type=float,
        default=0.0,
        help="Forwarded to the extrusion script. Only stop cap search when a removal-depth round contains a cap sweep whose IoU against the input enclosed mask reaches this threshold.",
    )
    parser.add_argument(
        "--side-cap-connect-tol",
        type=float,
        default=20.0,
        help="Forwarded to the extrusion script. Require every final side stroke to have an endpoint near the selected cap; <=0 disables this gate.",
    )
    parser.add_argument("--min-cap-total-arc", type=float, default=50)
    parser.add_argument("--split-segment-arc", type=float, default=30)
    parser.add_argument("--split-peak-min-distance", type=float, default=10)
    parser.add_argument("--split-optimize-max-iters", type=int, default=5)
    parser.add_argument(
        "--skeleton-gap-tol",
        type=float,
        default=0.0,
        help="Forwarded to the extrusion script. Connect mutual-nearest skeleton endpoints within this tolerance and remove unmatched dangling branches before stroke tracing.",
    )
    parser.add_argument(
        "--skeleton-small-loop-bbox-area-thresh",
        type=float,
        default=0.0,
        help="Forwarded to the extrusion script. After skeleton gap connection, remove newly added edges that close loops below this bbox-area threshold. 0 disables this cleanup.",
    )
    parser.add_argument(
        "--skeleton-branch-prune-max-pixels",
        type=float,
        default=0.0,
        help="Forwarded to the extrusion script. Maximum traced pixels for deleting a 02c3 endpoint-started dangling branch; 0 uses the extrusion script automatic guard.",
    )
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
    parser.add_argument(
        "--curve-straightness-thresh",
        type=float,
        default=0.9,
        help="Pass the curved-stroke straightness threshold to recover_rank_cap_endpoints_3d.py.",
    )
    parser.add_argument(
        "--curve-resample-step-px",
        type=float,
        default=15.0,
        help="Pass the curved-stroke 2D resample step to recover_rank_cap_endpoints_3d.py.",
    )
    parser.add_argument(
        "--curve-min-chord-px",
        type=float,
        default=30.0,
        help="Pass the minimum chord length for 3D curved-stroke recovery.",
    )
    parser.add_argument(
        "--curve-min-p90-chord-dev-px",
        type=float,
        default=5.0,
        help="Pass the minimum p90 endpoint-chord deviation for 3D curved-stroke recovery.",
    )
    parser.add_argument(
        "--curve-min-dev-ratio",
        type=float,
        default=0.04,
        help="Pass the minimum p90 deviation/chord ratio for 3D curved-stroke recovery.",
    )
    parser.add_argument(
        "--curve-min-pca-rms-px",
        type=float,
        default=2.0,
        help="Pass the minimum PCA fitted-line RMS error for 3D curved-stroke recovery.",
    )
    parser.add_argument(
        "--blender-cap-debug-png",
        default="debug/blender_reconstruction_cap_2d.png",
        help="Pass the Blender-cap 2D PNG debug output path to recover_rank_cap_endpoints_3d.py.",
    )
    parser.add_argument(
        "--blender-cap-debug-json",
        default="debug/blender_reconstruction_cap_2d.json",
        help="Pass the Blender-cap JSON debug output path to recover_rank_cap_endpoints_3d.py.",
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
            "--side-straightness",
            str(args.side_straightness),
            "--side-min-chord-px",
            str(args.side_min_chord_px),
            "--side-line-p90-error-px",
            str(args.side_line_p90_error_px),
            "--side-line-p90-error-ratio",
            str(args.side_line_p90_error_ratio),
            "--side-line-rms-error-px",
            str(args.side_line_rms_error_px),
            "--side-line-rms-error-ratio",
            str(args.side_line_rms_error_ratio),
            "--side-chord-dev-ratio-max",
            str(args.side_chord_dev_ratio_max),
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
            "--post-split-merge-protect-junction-radius",
            str(args.post_split_merge_protect_junction_radius),
            "--cap-loop-max-subset-size",
            str(args.cap_loop_max_subset_size),
            "--same-loop-endpoint-tol",
            str(args.same_loop_endpoint_tol),
            "--min-cap-bbox-area",
            str(args.min_cap_bbox_area),
            "--cap-sweep-iou-stop-thresh",
            str(args.cap_sweep_iou_stop_thresh),
            "--side-cap-connect-tol",
            str(args.side_cap_connect_tol),
            "--min-cap-total-arc",
            str(args.min_cap_total_arc),
            "--split-segment-arc",
            str(args.split_segment_arc),
            "--split-peak-min-distance",
            str(args.split_peak_min_distance),
            "--split-optimize-max-iters",
            str(args.split_optimize_max_iters),
            "--skeleton-gap-tol",
            str(args.skeleton_gap_tol),
            "--skeleton-small-loop-bbox-area-thresh",
            str(args.skeleton_small_loop_bbox_area_thresh),
            "--skeleton-branch-prune-max-pixels",
            str(args.skeleton_branch_prune_max_pixels),
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
            "--curve-straightness-thresh",
            str(args.curve_straightness_thresh),
            "--curve-resample-step-px",
            str(args.curve_resample_step_px),
            "--curve-min-chord-px",
            str(args.curve_min_chord_px),
            "--curve-min-p90-chord-dev-px",
            str(args.curve_min_p90_chord_dev_px),
            "--curve-min-dev-ratio",
            str(args.curve_min_dev_ratio),
            "--curve-min-pca-rms-px",
            str(args.curve_min_pca_rms_px),
            "--blender-cap-debug-png",
            str(resolve_path(args.blender_cap_debug_png)),
            "--blender-cap-debug-json",
            str(resolve_path(args.blender_cap_debug_json)),
        ]
        if args.reconstruct_blender:
            recover_args.append("--reconstruct-blender")
        run_script(recover_script, recover_args, cwd)


if __name__ == "__main__":
    main()
