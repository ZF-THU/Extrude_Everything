# 项目说明

## 项目定位

这个仓库是一个用于“从二维手绘/渲染草图中恢复挤出体几何”的 Python 实验项目。核心工作流大致是：

1. 用 Blender 生成或准备一个轴测挤出体场景、OBJ、相机标定和草图 PNG。
2. 用 OpenCV 对草图做二值化、骨架化、笔画追踪、方向聚类、侧边/端面候选检测。
3. 根据调试输出和相机标定，将选中的端面端点恢复到 3D。
4. 可选地再用 Blender 生成恢复结果的 `.blend` 和渲染图用于检查。

项目目前不是包结构，没有 `pyproject.toml`、`requirements.txt` 或自动化测试目录；主要通过单个脚本和命令行参数串联运行。

## 当前主线脚本

- `combined_extrusion_recover_pipeline.py`
  - 当前最像总入口的脚本。
  - 先通过 `runpy.run_path()` 调用二维草图挤出调试脚本，再调用 3D 端点恢复脚本。
  - 默认二维脚本是 `extrusion_debug_caploop_percluster_capviz_subsetskip_fixed2_bbox_caps.py`。
  - 默认 3D 恢复脚本是 `recover_rank_cap_endpoints_3d.py`。
  - 支持 `--skip-extrusion`、`--skip-recover`、`--reconstruct-blender` 等开关。

- `extrusion_debug_caploop_percluster_capviz_subsetskip_fixed2_bbox_caps.py`
  - 当前最新、体量最大的二维草图分析/调试脚本。
  - 做预处理、Zhang-Suen 细化、骨架笔画追踪、角点/短片段处理、方向聚类、side/cap 候选选择、cap endpoint graph、cap sweep/bbox mask、IoU 排名等。
  - 会写出大量 `debug/` 下的 PNG 和 TXT 调试产物，例如：
    - `00_input.png`、`01_binary.png`、`02_skeleton.png`
    - `03a_raw_strokes.png`、`04_stroke_info.png`
    - `05b_direction_cluster_scores.*`
    - `cap_endpoint_graphs/cap_endpoint_graph_summary.txt`
    - `per_cluster_side_cap/`、`ranked_side_cap_iou/` 相关 overlay/mask
  - 参数很多，常用参数包括 `--force-parallel`、`--trace-min-pixels`、`--straightness`、`--min-stroke-length`、`--parallel-angle-thresh`、`--cap-loop-endpoint-tol`、`--copy-side-iou-compare-percent`。

- `recover_rank_cap_endpoints_3d.py`
  - 从二维调试输出、相机 anchor calibration 和 OBJ 中恢复端面端点 3D 坐标。
  - 默认从 `debug/cap_endpoint_graphs/cap_endpoint_graph_summary.txt` 读取端点图，也可以从 overlay 图中读取。
  - 默认输出 `debug/iou_rank00_cap_endpoints_3d.json`。
  - 会生成 OBJ face-id debug render、support-plane fallback debug 输出。
  - 加 `--reconstruct-blender` 后会调用 Blender，写出恢复 `.blend`、solid `.blend` 和渲染 PNG。

## 数据生成和辅助脚本

- `generate_random_extrusion_dev_dataset.py`
  - Blender 脚本，用于生成随机正交/轴测挤出体开发数据集。
  - 默认输出目录是 `blender_axonometric_dev_dataset/`，但 README 里常用 `--out-dir .\my_run`。
  - 输出通常包括 `.blend`、OBJ/MTL、`dev_camera_anchor_calibration.json`、`dev_camera_anchor_calibrationInput.json`、渲染图和手绘风格线稿。
  - 会调用 `render_sketch_cv.py` 做 OpenCV 后处理；默认尝试使用 conda 环境 `blender45torch`，也可用 `--sketch-python` 指定 Python。

- `render_sketch_cv.py`
  - 将 Blender 渲染 PNG 转成线稿/手绘风格草图。
  - 依赖 `cv2` 和 `numpy`。
  - 支持 thinning、stroke width、wobble、ink rough、invert 等参数。

- `generate_axonometric_dev_dataset.py`
  - 固定 L 形挤出体开发场景生成脚本。
  - 输出固定的 Blender 场景、标定和轴测渲染。

- `generate_axonometric_extrusion_scene.py`
  - 较早的固定 L 形挤出体参考场景脚本。
  - 默认输出目录名为 `blender_axonometric_reference`。

- `sketch_to_bbox_obj.py`、`sketch_to_bbox_obj_v2.py`、`sketch_to_bbox_obj_v3.py`
  - 另一条较早的路线：从盒状手绘草图恢复简单 3D bounding box OBJ。
  - `v3` 会保存 root candidate、方向聚类、cuboid overlay、`summary.json` 和 `recovered_bbox.obj`。

## 历史/实验文件

仓库中有大量 `extrusion_debug_*.py` 迭代版本，它们大多是在同一条 OpenCV 草图解析流程上不断调参和加调试功能：

- 较基础版本：`extrusion.py`、`extrusion_debug_cn.py`、`extrusion_debug_caploop.py`
- cluster/方向调试：`extrusion_debug_cluster_v3.py`、`extrusion_debug_caploop_clusterdebug.py`、`extrusion_debug_caploop_directiondebug.py`
- cap loop/候选验证：`extrusion_debug_caploop_capvalidated*.py`、`extrusion_debug_caploop_percluster_capviz*.py`
- 当前最应优先参考/修改的是带 `fixed2_bbox_caps` 后缀的版本，除非任务明确指定旧脚本。

`recover_rank_cap_endpoints_3d_backup_20260518_163212.py` 是恢复脚本备份；一般不要改备份文件。

## 常用命令

README 当前记录的主流程命令如下，按需调整输入图、OBJ 和参数：

```powershell
blender --background --python .\generate_random_extrusion_dev_dataset.py -- `
  --seed 12345 `
  --out-dir .\my_run `
  --cap-vertex-min 5 `
  --cap-vertex-max 12 `
  --extrusion-depth-min 2.8 `
  --extrusion-depth-max 4.8 `
  --sketch-black-thr 40 `
  --sketch-close-iter 1 `
  --sketch-stroke-width 3 `
  --sketch-handdraw `
  --sketch-invert `
  --sketch-wobble-amp 4 `
  --sketch-wobble-smooth 61 `
  --sketch-ink-rough 1
```

```powershell
python .\combined_extrusion_recover_pipeline.py .\my_run\dev_axonometric_render_handdraw.png --debug-dir debug --calibration .\my_run\dev_camera_anchor_calibrationInput.json --obj .\my_run\scene_51.obj --force-parallel --trace-min-pixels 3 --straightness 0.65 --min-stroke-length 25 --parallel-angle-thresh 15 --cap-loop-endpoint-tol 50 --split-corner-angle 25 --post-split-merge-gap 3 --post-split-merge-angle 12 --cap-loop-max-subset-size 15 --same-loop-endpoint-tol 5 --min-cap-total-arc 50 --split-segment-arc30 --reconstruct-blender --support-plane-debug-dir .\debug\support_plane_fallback --support-plane-polygon-tol 15
```

```powershell
python .\combined_extrusion_recover_pipeline.py .\LZ_Test_22.png --debug-dir debug --calibration .\my_run\dev_camera_anchor_calibrationInput.json --obj .\my_run\scene_51.obj --force-parallel --trace-min-pixels 3 --straightness 0.65 --min-stroke-length 25 --parallel-angle-thresh 15 --cap-loop-endpoint-tol 50 --split-corner-angle 25 --post-split-merge-gap 3 --post-split-merge-angle 12 --cap-loop-max-subset-size 15 --same-loop-endpoint-tol 5 --min-cap-total-arc 50 --split-segment-arc30 --reconstruct-blender --support-plane-debug-dir .\debug\support_plane_fallback --support-plane-polygon-tol 15 --copy-side-iou-compare-percent 60
```

## 依赖和运行环境

- 普通 Python 脚本主要依赖：
  - `numpy`
  - `opencv-python` / `cv2`
  - `matplotlib`，主要用于 `sketch_to_bbox_obj_v*.py` 的可视化输出
- Blender 脚本依赖 Blender 自带的：
  - `bpy`
  - `mathutils`
  - `bpy_extras.object_utils.world_to_camera_view`
- `recover_rank_cap_endpoints_3d.py --reconstruct-blender` 默认 Blender 路径是：
  - `C:\Program Files\Blender Foundation\Blender 3.6\blender.EXE`
- `generate_random_extrusion_dev_dataset.py` 中存在硬编码项目根目录：
  - `D:\26_THU\01_ZL\12_ForOwnUse_V4`
  - 移动仓库后需要检查这些默认路径。

## 重要输入/输出

- 示例输入图：
  - `Sketch_Test*.png`
  - `LZ_Test*.png`
  - `my_run/dev_axonometric_render_handdraw.png`
- 当前 `my_run/` 中有示例数据：
  - `dev_camera_anchor_calibrationInput.json`
  - `scene_50.obj`、`scene_51.obj`
  - `dev_axonometric_render.png`
  - `dev_axonometric_render_handdraw.png`
- 常见输出：
  - `result.png`
  - `debug/`
  - `debug/iou_rank00_cap_endpoints_3d.json`
  - `debug/iou_rank00_cap_endpoints_3d_reconstruction.blend`
  - `debug/iou_rank00_cap_endpoints_3d_reconstruction.png`

## Git/产物约定

`.gitignore` 已忽略大部分本地产物：

- Python cache 和虚拟环境
- `debug/`
- `result*.png`、各种 mask/overlay PNG
- `.blend`、`.obj`、`.mtl`
- `my_run/`
- `blender_axonometric_dev_dataset/`
- `random_extrusion_dev_dataset/`
- `*_backup_*.py`
- `*.env`

当前仓库的 `readme` 是无扩展名文件，主要记录临时命令；修改前注意它可能包含用户正在试的参数。

## 修改建议

- 优先按现有脚本风格小步修改，不要先重构成包结构。
- 涉及主流程时，优先检查：
  - `combined_extrusion_recover_pipeline.py`
  - `extrusion_debug_caploop_percluster_capviz_subsetskip_fixed2_bbox_caps.py`
  - `recover_rank_cap_endpoints_3d.py`
- 不要随意删除 `my_run/`、`debug/`、示例 PNG 或用户新生成的 OBJ/Blend；这些通常是调试上下文。
- 如果要改路径默认值，注意当前多个 Blender 脚本有 Windows 绝对路径。
- 若要增加新参数，最好同时在 `combined_extrusion_recover_pipeline.py` 中透传到对应子脚本。
- 当前没有自动测试；改完后建议至少用一个 README 中的 pipeline 命令跑通，并检查 `result.png`、`debug/` 关键文件和 3D JSON 是否生成。
