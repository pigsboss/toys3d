"""
visualization 模块测试。

覆盖：
1. add_axes_to_scene：场景中添加坐标轴几何体；
2. make_double_sided：双面渲染面数翻倍；
3. set_face_alpha：透明度设置；
4. add_wireframe_to_scene：线框添加；
5. build_defect_visualization 与 build_reliable_visualization 的颜色检查；
6. load_proxy_mesh 与 add_proxy_overlay_to_scene 的基本流程。
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

# 确保可以导入 src/toys3d
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from toys3d.visualization import (
    add_axes_to_scene,
    make_double_sided,
    set_face_alpha,
    add_wireframe_to_scene,
    build_defect_visualization,
    build_reliable_visualization,
    load_proxy_mesh,
    add_proxy_overlay_to_scene,
)


@pytest.fixture
def box_mesh():
    """1x1x1 的标准立方体网格。"""
    return trimesh.creation.box(extents=[1.0, 1.0, 1.0])


def test_add_axes_to_scene(box_mesh):
    """坐标轴场景应包含 6 个几何对象：3 条轴 + 3 个箭头球。"""
    scene = trimesh.Scene()
    origin = box_mesh.bounding_box.centroid
    add_axes_to_scene(
        scene,
        origin=origin,
        u_x=np.array([1.0, 0.0, 0.0]),
        u_y=np.array([0.0, 1.0, 0.0]),
        u_z=np.array([0.0, 0.0, 1.0]),
        length=0.5,
        radius=0.01,
    )

    assert len(scene.geometry) == 6


def test_make_double_sided(box_mesh):
    """双面渲染后面片数量应翻倍。"""
    original_face_count = len(box_mesh.faces)
    double = make_double_sided(box_mesh)

    assert len(double.faces) == 2 * original_face_count


def test_set_face_alpha(box_mesh):
    """设置 alpha 后，所有面片颜色的 alpha 通道应更新。"""
    alpha = 0.3
    box_mesh.visual.face_colors = np.full((len(box_mesh.faces), 4), [200, 200, 200, 255], dtype=np.uint8)
    set_face_alpha(box_mesh, alpha)

    expected_alpha = int(alpha * 255)
    assert np.all(box_mesh.visual.face_colors[:, 3] == expected_alpha)


def test_add_wireframe_to_scene(box_mesh):
    """添加线框后，场景中几何体数量应等于唯一边数。"""
    scene = trimesh.Scene()
    add_wireframe_to_scene(scene, box_mesh, color=None, radius=0.005)

    assert len(scene.geometry) == len(box_mesh.edges_unique)


def test_build_defect_visualization(box_mesh):
    """缺陷可视化应根据掩码着色。"""
    n_faces = len(box_mesh.faces)
    open_mask = np.zeros(n_faces, dtype=bool)
    nonmanifold_mask = np.zeros(n_faces, dtype=bool)
    # 假设第一个面片是开放缺陷面
    open_mask[0] = True
    open_mask[1] = True
    nonmanifold_mask[1] = True  # 同时具有两种缺陷

    vis = build_defect_visualization(box_mesh, open_mask, nonmanifold_mask)

    colors = vis.visual.face_colors
    assert colors.shape == (n_faces, 4)

    # 同时缺陷面应为橙色 [255, 128, 0, 255]
    np.testing.assert_array_equal(colors[1], np.array([255, 128, 0, 255], dtype=np.uint8))
    # 仅开放面应为黄色 [255, 220, 0, 255]
    np.testing.assert_array_equal(colors[0], np.array([255, 220, 0, 255], dtype=np.uint8))


def test_build_reliable_visualization(box_mesh):
    """可靠面可视化应根据拓扑距离着色。"""
    n_faces = len(box_mesh.faces)
    distances = np.arange(n_faces, dtype=np.int32)
    min_distance = 2

    vis = build_reliable_visualization(box_mesh, distances, min_distance)

    colors = vis.visual.face_colors
    # 距离 >= 2 应为绿色，距离 0 或 1 应为其他颜色
    assert np.all(colors[distances >= min_distance] == [0, 200, 0, 255])
    assert np.all(colors[distances == 0] == [255, 0, 0, 255])  # 缺陷面红色
    assert np.all(colors[distances == 1] == [255, 220, 0, 255])  # 中间面黄色


def test_load_proxy_mesh(tmp_path, box_mesh):
    """从磁盘加载代理网格。"""
    proxy_path = tmp_path / "proxy.stl"
    box_mesh.export(proxy_path)

    proxy = load_proxy_mesh(str(proxy_path))
    assert proxy is not None
    assert len(proxy.faces) == len(box_mesh.faces)


def test_add_proxy_overlay_to_scene(box_mesh):
    """叠加代理网格应增加场景几何体。"""
    scene = trimesh.Scene()
    proxy = box_mesh.copy()
    add_proxy_overlay_to_scene(
        scene,
        proxy,
        color=[255, 0, 0],
        alpha=0.6,
        double_sided=False,
    )

    assert len(scene.geometry) == 1
    geom = list(scene.geometry.values())[0]
    assert len(geom.faces) == len(box_mesh.faces)

    # 检查颜色
    alpha_uint8 = int(0.6 * 255)
    expected_color = np.array([255, 0, 0, alpha_uint8], dtype=np.uint8)
    assert np.all(geom.visual.face_colors == expected_color)
