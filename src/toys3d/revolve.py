import argparse
import os
import numpy as np
import trimesh
import trimesh.transformations as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 命令行参数解析 (argparse)
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="3D扫描网格旋转对称轴与包线拟合及水密化重构工具 (命令行版)"
    )
    parser.add_argument(
        "-i", "--input", type=str, default="vase_scan.stl",
        help="输入STL格式网格文件路径 (默认: vase_scan.stl)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="vase_approximated_revolved.stl",
        help="输出生成的水密旋转体STL文件路径 (默认: vase_approximated_revolved.stl)"
    )
    parser.add_argument(
        "-s", "--slices", type=int, default=150,
        help="沿Z轴等高截面的切片数量 (默认: 150)"
    )
    return parser.parse_args()

# ==========================================
# 2. 核心算法：普吕克坐标与 SVD 拟合旋转对称轴
# ==========================================
def fit_rotational_axis_svd(mesh: trimesh.Trimesh):
    """通过面片中心与法线交汇，构建普吕克坐标线性方程组并使用 SVD 拟合轴线。"""
    centers = mesh.triangles_center
    normals = mesh.face_normals
    areas = mesh.area_faces

    # 平移至面积加权质心，避免空间力矩矩阵量纲失衡导致精度下降
    centroid = np.average(centers, axis=0, weights=areas)
    centered_c = centers - centroid

    # m_i = c_i × n_i
    moments = np.cross(centered_c, normals)
    weights = np.sqrt(areas)[:, np.newaxis]
    A = np.hstack([moments, normals]) * weights

    _, _, vh = np.linalg.svd(A, full_matrices=False)
    sol = vh[-1]

    u = sol[0:3] / np.linalg.norm(sol[0:3])
    m = sol[3:6] / np.linalg.norm(sol[0:3])
    p0 = np.cross(u, m) + centroid

    # 统一约定：确保轴线朝向Z轴正向
    if u[2] < 0:
        u = -u

    return p0, u

# ==========================================
# 3. 空间对齐与变换矩阵追踪
# ==========================================
def get_alignment_transforms(mesh: trimesh.Trimesh, p0: np.ndarray, u: np.ndarray):
    """
    计算将中心轴对齐到 Z 轴、底面最低点移至 Z=0 的正向变换矩阵 T_total，
    及其逆变换矩阵 T_inv (用于将重构结果变回原始网格坐标系进行叠加对比)。
    """
    target_dir = np.array([0.0, 0.0, 1.0])
    rot_axis = np.cross(u, target_dir)
    rot_angle = np.arccos(np.clip(np.dot(u, target_dir), -1.0, 1.0))

    # 1. 平移基点到原点
    T1 = tf.translation_matrix(-p0)
    
    # 2. 旋转对齐至 Z 轴
    T_rot = np.eye(4)
    if np.linalg.norm(rot_axis) > 1e-6:
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        T_rot[:3, :3] = tf.rotation_matrix(rot_angle, rot_axis)[:3, :3]

    # 临时计算对齐后的底面高度
    temp_mesh = mesh.copy()
    temp_mesh.apply_transform(T_rot @ T1)
    z_min = temp_mesh.bounds[0, 2]

    # 3. 沿 Z 轴平移使底面贴合原点 Z=0
    T2 = tf.translation_matrix([0.0, 0.0, -z_min])

    # 组合总正向变换与逆变换
    T_total = T2 @ T_rot @ T1
    T_inv = np.linalg.inv(T_total)

    aligned_mesh = mesh.copy()
    aligned_mesh.apply_transform(T_total)

    return aligned_mesh, T_total, T_inv

# ==========================================
# 4. XY 平面绕 Z 轴截面半径拟合 (无 Z 轴插值)
# ==========================================
def fit_slice_radii_xy(mesh: trimesh.Trimesh, num_slices: int = 150):
    """
    沿 Z 轴进行等步长切片。对每个截面层，在 XY 平面绕 Z 轴 (0,0)
    直接拟合切片轮廓点的最小二乘半径 R = np.mean(radii)。
    不进行任何 Z 轴方向的跨切片平滑拟合！
    """
    z_min, z_max = mesh.bounds[:, 2]
    # 剔除上下两端极薄的 0.5% 区域，避免瓶口封边或底部切割不平整带来的端面干扰
    z_start = z_min + (z_max - z_min) * 0.005
    z_end = z_max - (z_max - z_min) * 0.005
    z_sample = np.linspace(z_start, z_end, num_slices)

    valid_z = []
    valid_r = []

    for z in z_sample:
        lines = trimesh.intersections.mesh_plane(
            mesh, plane_normal=[0, 0, 1], plane_origin=[0, 0, z]
        )
        if len(lines) > 0:
            pts = lines.reshape(-1, 3)
            # 计算截面上所有交点到中心点 (0,0) 的 XY 平面径向距离
            radii = np.linalg.norm(pts[:, :2], axis=1)
            
            # 在 XY 平面绕 Z 轴的最小二乘半径，即该截面上径向距离的均值
            r_fit = np.mean(radii)
            
            if r_fit > 1e-3:  # 排除异常退化点
                valid_z.append(z)
                valid_r.append(r_fit)

    return np.array(valid_z), np.array(valid_r)

# ==========================================
# 5. 水密旋转体重构
# ==========================================
def generate_watertight_revolved_mesh(r_vals: np.ndarray, z_vals: np.ndarray, sections: int = 72):
    """直接使用各层面 discrete 截面拟合半径，构建侧面圆环并封闭顶底面，生成 100% 水密网格。"""
    vertices = []
    faces = []
    M = len(r_vals)
    N = sections

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    for i in range(M):
        r, z = r_vals[i], z_vals[i]
        for j in range(N):
            vertices.append([r * cos_a[j], r * sin_a[j], z])

    for i in range(M - 1):
        for j in range(N):
            next_j = (j + 1) % N
            v0, v1 = i * N + j, i * N + next_j
            v2, v3 = (i + 1) * N + next_j, (i + 1) * N + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    # 封底 (Z = z_vals[0], 法向朝下)
    bottom_center_idx = len(vertices)
    vertices.append([0.0, 0.0, z_vals[0]])
    for j in range(N):
        faces.append([bottom_center_idx, (j + 1) % N, j])

    # 封顶 (Z = z_vals[-1], 法向朝上)
    top_center_idx = len(vertices)
    vertices.append([0.0, 0.0, z_vals[-1]])
    top_start = (M - 1) * N
    for j in range(N):
        faces.append([top_center_idx, top_start + j, top_start + (j + 1) % N])

    revolved_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
    revolved_mesh.fix_normals()
    return revolved_mesh

# ==========================================
# 6. 多维可视化展示 (包含原始输入网格叠加)
# ==========================================
def visualize_results(raw_mesh, p0, u, z_vals, r_vals, revolved_mesh, T_inv):
    """
    左图显示各物理层的 XY 截面最小二乘半径拟合；
    右图在原始坐标系下将“原始输入网格”与“拟合出的3D旋转对称轴”叠加显示。
    """
    fig = plt.figure(figsize=(14, 6))

    # 子图 1: 各截面层在 XY 平面的圆半径拟合结果
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(r_vals, z_vals, 'o-', color='crimson', markersize=3, linewidth=1.5, label='XY-Plane Fitted Radius R(z)')
    ax1.set_title("Per-Slice XY-Plane Radius Fitting (No Z-Smoothing)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Fitted Radius in XY-Plane (mm)")
    ax1.set_ylabel("Slice Height Z (mm)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right')

    # 子图 2: 原始输入网格 + 3D 拟合轴线叠加展示 (世界坐标系)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # 降采样原始输入网格顶点以保正图表渲染流畅度
    sample_idx = np.random.choice(len(raw_mesh.vertices), size=min(4000, len(raw_mesh.vertices)), replace=False)
    pts = raw_mesh.vertices[sample_idx]
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='gray', alpha=0.3, s=2, label='Original Raw Mesh (Un-aligned)')

    # 绘制拟合出的对称轴空间直线 P0 + t * u
    t_min, t_max = -50.0, 200.0  # 延伸轴线长度以强化视觉效果
    axis_pts = np.array([p0 + t_min * u, p0 + t_max * u])
    ax2.plot(axis_pts[:, 0], axis_pts[:, 1], axis_pts[:, 2], color='blue', linewidth=3, label='Fitted Symmetry Axis')

    ax2.set_title("Original Input Mesh & Symmetry Axis Overlay", fontsize=12, fontweight='bold')
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_zlabel("Z (mm)")
    ax2.legend(loc='upper right')

    # 保持 3D 比例尺视觉等比
    max_range = np.array([pts[:, 0].max()-pts[:, 0].min(), pts[:, 1].max()-pts[:, 1].min(), pts[:, 2].max()-pts[:, 2].min()]).max() / 2.0
    mid_x, mid_y, mid_z = np.mean(pts[:, 0]), np.mean(pts[:, 1]), np.mean(pts[:, 2])
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

    # 交互式 3D 场景：将重构的水密旋转体变回原始坐标系，与原始网格完美重合对比
    print("\n[交互式3D对比] 正在启动 Trimesh 3D 渲染窗口（显示原始输入网格 vs 重构水密旋转体）...")
    try:
        revolved_world = revolved_mesh.copy()
        revolved_world.apply_transform(T_inv)
        
        # 赋予区分明显的半透明颜色
        raw_mesh.visual.face_colors = [160, 160, 160, 140]      # 银灰色半透明：原始网格
        revolved_world.visual.face_colors = [0, 140, 255, 200]  # 科技蓝：逼近旋转体
        
        scene = trimesh.Scene([raw_mesh, revolved_world])
        scene.show()
    except Exception as e:
        print(f"提示：当前图形环境不支持打开 Trimesh 交互窗口，已跳过 3D 渲染 ({e})。")

# ==========================================
# 7. 辅助功能：自动生成测试数据
# ==========================================
def create_dummy_vase_stl(filename):
    """当命令行输入的 STL 不存在时，自动创建一个倾斜摆放的拉胚花瓶网格。"""
    print(f"提示：输入文件 '{filename}' 不存在，正在自动生成测试用拉胚花瓶 STL ...")
    z = np.linspace(0, 160, 120)
    r = 28 + 16 * np.sin(z / 160 * 2 * np.pi) + 4 * np.cos(z / 160 * 4 * np.pi)
    
    dummy_mesh = generate_watertight_revolved_mesh(r, z, sections=60)
    # 施加任意空间平移与倾斜
    T = tf.euler_matrix(0.35, -0.25, 0.15)
    T[:3, 3] = [35.0, -20.0, 15.0]
    dummy_mesh.apply_transform(T)
    dummy_mesh.export(filename)

# ==========================================
# 主执行管道
# ==========================================
if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.input):
        create_dummy_vase_stl(args.input)

    print(f"[1/5] 读取输入网格: {args.input} ...")
    raw_mesh = trimesh.load(args.input)
    print(f"      原始网格规模: {len(raw_mesh.vertices)} 顶点, {len(raw_mesh.faces)} 面片")

    print("[2/5] 拟合空间旋转对称轴 (普吕克法线交汇 SVD) ...")
    p0, u = fit_rotational_axis_svd(raw_mesh)
    print(f"      轴线点 P0: {np.round(p0, 2)} | 方向 u: {np.round(u, 4)}")

    print("[3/5] 坐标对齐与变换矩阵追踪 ...")
    aligned_mesh, T_total, T_inv = get_alignment_transforms(raw_mesh, p0, u)

    print(f"[4/5] 沿 Z 轴进行 {args.slices} 层物理截面切片，在 XY 平面拟合每层截面半径 ...")
    z_vals, r_vals = fit_slice_radii_xy(aligned_mesh, num_slices=args.slices)

    print("[5/5] 基于 discrete 截面半径直接重构水密旋转体逼近模型 ...")
    revolved_mesh = generate_watertight_revolved_mesh(r_vals, z_vals, sections=72)
    print(f"      水密性检测 (Watertight) = {revolved_mesh.is_watertight} | 欧拉数 = {revolved_mesh.euler_number}")

    revolved_mesh.export(args.output)
    print(f"\n✅ 成功！逼近模型的标准旋转体已保存至: {os.path.abspath(args.output)}")

    print("正在生成原始网格与拟合结果的对比可视化...")
    visualize_results(raw_mesh, p0, u, z_vals, r_vals, revolved_mesh, T_inv)