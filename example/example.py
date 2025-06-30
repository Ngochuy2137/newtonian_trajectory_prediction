from python_utils.plotter import Plotter
util_plotter = Plotter()

import numpy as np
import os
from newtonian_trajectory_prediction import predictor
from nae_static.utils.submodules.preprocess_utils.data_raw_reader import RoCatDataRawReader

def convert_from_world_coordinate(points3d_world):
    """Chuyển đổi từ hệ world sang hệ fit_parabola."""
    points3d_fit = np.empty_like(points3d_world)
    points3d_fit[:, 0] = points3d_world[:, 0]  # x_fit = x_world
    points3d_fit[:, 1] = -points3d_world[:, 2] # y_fit = -z_world (y-fit dương xuống)
    points3d_fit[:, 2] = points3d_world[:, 1]  # z_fit = y_world (depth)
    return points3d_fit

def revert_to_world_coordinate(points3d_fit):
    """Chuyển đổi từ hệ fit_parabola về hệ world."""
    points3d_world = np.empty_like(points3d_fit)
    points3d_world[:, 0] = points3d_fit[:, 0]  # x_world = x_fit
    points3d_world[:, 1] = points3d_fit[:, 2]  # y_world = z_fit (depth)
    points3d_world[:, 2] = -points3d_fit[:, 1] # z_world = -y_fit (z-up)
    return points3d_world

# =============
# # 1. Tạo quỹ đạo mẫu trong hệ z-up, dấu dương hướng lên
# g_world = np.array([0.0, 0.0, -9.81])    # gravity vector in z-up (m/s^2)
# p0_world = np.array([0.0, 0.0, 1.0])     # Vị trí ban đầu (x, y, z)
# v0_world = np.array([2.0, 0.0, 5.0])     # Vận tốc ban đầu (vx, vy, vz)

# # Thời điểm lấy mẫu
# times = np.linspace(0, 1.5, 20)          # 20 điểm từ t=0 đến t=1.5s

# # Tính quỹ đạo lý thuyết trong hệ world
# points3d_world = 0.5 * g_world * times[:, None]**2 + v0_world * times[:, None] + p0_world
# # nhiễu 
# noise = np.random.normal(0.0, 0.1, points3d_world.shape)  # Thêm nhiễu Gaussian
# points3d_fit = points3d_world + noise

# =============
nae_data_reader = RoCatDataRawReader()
this_data_dir = '/home/server-huynn/workspace/robot_catching_project/nae_paper_dataset/origin/trimmed_Bamboo_168'
print(f'Loading data from: {this_data_dir}')
data_test = nae_data_reader.read(data_folder=this_data_dir, file_format='npz')

all_data = []
for traj_idx in range(len(data_test)):
    traj_org = data_test[traj_idx]
    pos = traj_org['position']
    time = traj_org['time_step']
    traj_org = np.column_stack((time, pos))
    all_data.append(traj_org)

trajectory = np.array(all_data[0])
points3d_fit = trajectory[:, 1:]  # Lấy quỹ đạo đầu tiên trong dữ liệu

# 2. Convert sang hệ fit_parabola (x_fit, y_fit, z_fit)
# convert to z up
points3d_fit = np.column_stack((
    points3d_fit[:, 0],            # x_old → x_new
    -points3d_fit[:, 2],           # z_old → y_new = -z_old
    points3d_fit[:, 1],            # y_old → z_new
))
# =============

points3d_fit = convert_from_world_coordinate(points3d_fit)
times = trajectory[:, 0] - trajectory[0, 0] # Lấy thời gian tương ứng

# points3d_fit = points3d_fit[:38]
# times = times[:38]
print('points3d_fit[0:5]:', points3d_fit[0:5])
print('times[0:5]:', times[0:5])
print('points3d_fit shape:', points3d_fit.shape); input()
# 3. Chạy RANSAC nội suy parabol
p_fit, v_fit, inliers = predictor.fit_parabola_to_bboxes(
    points_np=points3d_fit[:38],
    times_np=times[:38],
    confidence=0.99,
    num_max_ransac_iter=100,
    inlier_threshold=0.2
)


# 5. (Tuỳ chọn) Convert kết quả ngược về hệ world
p0_est_world = np.array([p_fit[0], p_fit[2], -p_fit[1]])
v0_est_world = np.array([v_fit[0], v_fit[2], -v_fit[1]])

# sample 
pred_traj = predictor.sample_parabola(t=times, p=p_fit, v=v_fit)
print('pred_traj shape:', pred_traj.shape)
util_plotter.plot_trajectory_dataset_matplotlib(samples=[points3d_fit, pred_traj], rotate_data_whose_y_up=True); 



# 6. In kết quả
print("=== Kết quả (hệ fit_parabola) ===")
print(f"p_fit  = {p_fit}")
print(f"v_fit  = {v_fit}")
print(f"inliers: {inliers.sum()} / {len(inliers)}")

print("\n=== Kết quả chuyển về hệ world (z-up) ===")
print(f"p0_est_world = {p0_est_world}")
print(f"v0_est_world = {v0_est_world}")
