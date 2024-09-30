import numpy as np
import logging
from scipy.spatial.transform import Rotation as R
from pose_interp import interpolate_poses
import os
import argparse
import open3d as open3d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Tlc = [
#         -0.9999487,
#         -0.0060765,
#         -0.0081089,
#         0.07317228523672217,
#         0.0080976,
#         0.0018884,
#         -0.9999654,
#         -0.048289129209371784,
#          0.0060916,
#         -0.9999797,
#         -0.0018391 ,
#         -0.06748855890332943,
#         0.0,
#         0.0,
#         0.0,
#         1.0
#     ]

# Tci = [
#         0.999916812873781,
#         0.0005159530022363122,
#         0.012888022534107549,
#         0.14019804643418302,
#         0.012897647385573308,
#         -0.02959566554827303,
#         -0.9994787377791847,
#         0.03680700838981143,
# 	-0.00013425445093047644,
# 	0.9995618191854141, 
# 	-0.029599858149797875, 
# 	-0.01400772611768411,
#         0.0,
#         0.0,
#         0.0,
#         1.0
#     ]


def read_scan_states(file_path):
    """
    读取scan_states文件并提取时间戳、位置和旋转数据.

    Args:
        file_path (str): 文件路径，包含scan_states文件的路径。

    Returns:
        list: 包含时间戳、位置和旋转数据的列表。每个元素是一个元组，包含三个值：
            - timestamp (float): 时间戳
            - position (numpy.ndarray): 位置数据，形状为(3,)
            - rotation (numpy.ndarray): 旋转数据，形状为(4,)，表示四元数 [x, y, z, w]

    Raises:
        FileNotFoundError: 如果文件路径无效或文件不存在，则引发此异常。
    """
    scan_states = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) < 8:
                    logging.error("Invalid line with insufficient data: %s", line)
                    continue
                timestamp = float(parts[0])
                position = np.array(parts[1:4], dtype=float)
                rotation = np.array(parts[4:8], dtype=float)  # quaternion [w, x, y, z]
                rotation = np.array([rotation[1],rotation[2],rotation[3],rotation[0]]) #[x, y, z, w]
                scan_states.append((timestamp, position, rotation))
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise

    # save_camera_odometry(scan_states, path_to_TUM_odometry)
    return scan_states

def gen_reconstruct_pose_plots(path_to_interpolated_vo_data, path_to_lio_data, path_to_pointcloud):
    """Generate reconstruct pose plots that contain visual odometry poses, lidar odometry poses, and point cloud."""
    vo_data = read_scan_states(path_to_interpolated_vo_data)
    lio_data = read_scan_states(path_to_lio_data)

    # Load point cloud
    o3d_pcd = open3d.io.read_point_cloud(path_to_pointcloud)
    # Scale the point cloud by a factor of 2 around its center
    scale_factor = 1.4
    o3d_pcd.points = open3d.utility.Vector3dVector(np.asarray(o3d_pcd.points) * scale_factor)

    # Create visual odometry trajectory
    vo_trajectory = np.array([state[1] for state in vo_data])
    vo_colors = np.array([[1, 0, 0] for _ in range(vo_trajectory.shape[0])])

    # Create lidar odometry trajectory
    lio_trajectory = np.array([state[1] for state in lio_data])
    lio_colors = np.array([[0, 1, 0] for _ in range(lio_trajectory.shape[0])])

    # Create Open3D point cloud for trajectories
    vo_points = open3d.geometry.PointCloud()
    vo_points.points = open3d.utility.Vector3dVector(vo_trajectory)
    vo_points.colors = open3d.utility.Vector3dVector(vo_colors)

    lio_points = open3d.geometry.PointCloud()
    lio_points.points = open3d.utility.Vector3dVector(lio_trajectory)
    lio_points.colors = open3d.utility.Vector3dVector(lio_colors)

    # Visualize
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(o3d_pcd)
    vis.add_geometry(vo_points)
    vis.add_geometry(lio_points)

    vis_step = 15  # Visualization step size
    # Add pose coordinate frames, every 10 frames
    for i, state in enumerate(vo_data):
        position = state[1]
        rotation = R.from_quat(state[2])  # Convert quaternion to rotation object

        if i % vis_step == 0:  # Draw coordinate frame every 10 frames
            # Define axes length
            axis_length = 0.1

            # Create lines for each axis
            axes = [
                rotation.apply([axis_length, 0, 0]),   # X-axis
                rotation.apply([0, axis_length, 0]),   # Y-axis
                rotation.apply([0, 0, axis_length])      # Z-axis
            ]

            # Create line segments for each axis
            for axis_color, axis in zip([[1, 0, 0], [0, 1, 0], [0, 0, 1]], axes):  # Red, Green, Blue
                line = open3d.geometry.LineSet()
                line.points = open3d.utility.Vector3dVector([position, position + axis])
                line.lines = open3d.utility.Vector2iVector([[0, 1]])
                line.paint_uniform_color(axis_color)
                vis.add_geometry(line)

    for i, state in enumerate(lio_data):
        position = state[1]
        rotation = R.from_quat(state[2])

        if i % vis_step == 0:  # Draw coordinate frame every 10 frames
            # Define axes length
            axis_length = 0.1

            # Create lines for each axis
            axes = [
                rotation.apply([axis_length, 0, 0]),   # X-axis
                rotation.apply([0, axis_length, 0]),   # Y-axis
                rotation.apply([0, 0, axis_length])      # Z-axis
            ]

            # Create line segments for each axis
            for axis_color, axis in zip([[1, 0, 0], [0, 1, 0], [0, 0, 1]], axes):  # Red, Green, Blue
                line = open3d.geometry.LineSet()
                line.points = open3d.utility.Vector3dVector([position, position + axis])
                line.lines = open3d.utility.Vector2iVector([[0, 1]])
                line.paint_uniform_color(axis_color)
                vis.add_geometry(line)

    # Set view parameters
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, -1, 0])
    view_control.set_zoom(0.5)

    # Capture and save image
    # Register key callback for saving images
    vis.register_key_action_callback(ord("S"), key_callback)

    vis.run()
    vis.destroy_window()

def key_callback(vis, action, mods):
    if action == open3d.visualization.KeyAction.KEY_DOWN and mods == ord('S'):
        save_visualization_image(vis)
        
def save_visualization_image(vis):
    """Save the visualization screen as an image when 's' key is pressed."""
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("visualization_image.png", do_render=True)
    print("Image saved as visualization_image.png")
    
def main(args):
    """
    Main function to process visual odometry data.

    Args:
        args (object): Command-line arguments.

    Returns:
        None
    """

    path_to_scan_states = os.path.join(args.root_dir, "visual_odom_in_lidar_ts.txt")
    # path_to_visual_odom_timestamp = os.path.join(args.root_dir, "visual_odom.txt")
    path_to_output = os.path.join(args.root_dir, "vo_interpolated_odom.txt")
    path_to_pointcloud = os.path.join(args.root_dir, "cloudInWorldWithRGBandMask(zoomout).pcd")

    gen_reconstruct_pose_plots(path_to_output, path_to_scan_states, path_to_pointcloud)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="Root directory path")
    args = parser.parse_args()

    main(args)
