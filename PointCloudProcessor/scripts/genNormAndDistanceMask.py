import open3d as o3d
import cv2 as cv2
import argparse 
import os 
# import loguru as logger
import logging
import numpy as np
# EDT
from scipy.ndimage import distance_transform_edt as distance
from scipy.ndimage import sobel

from skimage import segmentation as skimage_seg
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 后端
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
from tqdm import tqdm
import json 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrackDataFrame():
    """
    Represents a crack data frame.

    Attributes:
        frame_timestamp (int): The timestamp of the frame.
        pcd (PointCloud): The point cloud data.
        crack_mask (ndarray): The crack mask.
        norm_mask (ndarray): The normal mask.
        distance_mask (ndarray): The distance mask.
    """

    def __init__(self, timestamp, pcd_path):
        """
        Initializes a new instance of the CrackDataFrame class.

        Args:
            timestamp (int): The timestamp of the frame.
            pcd_path (str): The path to the point cloud data file.
        """
        self.frame_timestamp = timestamp
        self.pcd = o3d.io.read_point_cloud(pcd_path)
        self.crack_mask = None
        self.norm_mask = None
        self.distance_mask = None
        self.points_3d_mask = None
        self.distance_transformed = None
        self.skeleton = None
        self.skeleton_2d_pts = [] # dictionary of crack skeleton, [(pleft_1, pske_1, pright_1), (pleft_2, pske_2, pright_2), ...]
        # self.skeleton_2d_pts = [] # list of crack skeleton, [(pleft_1, pske_1, pright_1), (pleft_2, pske_2, pright_2), ...]
        self.skeleton_3d_info = [] # list of crack skeleton, [(pleft_1, pske_1, pright_1), (pleft_2, pske_2, pright_2), ...] 
        
    def add_crack_mask(self, crack_mask_path):
        self.crack_mask = cv2.imread(crack_mask_path, cv2.IMREAD_GRAYSCALE)
                
    def add_norm_mask(self, norm_mask):
        self.norm_mask = norm_mask

    def add_dist_mask(self, distance_mask):
        self.distance_mask = distance_mask
        
    def add_pts_3d_mask(self, points_3d_mask):
        self.points_3d_mask = points_3d_mask
        
    def add_distance_transform_EDT(self, distance_transformed):
        self.distance_transformed = distance_transformed
        
    def add_skeleton(self, skeleton):
        self.skeleton = skeleton


class Crack():
    """
    The Crack class is responsible for processing crack data, generating normal masks and distance masks,
    and saving the results. It contains methods for reading data, generating masks, and saving the masks.

    Attributes:
    - timestamps (list): A list of timestamps for each frame.
    - frames (list): A list of CrackDataFrame objects representing each frame.
    - filtered_pcd_folder (str): The path to the folder containing filtered point cloud data.
    - crack_mask_folder (str): The path to the folder containing crack mask images.
    - norm_mask_folder (str): The path to the folder where normal masks will be saved.
    - distance_mask_folder (str): The path to the folder where distance masks will be saved.
    - intrinsic_matrix (numpy.ndarray): The intrinsic matrix for camera calibration.
    - distortion_coefficients (numpy.ndarray): The distortion coefficients for camera calibration.

    Methods:
    - __init__(data_root_dir, intrinsic_matrix, distortion_coefficients): Initializes a Crack object.
    - process(): Processes the Crack object.
    - read_data(): Reads point cloud data and mask images.
    - read_point_cloud(): Reads point cloud data from the filtered_pcd_folder.
    - read_mask_images(): Reads mask images from the crack_mask_folder.
    - generate_norm_masks(): Generates normal masks for each crack mask.
    - generate_distance_masks(): Generates distance masks for each crack mask.
    - save_results(): Saves the generated masks.
    """

    def __init__(self, data_root_dir, intrinsic_matrix, distortion_coefficients):
        """
        Initializes a Crack object.

        Parameters:
        - data_root_dir (str): The root directory of the data.
        - intrinsic_matrix (numpy.ndarray): The intrinsic matrix for camera calibration.
        - distortion_coefficients (numpy.ndarray): The distortion coefficients for camera calibration.
        """
        self.timestamps = []        
        self.frames = []
        self.data_root_dir = data_root_dir
        self.filtered_pcd_folder = os.path.join(data_root_dir, "filtered_pcd")
        self.crack_mask_folder = os.path.join(data_root_dir, "mask_select")

        self.norm_mask_folder = os.path.join(data_root_dir, "norm_masks")
        self.distance_mask_folder = os.path.join(data_root_dir, "distance_mask")
        
        self.edt_skeleton_folder = os.path.join(data_root_dir, "edt_skeleton")
        if not os.path.exists(self.edt_skeleton_folder):
            os.makedirs(self.edt_skeleton_folder)

        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coefficients = distortion_coefficients
        
        self.read_data()

    def preprocess(self):
        """
        Preprocesses crack mask images to compute distance transform and skeleton from mask images.

        This method performs the following steps for each frame:
        1. Thresholds the crack mask image to obtain a binary image.
        2. Computes the distance transform of the binary image.
        3. Adds the distance transform to the frame.
        4. Computes and adds the skeleton of the input image.
        5. Saves the temporary results of the distance transform and skeleton.

        Returns:
            None
        """

        logger.info("Preprocessing crack mask images, to compute distance transform and skeleton from mask images")
        for frame in tqdm(self.frames):
            _, binary_image = cv2.threshold(frame.crack_mask, 0, 255, cv2.THRESH_BINARY)
            distance_transformed = distance(binary_image)
            frame.add_distance_transform_EDT(distance_transformed)

            # compute and add skeleton
            
            # Set global debug behavior to None (default), "print" (to file), 
            # or "plot" (Jupyter Notebooks or X11)
            pcv.params.debug = "None"
            skeleton = pcv.morphology.skeletonize(mask=binary_image)
            frame.add_skeleton(skeleton)

            # save temp results of EDT and skeleton
            edt_path = os.path.join(self.edt_skeleton_folder, f"{frame.frame_timestamp}_edt.png")
            edt_vis_path = os.path.join(self.edt_skeleton_folder, f"{frame.frame_timestamp}_edt_vis.png")

            skeleton_path = os.path.join(self.edt_skeleton_folder, f"{frame.frame_timestamp}_skeleton.png")

            cv2.imwrite(edt_path, frame.distance_transformed)
            cv2.imwrite(skeleton_path, frame.skeleton)
            
            # close interactive mode
            plt.ioff()
            plt.imshow(distance_transformed, cmap='viridis')
            plt.colorbar()
            plt.title('distance_transformed')
            plt.savefig(edt_vis_path)
            plt.close()
            
            logger.info(f"Saved edt result: {edt_path}")
            logger.info(f"Saved edt visualization result: {edt_vis_path}")
            logger.info(f"Saved skeleton result: {skeleton_path}")

    def process(self):
        """
        Processes the Crack object.
        """
        self.preprocess()
        self.generate_norm_masks()
        self.generate_distance_masks()
        self.generate_skeletion_direction()
        self.save_results()
        
    # def compute_skeleton_edge_pts(self, radius = 10):
    #     # Extract 2d skeleton pts and its left and right edge pts
    #     for frame in self.frames:
    #         frame.skeleton_2d_pts = []  # Initialize as an empty list
    #         frame.skeleton_direction_2d = []  # Initialize as an empty list

    #         skeleton_points = np.argwhere(frame.skeleton > 0)
    #         for point in skeleton_points:
    #             y, x = point
                
    #             distance_to_edge = frame.distance_transformed[y, x]
    #             direction_2d = self.compute_skeleton_direction(frame.skeleton, x, y, radius)
    #             # frame.skeleton_direction_2d.append({'skeleton_pt': (x, y), 'direction_2d': direction_2d.tolist()})        
    #             # left_edge, right_edge = self.find_edges(frame.crack_mask, frame.distance_transformed, x, y, distance_to_edge)
    #             left_edge_pt2d, right_edge_pt2d = self.find_edges_by_direction(frame.crack_mask, x, y, direction_2d)
    #             local_plane_3d = self.find_local_plane(frame.points_3d_mask, x, y, radius)

    #             # frame.skeleton_2d_pts.append((left_edge, (x, y), right_edge))
    #             if(left_edge_pt2d and right_edge_pt2d):
    #                 frame.skeleton_2d_pts.append({
    #                     'skeleton_pt': (x, y), 
    #                     'left_edge_pt2d': left_edge_pt2d, 
    #                     'right_edge_pt2d': right_edge_pt2d,
    #                     'direction_2d': direction_2d.tolist(),
    #                     'local_plane_3d': local_plane_3d
    #                 })

    def compute_skeleton_edge_pts(self, radius=10, sample_rate=0.001, search_radius=0.1, epsilon=1e-3):
        """Compute skeleton edge points for each frame.

        Args:
            radius (int, optional): Radius(in pixel) of the neighborhood for computing skeleton direction. Defaults to 10.
            sample_rate (float, optional): Sample rate (in meter) for searching 3D edge points. Defaults to 0.001.
            search_radius (int, optional): Search radius (in meter) for searching 3D edge points. Defaults to 0.1.
            epsilon (int, optional): Epsilon value (in meter) for searching 3D edge points. Defaults to 1e-3.
        """
        for frame in self.frames:
            frame.skeleton_2d_pts = []
            frame.skeleton_direction_2d = []

            # Find skeleton points in the frame
            skeleton_points = np.argwhere(frame.skeleton > 0)
            for point in skeleton_points:
                y, x = point

                # Compute skeleton direction
                direction_2d = self.compute_skeleton_direction(frame.skeleton, x, y, radius)

                # Find left and right edge points based on the skeleton direction
                left_edge_pt2d, right_edge_pt2d = self.find_edges_by_direction(frame.crack_mask, x, y, direction_2d)

                # Find local plane in the neighborhood
                local_plane_3d = self.find_local_plane(frame.points_3d_mask, x, y, radius)

                if left_edge_pt2d and right_edge_pt2d:
                    # Search for left edge point's 3D coordinates
                    left_edge_3d_pt = self.search_3d_edge_points(local_plane_3d, left_edge_pt2d, sample_rate, search_radius, epsilon)
                    # Search for right edge point's 3D coordinates
                    right_edge_3d_pt = self.search_3d_edge_points(local_plane_3d, right_edge_pt2d, sample_rate, search_radius, epsilon)

                    # Append the skeleton edge points to the frame
                    frame.skeleton_2d_pts.append({
                        'skeleton_pt': (x, y),
                        'left_edge_pt2d': left_edge_pt2d,
                        'right_edge_pt2d': right_edge_pt2d,
                        'direction_2d': direction_2d.tolist(),
                        'local_plane_3d': local_plane_3d,
                        'left_edge_3d_pt': left_edge_3d_pt,
                        'right_edge_3d_pt': right_edge_3d_pt
                    })

    def sample_3d_points_on_plane(plane_coefficients, center_3d_point, sample_rate=0.001, search_radius=0.1):
        """Sample 3D points on a plane.

        Args:
            plane_coefficients (tuple): Coefficients of the plane equation (a, b, c, d).
            center_3d_point (tuple): Center point of the plane (x0, y0, z0).
            sample_rate (float): Rate at which to sample the points.
            search_radius (float): Radius around the center point to sample points.

        Returns:
            numpy.ndarray: Array of sampled 3D points on the plane.
        """
        a, b, c, d = plane_coefficients
        x0, y0, z0 = center_3d_point

        # Calculate the range of sampling points
        x_min, x_max = x0 - search_radius, x0 + search_radius
        y_min, y_max = y0 - search_radius, y0 + search_radius

        # Generate the grid points
        x_range = np.arange(x_min, x_max, sample_rate)
        y_range = np.arange(y_min, y_max, sample_rate)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        z_grid = (-a * x_grid - b * y_grid - d) / c

        # Convert the grid points to a list of points
        sampled_points = np.vstack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel())).T

        return sampled_points

    def search_3d_edge_points(self, plane_coefficients, edge_2d_pt, sample_rate=0.001, search_radius=0.1, epsilon=0.001):
        """Searches for 3D edge points within a given plane.

        Args:
            plane_coefficients (list): The coefficients of the plane equation.
            edge_2d_pt (list): The 2D coordinates of the edge point.
            sample_rate (float): The rate at which to sample 3D points on the plane, Unit(m).
            search_radius (float): The radius within which to search for 3D points.
            epsilon (float): The maximum error allowed for a 3D point to be considered an edge point.

        Returns:
            list: The best 3D point found within the search criteria.
        """
        # Sample 3D points on the local plane
        sampled_points = sample_3d_points_on_plane(plane_coefficients, edge_2d_pt, sample_rate, search_radius)

        min_error = float('inf')
        best_3d_pt = None

        for point in sampled_points:
            # Project these neighboring points onto the 2D plane
            points_2d, _ = cv2.projectPoints(point.reshape(1, -1), rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)), 
                                            cameraMatrix=self.intrinsic_matrix, distCoeffs=self.distortion_coefficients)
            projected_2d = points_2d.squeeze().astype(int)
            
            # Calculate the error with respect to the given 2D edge point
            error = np.linalg.norm(projected_2d - np.array(edge_2d_pt))
            
            if error < min_error:
                min_error = error
                best_3d_pt = point.tolist()
            
            if error < epsilon:
                break
        
        return best_3d_pt
    
    def find_local_plane(self, points_3d_mask, x, y, radius=20):
        """Return local plane of the skeleton point, using the points in the radius.
        Project 2D points to 3D using the projection matrix.

        Args:
            points_3d_mask (np.ndarray): 3D points mask.
            x (int): x-coordinate of the skeleton point.
            y (int): y-coordinate of the skeleton point.
            radius (int): Radius to consider for local plane calculation.

        Returns:
            np.ndarray: Coefficients of the local plane (ax + by + cz + d = 0).
        """
        # Extract the points within the radius
        y_min, y_max = max(0, y - radius), min(points_3d_mask.shape[0], y + radius)
        x_min, x_max = max(0, x - radius), min(points_3d_mask.shape[1], x + radius)
        local_points_3d = points_3d_mask[y_min:y_max, x_min:x_max].reshape(-1, 3)

        # Remove zero points (assuming zero points are invalid)
        local_points_3d = local_points_3d[np.any(local_points_3d != 0, axis=1)]

        if local_points_3d.shape[0] < 3:
            return None  # Not enough points to define a plane

        # Fit a plane to the local points using SVD
        centroid = np.mean(local_points_3d, axis=0)
        centered_points = local_points_3d - centroid
        _, _, vh = np.linalg.svd(centered_points)
        normal = vh[2, :]

        # Plane equation: ax + by + cz + d = 0
        d = -np.dot(normal, centroid)
        plane_coefficients = np.append(normal, d)

        return plane_coefficients
                            
    # def compute_3d_crack_width(self, radius = 10):
    #     """Compute 3d crack width for each skeleton point
    #     For each frame:
    #     1. Extracts skeleton points.
    #     2. For each skeleton point:
    #         - Calculates the 2d distance to the nearest edge.
    #         - Finds the nearest left and right edge points using circular search.
    #         - Appends the results to skeleton_2d_pts.
    #     """
                    
    #     result_data = []  # Initialize the result data list
    #     # Extract 3d skeleton pts from distance mask
    #     for frame in self.frames:
    #         frame.skeleton_3d_info = []
    #         for skeleton_2d_pt in frame.skeleton_2d_pts:
    #             skeleton_pt = skeleton_2d_pt['skeleton_pt']
    #             left_edge_pt = skeleton_2d_pt['left_edge_pt']
    #             right_edge_pt = skeleton_2d_pt['right_edge_pt']
                
    #             # Ensure left_edge_pt and right_edge_pt are not None
    #             if left_edge_pt is not None and right_edge_pt is not None:
    #                 skeleton_3d_pt = frame.points_3d_mask[skeleton_pt[1], skeleton_pt[0]].tolist()
    #                 left_edge_3d_pt = frame.points_3d_mask[left_edge_pt[1], left_edge_pt[0]].tolist()
    #                 right_edge_3d_pt = frame.points_3d_mask[right_edge_pt[1], right_edge_pt[0]].tolist()
    #                 width = float(np.linalg.norm(np.array(left_edge_3d_pt) - np.array(right_edge_3d_pt)))
                    
    #                 frame.skeleton_3d_info.append({
    #                     'skeleton_3d_pt': skeleton_3d_pt, 
    #                     'left_edge_3d_pt': left_edge_3d_pt, 
    #                     'right_edge_3d_pt': right_edge_3d_pt,
    #                     'crack_width_3d': width
    #                 })
    #             else:
    #                 frame.skeleton_3d_info.append({
    #                     'skeleton_3d_pt': frame.points_3d_mask[skeleton_pt[1], skeleton_pt[0]].tolist(),
    #                     'left_edge_3d_pt': None,
    #                     'right_edge_3d_pt': None,
    #                     'crack_width_3d': None
    #                 })
    #         result_data.extend(frame.skeleton_3d_info)

    #     json_file = os.path.join(self.data_root_dir, 'crack_width_3d_results.json')
    #     with open(json_file, 'w') as jsonfile:
    #         json.dump(result_data, jsonfile, indent=4)

    def compute_3d_crack_width_via_searching(self, radius = 10, epsilon = 1e-3):
        """Compute 3d crack width for each skeleton point
        For each frame:
        1. Extracts skeleton points.
        2. For each skeleton point:
            - Calculates the 2d distance to the nearest edge.
            - Finds the nearest left and right edge points using circular search.
            - Appends the results to skeleton_2d_pts.
        """
                    
        result_data = []  # Initialize the result data list
        # Extract 3d skeleton pts from distance mask
        for frame in self.frames:
            frame.skeleton_3d_info = []
            for skeleton_2d_pt in frame.skeleton_2d_pts:
                # skeleton_3d_pt = skeleton_2d_pt['skeleton_3d_pt']
                # left_edge_pt2d = skeleton_2d_pt['left_edge_pt2d']
                # right_edge_pt2d = skeleton_2d_pt['right_edge_pt2d']
                left_edge_3d_pt = skeleton_2d_pt['left_edge_3d_pt']
                right_edge_3d_pt = skeleton_2d_pt['right_edge_3d_pt']
                
                # Ensure left_edge_pt and right_edge_pt are not None
                if left_edge_3d_pt is not None and right_edge_3d_pt is not None:
                    width = np.linalg.norm(np.array(left_edge_3d_pt) - np.array(right_edge_3d_pt))
                    
                    frame.skeleton_3d_info.append({
                        # 'skeleton_3d_pt': skeleton_3d_pt, 
                        'left_edge_3d_pt': left_edge_3d_pt, 
                        'right_edge_3d_pt': right_edge_3d_pt,
                        'crack_width_3d': width
                    })
                else:
                    frame.skeleton_3d_info.append({
                        # 'skeleton_3d_pt': frame.points_3d_mask[skeleton_pt[1], skeleton_pt[0]].tolist(),
                        'left_edge_3d_pt': None,
                        'right_edge_3d_pt': None,
                        'crack_width_3d': None
                    })
            result_data.extend(frame.skeleton_3d_info)

        json_file = os.path.join(self.data_root_dir, 'crack_width_3d_results.json')
        with open(json_file, 'w') as jsonfile:
            json.dump(result_data, jsonfile, indent=4)

    def find_edges(self, mask, distance_transformed, x, y, distance_to_edge):
        search_radius = int(np.ceil(distance_to_edge))
        min_distance_left = float('inf')
        min_distance_right = float('inf')
        left_edge = None
        right_edge = None
                
        for i in range(-search_radius, search_radius + 1):
            for j in range(-search_radius, search_radius + 1):
                dist = np.sqrt(i**2 + j**2)
                if dist <= distance_to_edge:
                    nx, ny = x + j, y + i
                    if 0 <= nx < mask.shape[1] and 0 <= ny < mask.shape[0] and mask[ny, nx] == 0:
                        if nx < x and dist < min_distance_left:
                            min_distance_left = dist
                            left_edge = (nx, ny)
                        elif nx > x and dist < min_distance_right:
                            min_distance_right = dist
                            right_edge = (nx, ny)
        return left_edge, right_edge
        
    def find_edges_by_direction(self, mask, x, y, direction):
        """
        Find the left and right edge points using the direction vector.

        Args:
            mask: 2D array of the crack mask.
            x: x coordinate of the skeleton point.
            y: y coordinate of the skeleton point.
            direction: Direction vector at the skeleton point.

        Returns:
            A tuple of the left edge point and the right edge point.
        """
        normal = np.array([-direction[1], direction[0]])
        left_edge = self.trace_edge(mask, x, y, normal)
        right_edge = self.trace_edge(mask, x, y, -normal)
        return left_edge, right_edge

    def trace_edge(self, mask, x, y, direction):
        """
        Trace along the given direction until an edge point is found.

        Args:
            mask: 2D array of the crack mask.
            x: x coordinate of the starting point.
            y: y coordinate of the starting point.
            direction: Direction vector to trace along.

        Returns:
            The coordinates of the edge point.
        """
        pos = np.array([x, y])
        while 0 <= pos[0] < mask.shape[1] and 0 <= pos[1] < mask.shape[0] and mask[int(pos[1]), int(pos[0])] > 0:
            pos += direction
        return tuple(pos.astype(int))        
        
    def read_data(self):
        """
        Reads point cloud data and mask images.

        1. read_point_cloud: Read point cloud from filtered_pcd_folder, each point cloud is named as "{timestamp}.pcd".
        Use open3d to read all point cloud and save them inside self.pcd_list .Besides, extract the timestamp for
        each point cloud and save into self.timestamps
        
        2. read_mask_images: Read mask images from crack_mask_folder, based on the self.timestamps list which extracted from read_point_cloud steps. 
        Each mask image is named as "{timestamp}.png". only select the mask which in the timestamps list  
        """
        self.read_point_cloud()
        self.read_mask_images()

    def read_point_cloud(self):
        """
        Reads point cloud data from the filtered_pcd_folder.

        Each point cloud file is named as "{timestamp}.pcd".
        Uses open3d to read all point clouds and saves them inside self.frames.
        Extracts the timestamp for each point cloud and saves it into self.timestamps.
        """
        for filename in os.listdir(self.filtered_pcd_folder):
            if filename.endswith(".pcd"):
                timestamp = float(filename[:-4])
                pcd_path = os.path.join(self.filtered_pcd_folder, filename)
                frame = CrackDataFrame(timestamp, pcd_path)
                
                self.timestamps.append(timestamp)
                self.frames.append(frame)

    def read_mask_images(self):
        """
        Reads mask images from the crack_mask_folder.

        Based on the self.timestamps list which was extracted from read_point_cloud method,
        reads mask images named as "{timestamp}.png" and adds them to the corresponding frames.
        """
        for frame in self.frames:
            mask_path = os.path.join(self.crack_mask_folder, f"{frame.frame_timestamp}.png")
            if os.path.exists(mask_path):
                frame.add_crack_mask(mask_path)
            else:
                logger.info(f"Mask image not found: {mask_path}")
                
    def generate_norm_masks(self):
        """
        Generates normal masks for each crack mask.

        For each frame, computes normals for each point in the point cloud.
        Projects 3D points to 2D using the camera intrinsic matrix.
        Creates a normal mask by iterating all pixels of the mask and setting the mask value to the corresponding normal value.
        """
        for frame in self.frames:
            pcd = frame.pcd
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=60))

            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)

            points_2d, _ = cv2.projectPoints(points, rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)), 
                                            cameraMatrix=self.intrinsic_matrix, distCoeffs=self.distortion_coefficients)
            points_2d = points_2d.squeeze().astype(int)
            
            # logger.info(f"frame.crack_mask.shape: {frame.crack_mask.shape}")
            # norm_mask = np.zeros_like(frame.crack_mask, dtype=np.float32)
            norm_mask = np.zeros((frame.crack_mask.shape[0], frame.crack_mask.shape[1], 3), dtype=np.float32)
            mask_height, mask_width = frame.crack_mask.shape
            for point, normal in zip(points_2d, normals):
                x, y = point
                if 0 <= x < mask_width and 0 <= y < mask_height and frame.crack_mask[y, x] > 0:
                    # logging.info(f"Normal: {normal}")
                    norm_mask[y, x] = normal

            frame.add_norm_mask(norm_mask)

    def generate_distance_masks(self):
        """
        Generates distance masks for each crack mask.

        For each frame, projects 3D points to 2D using the camera intrinsic matrix.
        Creates a distance mask by iterating all pixels of the mask and setting the mask value to the corresponding distance value.
        """
        for frame in self.frames:
            pcd = frame.pcd
            points = np.asarray(pcd.points)
            points_2d, _ = cv2.projectPoints(points, rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)), 
                                            cameraMatrix=self.intrinsic_matrix, distCoeffs=self.distortion_coefficients)
            points_2d = points_2d.squeeze().astype(int)

            distance_mask = np.zeros_like(frame.crack_mask, dtype=np.float32)
            points_3d_mask = np.zeros((frame.crack_mask.shape[0], frame.crack_mask.shape[1], 3), dtype=np.float32)
            
            mask_height, mask_width = frame.crack_mask.shape
            for point, xyz in zip(points_2d, points):
                x, y = point
                distance = np.linalg.norm(xyz)

                if 0 <= x < mask_width and 0 <= y < mask_height and frame.crack_mask[y, x] > 0:
                    distance_mask[y, x] = distance
                    points_3d_mask[y, x] = (xyz[0], xyz[1], xyz[2])

            frame.add_dist_mask(distance_mask)
            frame.add_pts_3d_mask(points_3d_mask)

    def compute_skeleton_direction(self, skeleton, x, y, radius):
        """
        Compute the direction of the skeleton at point (x, y).

        Args:
            skeleton: 2D array of the skeleton.
            x: x coordinate of the point.
            y: y coordinate of the point.
            radius: Radius of the neighbourhood to consider.

        Returns:
            A tuple representing the direction vector (dx, dy).
        """
        # Extract the neighbourhood
        y_min, y_max = max(0, y - radius), min(skeleton.shape[0], y + radius + 1)
        x_min, x_max = max(0, x - radius), min(skeleton.shape[1], x + radius + 1)
        neighbourhood = skeleton[y_min:y_max, x_min:x_max]

        # Compute gradients using Sobel operator
        gx = sobel(neighbourhood, axis=1)
        gy = sobel(neighbourhood, axis=0)

        # Average gradient to get direction
        direction_x = np.mean(gx)
        direction_y = np.mean(gy)

        return np.array([direction_x, direction_y])
    
    def generate_skeletion_direction(self, radius = 10):
        """
        Generate the skeleton direction for each frame.

        For each frame, extracts the skeleton points and its neighbours, and compute the direction of these points.
        Appends the results to skeleton_2d_pts.
        """
        for frame in self.frames:
            frame.skeleton_direction_2d = []  # Initialize as an empty list
            skeleton_points = np.argwhere(frame.skeleton > 0)
            for point in skeleton_points:
                y, x = point
                direction_2d = self.compute_skeleton_direction(frame.skeleton, x, y, radius)
                frame.skeleton_direction_2d.append({'skeleton_pt': (x, y), 'direction_2d': direction_2d.tolist()})
            
    def save_results(self):
        """
        Saves the generated masks.

        Creates the norm_mask_folder and distance_mask_folder if they don't exist.
        Saves the normal masks and distance masks for each frame.
        """
        if not os.path.exists(self.norm_mask_folder):
            os.makedirs(self.norm_mask_folder)
        if not os.path.exists(self.distance_mask_folder):
            os.makedirs(self.distance_mask_folder)

        for frame in self.frames:
            norm_mask_path = os.path.join(self.norm_mask_folder, f"{frame.frame_timestamp}_norm.png")
            distance_mask_path = os.path.join(self.distance_mask_folder, f"{frame.frame_timestamp}_distance.png")

            cv2.imwrite(norm_mask_path, frame.norm_mask)
            cv2.imwrite(distance_mask_path, frame.distance_mask)
            
            logger.info(f"Saved normal mask: {norm_mask_path}")
            logger.info(f"Saved distance mask: {distance_mask_path}")

def main():
    data_root_dir = "/sandbox/Documents/zhongnan/fastlio-color/20240715/_2024-07-15-17-12-54_reconstruct_3dcrack"
    intrinsic_matrix = np.array(
        [[4818.200388954926, 0.0, 2032.4178620390019], 
        [0.0, 4819.10345841615, 1535.1895959282901], 
        [0.0, 0.0, 1.0]]
        )  # Replace fx, fy, cx, cy with actual values
    distortion_coefficients = np.array([0.003043514741045163, 0.06634739187544138, -0.000217681797407554, -0.0006654964142658197, 0])  # Replace with actual values

    crack = Crack(data_root_dir, intrinsic_matrix, distortion_coefficients)
    crack.process()
    crack.compute_skeleton_edge_pts()
    crack.compute_3d_crack_width_via_searching()
    
if __name__ == '__main__':
    main()



