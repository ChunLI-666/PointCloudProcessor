import open3d as o3d
import cv2 as cv2
import argparse 
import os 
import loguru as logger
import numpy as np


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
        self.pcd = o3d.read_point_cloud(pcd_path)
        self.crack_mask = None
        self.norm_mask = None
        self.distance_mask = None

    def add_crack_mask(self, crack_mask_path):
        self.crack_mask = cv2.imread(crack_mask_path, cv2.IMREAD_GRAYSCALE)
                
    def add_norm_mask(self, norm_mask):
        self.norm_mask = norm_mask

    def add_crack_mask(self, distance_mask):
        self.distance_mask = distance_mask


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
        
        self.filtered_pcd_folder = os.path.join(data_root_dir, "filtered_pcd")
        self.crack_mask_folder = os.path.join(data_root_dir, "mask_select")

        self.norm_mask_folder = os.path.join(data_root_dir, "norm_masks")
        self.distance_mask_folder = os.path.join(data_root_dir, "distance_mask")
        
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coefficients = distortion_coefficients
        
        self.read_data()


    def process(self):
        """
        Processes the Crack object.
        """
        self.generate_norm_masks()
        self.generate_distance_masks()
        self.save_results()

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
                logger.error(f"Mask image not found: {mask_path}")
                
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

            norm_mask = np.zeros_like(frame.crack_mask, dtype=np.float32)
            mask_height, mask_width = frame.crack_mask.shape
            for point, normal in zip(points_2d, normals):
                x, y = point
                if 0 <= x < mask_width and 0 <= y < mask_height and frame.crack_mask[y, x] > 0:
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
            mask_height, mask_width = frame.crack_mask.shape
            for point, xyz in zip(points_2d, points):
                x, y = point
                distance = np.linalg.norm(xyz)

                if 0 <= x < mask_width and 0 <= y < mask_height and frame.crack_mask[y, x] > 0:
                    distance_mask[y, x] = distance

            frame.add_distance_mask(distance_mask)

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


if __name__ == __main__():
    data_root_dir = "/home/charles/Documents/zhongnan/fastlio-color/test-offline-color/test-new-extrinsic"
    intrinsic_matrix = np.array(
        [[4818.200388954926, 0.0, 2032.4178620390019], 
         [0.0, 4819.10345841615, 1535.1895959282901], 
         [0.0, 0.0, 1.0]]
        )  # Replace fx, fy, cx, cy with actual values
    distortion_coefficients = np.array([0.003043514741045163, 0.06634739187544138, -0.000217681797407554, -0.0006654964142658197, 0])  # Replace with actual values

    crack = Crack(data_root_dir, intrinsic_matrix, distortion_coefficients)
    crack.process()



