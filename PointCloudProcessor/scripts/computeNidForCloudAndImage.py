import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_histogram(data, bins, range):
    hist, bin_edges = np.histogram(data, bins=bins, range=range)
    return hist, bin_edges

def save_histogram_as_image(hist, bin_edges, output_path, title):
    plt.figure()
    plt.plot(bin_edges[:-1], hist, color='blue')
    plt.title(title)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def print_point_cloud_channels(pcd):
    print("Available point cloud channels:")
    for key, value in pcd.point.items():
        print(f"Channel: {key}, Shape: {value.shape}")

# def main(pcd_path, image_path, pcd_hist_path, image_hist_path):
#     # Load PCD file
#     pcd = o3d.io.read_point_cloud(pcd_path, format='pcd')
#     # intensities = np.asarray(pcd.points)[:, 3]  # Assuming intensity is in the z-coordinate

#     # intensities = np.asarray(pcd.intensities)[:, 3]

#     # if pcd.has_intensity():
#     #     intensities = np.asarray(pcd.points)[:, 2]  # Assuming intensity is in the z-coordinate
#     # else:
#     #     intensities = np.asarray(pcd.point['intensity'])  # Assuming intensity is stored as a custom attribute

#     intensities = np.asarray(pcd.point['intensity'])
#     # Compute PCD histogram
#     pcd_hist, pcd_bin_edges = compute_histogram(intensities, bins=256, range=(0, 255))
#     save_histogram_as_image(pcd_hist, pcd_bin_edges, pcd_hist_path, 'PCD Intensity Histogram')

#     # Load image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         print(f"Failed to load image: {image_path}")
#         return

#     # Compute image histogram
#     image_hist, image_bin_edges = compute_histogram(image.flatten(), bins=256, range=(0, 255))
#     save_histogram_as_image(image_hist, image_bin_edges, image_hist_path, 'Image Intensity Histogram')

#     print(f"Histograms saved: {pcd_hist_path}, {image_hist_path}")

def main(pcd_path, image_path, pcd_hist_path, image_hist_path):
    # Load PCD file
    pcd = o3d.t.io.read_point_cloud(pcd_path, format='pcd')
    print_point_cloud_channels(pcd)
    # Extract intensity data
    if 'intensity' in pcd.point:
        intensities = pcd.point['intensity'].numpy()
    else:
        print(f"No intensity channel found in the point cloud: {pcd_path}")
        return

    # Compute PCD histogram
    pcd_hist, pcd_bin_edges = compute_histogram(intensities, bins=256, range=(0, 255))
    save_histogram_as_image(pcd_hist, pcd_bin_edges, pcd_hist_path, 'PCD Intensity Histogram')

    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Compute image histogram
    image_hist, image_bin_edges = compute_histogram(image.flatten(), bins=256, range=(0, 255))
    save_histogram_as_image(image_hist, image_bin_edges, image_hist_path, 'Image Intensity Histogram')

    print(f"Histograms saved: {pcd_hist_path}, {image_hist_path}")

if __name__ == "__main__":
    pcd_path = "/mnt/disk01/data/zhongnna/_2024-05-16-17-22-50-reconstruction/fast_lio_result/NID_compare/1715851370.262931_beforeNID.pcd"
    image_path = "/mnt/disk01/data/zhongnna/_2024-05-16-17-22-50-reconstruction/fast_lio_result/NID_compare/1715851370.262931.jpg"
    pcd_hist_path = "/mnt/disk01/data/zhongnna/_2024-05-16-17-22-50-reconstruction/fast_lio_result/NID_compare/pcd_histogram.jpg"
    image_hist_path = "/mnt/disk01/data/zhongnna/_2024-05-16-17-22-50-reconstruction/fast_lio_result/NID_compare/image_histogram.jpg"

    main(pcd_path, image_path, pcd_hist_path, image_hist_path)
