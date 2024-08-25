import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_gaussian_filter(image, kernel_size=(5, 5), sigma=1):
    """
    Apply Gaussian filter to an image to blur/smooth it.

    Args:
        image (2D array): Input image.
        kernel_size (tuple): The size of the Gaussian kernel (width, height).
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        2D array: Blurred image.
    """
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

def display_images(original, modified):
    """
    Display two images side by side for comparison.

    Args:
        original (2D array): Original image.
        modified (2D array): Modified image.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(modified, cmap='gray')
    plt.title('Image with Gaussian Filter')
    plt.axis('off')

    plt.show()

# Load the mask image (ensure this path points to your image)
image_path = '/sandbox/Documents/zhongnan/fastlio-color/20240715/_2024-07-15-17-12-54_reconstruct_3dcrack/edt_skeleton/1721034888.068144_skeleton.png'
mask_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded
if mask_image is None:
    print("Image not loaded. Please check the file path.")
else:
    # Apply Gaussian filter to the image
    blurred_mask = apply_gaussian_filter(mask_image, kernel_size=(5, 5), sigma=5)

    # Display both the original and the blurred images
    display_images(mask_image, blurred_mask)

# Load the mask image (ensure this path points to your image)
# mask_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Check if the image was loaded
# if mask_image is None:
#     print("Image not loaded. Please check the file path.")
# else:
#     # Apply Gaussian noise to the image
#     noisy_mask = add_gaussian_noise(mask_image, mean=0, std_dev=25)

#     # Display both the original and noisy images
#     display_images(mask_image, noisy_mask)
