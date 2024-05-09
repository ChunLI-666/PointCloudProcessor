#include "PointCloudProcessor.hpp"
#include <iostream>
#include <string>

// Function to convert command line string to bool
std::optional<bool> parseBool(const char* arg) {
    std::string argStr(arg);
    if (argStr == "1" || argStr == "true") {
        return true;
    } else if (argStr == "0" || argStr == "false") {
        return false;
    }
    return std::nullopt; // Invalid boolean value
}


int main(int argc, char** argv) {
    // Expect at least 4 arguments, and at most 6
    if (argc < 5 || argc > 7) {
        std::cerr << "Usage: " << argv[0] << " <point_cloud_path> <odometry_path> <images_folder> [<mask_image_folder>] [<output_path>] [<enableMLS>]" << std::endl;
        std::cerr << "  point_cloud_path        Path to the point cloud data file." << std::endl;
        std::cerr << "  odometry_path           Path to odometry data file." << std::endl;
        std::cerr << "  images_folder           Path to directory containing images." << std::endl;
        std::cerr << "  mask_image_folder       (Optional) Path to directory for segmented images. Defaults to images_folder if not provided." << std::endl;
        std::cerr << "  output_path             (Optional) Path to save processed output. Defaults to current directory if not provided." << std::endl;
        std::cerr << "  enableMLS               (Optional) Enable MLS smoothing. Accepts true/false or 1/0. Defaults to false if not provided." << std::endl;
        return -1;
    }

    // std::string pointCloudPath = argv[1];
    // std::string odometryPath = argv[2];
    // std::string imagesFolder = argv[3];
    // std::string maskImageFolder = argv[4];
    // std::string outputPath = argv[5];
    // bool enableMLS = std::stoi(argv[6]);

    std::string pointCloudPath = argv[1];
    std::string odometryPath = argv[2];
    std::string imagesFolder = argv[3];
    std::optional<std::string> maskImageFolder = (argc > 4) ? std::optional<std::string>(argv[4]) : std::nullopt;
    std::optional<std::string> outputPath = (argc > 5) ? std::optional<std::string>(argv[5]) : std::nullopt;
    std::optional<bool> enableMLS = (argc > 6) ? parseBool(argv[6]) : std::nullopt;

    if (enableMLS == std::nullopt && argc > 6) {
        std::cerr << "Invalid value for enableMLS. Use 'true' or 'false', or '1' or '0'." << std::endl;
        return -1;
    }

    // PointCloudProcessor processor(
    //     pointCloudPath, 
    //     odometryPath, 
    //     imagesFolder,
    //     maskImageFolder, 
    //     outputPath, 
    //     enableMLS);

    PointCloudProcessor processor(
        pointCloudPath,
        odometryPath,
        imagesFolder,
        maskImageFolder.value_or(imagesFolder), // Use imagesFolder if maskImageFolder is not provided
        outputPath.value_or("."), // Default output path is the current directory
        enableMLS.value_or(false) // Default value for enableMLS is false
    );

    try {
        processor.process();
        std::cout << "Processing completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -2;
    }

    return 0;
}
