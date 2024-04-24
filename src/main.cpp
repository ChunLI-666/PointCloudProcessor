#include "PointCloudProcessor.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <point_cloud_path> <odometry_path> <images_folder> <segment_images_folder> <output_path> <enableMLS>" << std::endl;
        return -1;
    }

    std::string pointCloudPath = argv[1];
    std::string odometryPath = argv[2];
    std::string imagesFolder = argv[3];
    std::string maskImageFolder = argv[4];
    std::string outputPath = argv[5];
    bool enableMLS = std::stoi(argv[6]);

    PointCloudProcessor processor(
        pointCloudPath, 
        odometryPath, 
        imagesFolder,
        maskImageFolder, 
        outputPath, 
        enableMLS);

    try {
        processor.process();
        std::cout << "Processing completed successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -2;
    }

    return 0;
}
