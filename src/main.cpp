#include "PointCloudProcessor.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <point_cloud_path> <odometry_path> <images_folder> <output_path> <enableMLS>" << std::endl;
        return -1;
    }

    std::string pointCloudPath = argv[1];
    std::string odometryPath = argv[2];
    std::string imagesFolder = argv[3];
    std::string outputPath = argv[4];
    bool enableMLS = std::stoi(argv[5]);

    PointCloudProcessor processor(pointCloudPath, odometryPath, imagesFolder, outputPath, enableMLS);

    try {
        processor.process();
        std::cout << "Processing completed successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -2;
    }

    return 0;
}
