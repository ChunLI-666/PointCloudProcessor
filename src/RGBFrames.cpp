#include "RGBFrames.hpp"
#include <algorithm> // For std::find_if

void RGBFrames::addFrame(const FrameData &frame)
{
    frames.push_back(frame);
}

// FrameData *RGBFrames::getFrameByTimestamp(double timestamp)
// {
//     auto it = std::find_if(frames.begin(), frames.end(), [timestamp](const FrameData &frame)
//                            { return frame.image_timestamp == timestamp; });

//     if (it != frames.end())
//     {
//         return &(*it); // Return a pointer to the found frame
//     }
//     else
//     {
//         return nullptr; // Return nullptr if not found
//     }
// }
