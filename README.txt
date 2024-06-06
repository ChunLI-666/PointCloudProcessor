cmake .. -DCMAKE_INSTALL_PREFIX=/root/open3d_install/lib/cmake/Open3D
make -j

./PointCloudColorization \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/color_scans.pcd \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/visual_odom.txt \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/_2024-03-19-15-27-57-images/ \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/
0

./PointCloudColorization \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test03/scans-clean-mls.pcd \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test03/vo_interpolated_odom.txt \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test03/raw_images/ \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test03/ \
0

./PointCloudColorization \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test-new-extrinsic/scans-clean-mls-clean.pcd \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test-new-extrinsic/vo_interpolated_odom.txt \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test-new-extrinsic/raw_images/ \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test-new-extrinsic/ \
0



Segment mask:
./PointCloudColorization \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test-new-extrinsic/scans-clean-mls-clean.pcd \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test-new-extrinsic/vo_interpolated_odom.txt \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test-new-extrinsic/raw_images/ \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test-new-extrinsic/mask_select/ \
/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test-new-extrinsic/ \
0


./PointCloudProcessor \
--point_cloud_path /mnt/disk01/data/zhongnna/_2024-05-16-17-22-50-reconstruction/fast_lio_result/scans-mls-clean.pcd \
--odometry_path /mnt/disk01/data/zhongnna/_2024-05-16-17-22-50-reconstruction/fast_lio_result/vo_interpolated_odom.txt \
--images_folder /mnt/disk01/data/zhongnna/_2024-05-16-17-22-50-reconstruction/fast_lio_result/raw_images/ \
--output_path /mnt/disk01/data/zhongnna/_2024-05-16-17-22-50-reconstruction/fast_lio_result/ \
--enableMLS 0 \
--enableNIDOptimize 1