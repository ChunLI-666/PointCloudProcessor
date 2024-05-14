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


./PointCloudColorization \
--point_cloud_path /sandbox/Documents/zhongnan/fastlio-color/20240507/_2024-05-07-10-57-20-reconstruction/fast_lio_result/scans.pcd \
--odometry_path /sandbox/Documents/zhongnan/fastlio-color/20240507/_2024-05-07-10-57-20-reconstruction/fast_lio_result/vo_interpolated_odom.txt \
--images_folder /sandbox/Documents/zhongnan/fastlio-color/20240507/_2024-05-07-10-57-20-reconstruction/fast_lio_result/raw_images/ \
--output_path /sandbox/Documents/zhongnan/fastlio-color/20240507/_2024-05-07-10-57-20-reconstruction/fast_lio_result/ \
--enableMLS 1