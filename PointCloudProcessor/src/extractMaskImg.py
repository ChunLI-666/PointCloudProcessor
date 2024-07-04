import os
import shutil
import pandas as pd
from datetime import datetime, timedelta

# Set the directory paths
source_folder = '/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test-new-extrinsic/mask_org'
destination_folder = '/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test-new-extrinsic/mask_select'
data_file = '/sandbox/Documents/zhongnan/fastlio-color/test-offline-color/test-new-extrinsic/vo_interpolated_odom.txt'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Load the data file
# data = pd.read_csv(data_file, delimiter=',', names=['timestamp', 'x', 'y', 'z', 'qx','qy','qz','qw',])
data = pd.read_csv(data_file, delimiter=' ', usecols=[0], names=['timestamp'])
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')  # Assuming epoch time in seconds

# Function to find the closest timestamp
def find_closest_timestamp(image_timestamp, dataframe):
    # Calculate the time difference
    time_diff = (dataframe['timestamp'] - image_timestamp).abs()
    # Find the minimum time difference
    min_diff = time_diff.min()
    if min_diff <= timedelta(seconds=0.5):
        return dataframe.loc[time_diff.idxmin(), 'timestamp']
    return None

# Process each image in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith('.png'):
        # Extract the timestamp from the filename
        timestamp_str = filename.split('.png')[0]
        image_timestamp = pd.to_datetime(float(timestamp_str), unit='s')  # Handle fractional seconds correctly
        # Find the closest timestamp in the data
        closest_timestamp = find_closest_timestamp(image_timestamp, data)
        if closest_timestamp is not None:
            new_filename = str(float(closest_timestamp.timestamp())) + '.png'
            # Move and rename the file
            shutil.move(os.path.join(source_folder, filename),
                        os.path.join(destination_folder, new_filename))
            
print("Processing complete.")
