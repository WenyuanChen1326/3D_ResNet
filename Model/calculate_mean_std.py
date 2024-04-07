import os
import sys
import numpy as np
import pandas as pd
import h5py
sys.path.append("./")
def incremental_mean_std(hdf5_paths):
    n = 0
    mean = 0.0
    M2 = 0.0

    for hdf5_path in hdf5_paths:
        try:
            with h5py.File(hdf5_path, 'r') as file:
                # Assuming your data is stored in datasets named 'positive' and 'negative'
                for dataset_name in ['positive', 'negative']:
                    data = file[dataset_name][:]
                    data = data.astype(np.float32)  # Ensure float32 for numerical stability
                    for sample in data:
                        n += 1
                        x = np.mean(sample)
                        delta = x - mean
                        mean += delta / n
                        delta2 = x - mean
                        M2 += delta * delta2
        except FileNotFoundError:
            pass

    std = np.sqrt(M2 / n)
    return mean, std

# Load your CSV to get the HDF5 file paths for the training split
csv_file = './Data/Data_Split/data_with_splits.csv'
df = pd.read_csv(csv_file)
train_df = df[df['split'] == 'train']
root_dir = './Data/Processed_Block_block_size3'

# Assuming the HDF5 file paths are stored in a column named 'file_path'
hdf5_paths = [os.path.join(root_dir, row['patient_id'], row['study_id'], row['study_id'] + '_blocks.hdf5') for _, row in train_df.iterrows()]

mean, std = incremental_mean_std(hdf5_paths)
print(f'Mean: {mean}, Std: {std}')
# Should output Mean for block size 3: 3.272275845858922, Std: 3.7406388529889547
