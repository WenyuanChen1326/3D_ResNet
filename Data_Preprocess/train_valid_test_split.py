import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

path = './Data/all_patients_sample_size_100_for_both_pos_neg.csv'
raw_df = pd.read_csv(path)
print(f'raw_df shape: {raw_df.shape}')
raw_df.groupby(['Patient ID', 'Study ID']).size()
num_unique_patients = raw_df.groupby(['Patient ID']).size()
assert num_unique_patients.shape[0] == 900-1
num_unique_studies = raw_df.groupby(['Patient ID', 'Study ID']).size()
assert num_unique_studies.shape[0] == 1014-1

remove_all_0s = raw_df[~(raw_df['Block Size'] == '0')]
print(remove_all_0s.shape)
# Load the dataset
df = remove_all_0s.reset_index(drop = True)

# First split: Separate out the test set
train_val_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=df['Positive Tumor'], random_state=42)
# Second split: Split the remaining data into training and validation sets
train_val_df = df.loc[train_val_idx]
test_df = df.loc[test_idx]
train_idx, valid_idx = train_test_split(train_val_idx, test_size=0.25, stratify=train_val_df['Positive Tumor'], random_state=42)
train_df = df.loc[train_idx]
val_df = df.loc[valid_idx]

# Create a new column called 'Diagnosis' in the raw_df
# tuples = list(zip(raw_df['Patient ID'], raw_df['Study ID']))
# Now map these tuples to their corresponding diagnosis using the diagnosis_dict
# raw_df['Diagnosis'] = pd.Series(tuples).map(diagnosis_dict)
path = './Data/400_by_400_ct_dataset.npy'
features = np.load(path, allow_pickle=True)
# features = np.load("./Data/ori_reso_all_dataset.npy", allow_pickle=True)
# print(features.shape)
# labels =np.load("./Data/ori_reso_all_labels.npy", allow_pickle=True)
# print(labels.shape)

train_features = features[train_idx]
valid_features = features[valid_idx]
test_features = features[test_idx]

# np.save('./Data/train_ct_features.npy', train_features)
# np.save('./Data/valid_ct_features.npy', valid_features)
# np.save('./Data/test_ct_features.npy', test_features)


# train_features, train_labels = features[train_idx], labels[train_idx]
# valid_features, valid_labels = features[valid_idx], labels[valid_idx]
# test_features, test_labels = features[test_idx], labels[test_idx]

import numpy as np

# Assuming 'features' and 'labels' are your numpy arrays and 'train_idx', 'valid_idx', 'test_idx' are your indices
# train_features, train_labels = features[train_idx], labels[train_idx]
# valid_features, valid_labels = features[valid_idx], labels[valid_idx]
# test_features, test_labels = features[test_idx], labels[test_idx]


# # Save the arrays to .npy files
# np.save('./Data/train_features.npy', train_features)
# np.save('./Data/train_labels.npy', train_labels)
# np.save('./Data/valid_features.npy', valid_features)
# np.save('./Data/valid_labels.npy', valid_labels)
# np.save('./Data/test_features.npy', test_features)
# np.save('./Data/test_labels.npy', test_labels)

# print(train_features.shape)


