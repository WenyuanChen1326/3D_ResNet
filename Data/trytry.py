import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

def most_common_diagnosis(diagnoses):
    if not diagnoses:
        return None
    return Counter(diagnoses).most_common(1)[0][0]

def create_stratified_patient_splits(metadata_csv_path, output_dir, test_size=0.2, val_size=0.1):
    # Load metadata
    df = pd.read_csv(metadata_csv_path)
    df = df[df['study_location'].apply(lambda x: x.split('/')[2]) != 'PETCT_1285b86bea']
    
    # Extract new patient_id and study_id from the 'study_location' field
    df['patient_id'] = df['study_location'].apply(lambda x: x.split('/')[2])
    df['study_id'] = df['study_location'].apply(lambda x: x.split('/')[3])
    
    # Use the new patient_id for further grouping
    patient_diagnoses = df.groupby('patient_id')['diagnosis'].agg(list).reset_index()
    
    # Determine the most common diagnosis for stratification
    patient_diagnoses['most_common_diagnosis'] = patient_diagnoses['diagnosis'].apply(most_common_diagnosis)
    
    # Stratify based on the most common diagnosis
    patient_ids = patient_diagnoses['patient_id']
    stratify_on = patient_diagnoses['most_common_diagnosis']
    
    # Split the data into train and test sets while maintaining the distribution of the most common diagnosis
    train_ids, test_ids = train_test_split(patient_ids, test_size=test_size, 
                                           stratify=stratify_on, random_state=42)
    
    # Further split the train set into training and validation sets
    train_ids, val_ids = train_test_split(train_ids, test_size=val_size / (1 - test_size), 
                                          stratify=stratify_on[train_ids.index], random_state=42)
    
    # Assign split labels
    patient_diagnoses['split'] = 'test'
    patient_diagnoses.loc[patient_diagnoses['patient_id'].isin(train_ids), 'split'] = 'train'
    patient_diagnoses.loc[patient_diagnoses['patient_id'].isin(val_ids), 'split'] = 'val'
    
    # Merge the split information back into the original DataFrame
    df_with_splits = df.merge(patient_diagnoses[['patient_id', 'split']], on='patient_id', how='left')
    
    # Save the updated DataFrame with split information to a CSV file
    os.makedirs(output_dir, exist_ok=True)
    df_with_splits.to_csv(os.path.join(output_dir, 'data_with_splits.csv'), index=False)
    
    # Return the DataFrame with splits for further processing if needed
    return df_with_splits

# Run the function with your data
metadata_csv_path = '/home/ubuntu/jupyter-sandy/3D_ResNet/Data/autoPETmeta.csv' # Replace with your actual path
output_dir = '/home/ubuntu/jupyter-sandy/3D_ResNet/Data/Data_Split'  # Replace with your actual path
df_with_splits = create_stratified_patient_splits(metadata_csv_path, output_dir)