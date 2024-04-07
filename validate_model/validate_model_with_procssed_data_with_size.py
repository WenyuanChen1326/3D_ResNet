import sys
import nibabel as nib
from tqdm import tqdm
import logging
import pickle

import contextlib
sys.path.append("./")
from src.utils import *
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
from torchvision.transforms import Compose, Lambda
from Model.r3d_18 import CustomDataset, test_model ,load_data


import logging

# Create a unique logger for this file only
logger = logging.getLogger('validate_model_with_processed_region_suv_block_with_size')
# Set the logging level for this logger
logger.setLevel(logging.INFO)

# Create a file handler specific to this logger
file_handler = logging.FileHandler('validate_model_with_processed_region_suv_block_with_size.log', mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# Optionally, create a console handler if you want to see the logs in the console as well
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# Add the handlers to the unique logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
# Ensure no propagation to the root logger
logger.propagate = False

def test_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # No gradients need to be calculated
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = torch.cat([inputs]*3, dim=1)  # Adjust input dimensions if necessary
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)

            # loss = criterion(outputs.squeeze(-1), labels.float())
            loss = criterion(outputs.squeeze(-1), labels.float()) 

            test_loss += loss.item()

            predicted = torch.round(torch.sigmoid(outputs.squeeze(-1)))
            # Convert logits to probabilities
            # probabilities = torch.softmax(outputs, dim=1)

            # Get the predicted class indices based on the higher probability
            # predicted = torch.argmax(probabilities, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    # Additional metrics can be calculated here using all_labels and all_predictions
    # For example:
    # from sklearn.metrics import precision_score, recall_score, f1_score
    # precision = precision_score(all_labels, all_predictions)
    # recall = recall_score(all_labels, all_predictions)
    # f1 = f1_score(all_labels, all_predictions)

    logging.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    # print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')

    return all_predictions, test_loss, test_accuracy  #, precision, recall, f1

def load_model(model_path, device):
    # Initialize the model
    model = r3d_18(weights=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    return model
def create_test_loader(transform, test_features):
    test_labels = np.ones(len(test_features))
    test_dataset = CustomDataset(test_features, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader
def get_accuracy(suv_block_lst, model, trasnform, criterion, device):
    if len(suv_block_lst) <=0:
        return "N/A", "N/A", "N/A"
    loader = create_test_loader(trasnform, suv_block_lst)
    all_predictions, test_loss, test_accuracy  = test_model(model, loader,criterion,device)
    return all_predictions, test_loss, test_accuracy
def get_region_coorindates(data_directory, output_csv, model, transform, criterion, device,):
    with open(data_directory, "rb") as file:
    #Load the list of dictionaries from the file
        loaded_data = pickle.load(file)
    
    with open(output_csv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header_written = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
        if not header_written:
            header = ['Patient ID', 'Study ID', 'Prediction','From tumor size']
            csvwriter.writerow(header)

        for dictionary in tqdm(loaded_data, desc="Processing patients"):
            patient_id = dictionary['patient_id']
            study_id = dictionary['study_id']
            suv_block_lst = dictionary['suv_block_lst']


            if patient_id == 'PETCT_1285b86bea':
                continue
            # if len(suv_block_lst) > 0:
            filtered_lists_with_indices = [(index, inner_list) for index, inner_list in enumerate(suv_block_lst) if np.max(inner_list) <= 3]
            if len(filtered_lists_with_indices) <= 0:
                continue
            indices, filtered_suv_block_lst = zip(*filtered_lists_with_indices)
            tumor_voxel_lst = dictionary['voxel_size_lst']
            filtered_tumor_voxel_lst = [tumor_voxel_lst[index] for index in indices]
        # else:
            #     continue
            # patient_folder = os.path.join(data_directory, patient_id)
            # subfolders = sorted([f.name for f in os.scandir(patient_folder) if f.is_dir() and not f.name.startswith('.')])
            # for study_id in subfolders:
            #     unique_id = f"{patient_id}_{study_id}"
                # if checkpoint and (checkpoint.get('patient_id') == patient_id and checkpoint.get('study_id') >= study_id):
                #     logger.info(f"Skipping already processed combination {unique_id}")
                #     continue
                
            # logger.info(f"-------------------------Processing patient {patient_id}-----------------------------")
            logger.info(f"-------------------------Processing patient {patient_id}-----------------------------")
            # subfolder_path = os.path.join(patient_folder, study_id)
            # print(subfolder_path)
            # with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            # suv_block_lst = process_patient_folder(subfolder_path=subfolder_path, block_size = block_size)
            all_predictions, test_loss, test_accuracy = get_accuracy(filtered_suv_block_lst, model, transform, criterion, device)
            # number_of_tumor_block = len(suv_block_lst)
            if all_predictions == "N/A":
                row_data = [patient_id, study_id, all_predictions, "N/A"]
            else:
                for idx, prediction in enumerate(all_predictions):
                    logger.info(f'test loss is {test_loss}, test accuracy is {test_accuracy}')
                    row_data = [patient_id, study_id, prediction, filtered_tumor_voxel_lst[idx]]
                    csvwriter.writerow(row_data)
            logger.info(f"-------------------------Finished processing patient {patient_id}-----------------------------")
            # print(suv_block)

def main(data_directory, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get_region_coorindates(data_directory,modality=modality, block_size = block_size)
    data =load_data()
    train_features = data['pet_train_features']
    mean = np.mean(train_features)  # Calculate the mean of the training data
    std = np.std(train_features)  
    transform = Compose([
        Lambda(lambda x: scale_up_block(x, new_resol=[112, 112, 112])),
        Lambda(lambda x: (x - mean) / std) 
        # Add other transforms here as needed
    ])
    model_path = "model_after_training_2024-03-07_14-38-49.pth"
    model = load_model(model_path, device)
    logger.info(f"model path{model_path}")
    criterion = nn.BCEWithLogitsLoss()
    # output_csv = "all_study_test_accuracy_loss.csv"
    get_region_coorindates(data_directory, output_csv, model, transform, criterion, device)

if __name__ == "__main__":
    # data_directory = "/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data"
    logger.info('start!')
    # data_directory = "/Volumes/T7 Shield/IBM/FDG-PET-CT-Lesions"
    # data_directory = "/home/ubuntu/jupyter-sandy/3D_ResNet/FDG-PET-CT-Lesions"
    # data_directory ='/home/ubuntu/jupyter-sandy/3D_ResNet/Data/pos_region_vox10_sample_size_5.pkl'
    # data_directory = '/home/ubuntu/jupyter-sandy/3D_ResNet/Data/pos_region_vox_inf_sample_size_5.pkl'
    # data_directory = '/home/ubuntu/jupyter-sandy/3D_ResNet/Data/neg_region_max_suv_min_threshold_5_sample_size_10.pkl'
    data_directory = '/home/ubuntu/jupyter-sandy/3D_ResNet/Data/pos_region_with_size_vox_inf_sample_size_5.pkl'
    file_name = os.path.basename(data_directory)[:-4]+ "_with_SUV_max_smaller_than_3." + "csv"
    output_csv = f"test_accuracy_loss_with_size_for_{file_name}"
    # logger.info(f"processing neg high suv block")
    logger.info(f"Validating with data {data_directory}")
    logger.info(f"writing to output csv {output_csv}")
    main(data_directory, output_csv)
    logger.info('done!')