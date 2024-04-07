
import sys
import nibabel as nib
from tqdm import tqdm
import contextlib
sys.path.append("./")
# from src.utils import scale_up_block
from src.utils import *
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
from torchvision.transforms import Compose, Lambda
from Model.r3d_18 import *
# def get_pos_block(seg_data):
def process_patient_folder(subfolder_path, sample_size = 5, block_size = (3, 3, 3)):
    seg_file_path = os.path.join(subfolder_path, 'SEG.nii.gz')
    suv_file_path = os.path.join(subfolder_path, 'SUV.nii.gz')
    ct_file_path = os.path.join(subfolder_path, 'CTres.nii.gz')


    seg_img = nib.load(seg_file_path)
    seg_data = seg_img.get_fdata()
    voxel_dimensions = seg_img.header['pixdim'][1:4]

    suv_img = nib.load(suv_file_path)
    suv_data = suv_img.get_fdata()

    ct_img = nib.load(ct_file_path)
    ct_data = ct_img.get_fdata()

    separate_segmentation_masks = get_connected_components_3D(seg_data)
    filtered_separate_segmentation_masks = filter_separate_segmentation_mask_by_diameter_and_SUV_max_and_voxel_of_interest(
    suv_data, voxel_dimensions, separate_segmentation_masks, 
    diameter_in_cm = 0.6, SUV_max = 3, voxel_of_interst = 10)
    suv_block_lst = []
    for mask in filtered_separate_segmentation_masks:
        coordinates = np.where(mask == 1)
        pos_xyz_coordinates = np.array(list(zip(coordinates[0], coordinates[1], coordinates[2])))
        non_restricted_positive_coordinates = sample_neg_block(seg_data,pos_xyz_coordinates,block_size, sample_size, negative=False)
        for idx, coordinate in enumerate(non_restricted_positive_coordinates):
            suv_block = np.array(suv_data[slice(coordinate[0], coordinate[0] + block_size[0]),
                        slice(coordinate[1], coordinate[1] + block_size[1]),
                        slice(coordinate[2], coordinate[2] + block_size[2])])
            suv_block_lst.append(suv_block)
            # print(f"shape of suv_block {suv_block.shape}")
    # print(len(suv_block_lst))

    return suv_block_lst

    # return {
    #     'suv_block_lst': suv_block_lst
    # }
def get_accuracy(suv_block_lst, model, trasnform, criterion, device):
    if len(suv_block_lst) <=0:
        return "N/A", "N/A"
    loader = create_test_loader(trasnform, suv_block_lst)
    test_loss, test_accuracy  = test_model(model, loader,criterion,device)
    return test_loss, test_accuracy

def get_region_coorindates(data_directory, output_csv, model, transform, criterion, device, modality = 'pet', block_size = (3, 3, 3)):
    with open(output_csv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header_written = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
        if not header_written:
            header = ['Patient ID', 'Study ID', 'Test Accuracy', 'Test Loss']
            csvwriter.writerow(header)

        assert modality in ['pet', 'ct'], "modality must be either 'pet', 'ct'"
        patients = sorted([pid for pid in os.listdir(data_directory) if not pid.startswith('.') and os.path.isdir(os.path.join(data_directory, pid))])
        for patient_id in tqdm(patients, desc="Processing patients"):
            if patient_id == 'PETCT_1285b86bea':
                continue
            patient_folder = os.path.join(data_directory, patient_id)
            subfolders = sorted([f.name for f in os.scandir(patient_folder) if f.is_dir() and not f.name.startswith('.')])
            for study_id in subfolders:
                unique_id = f"{patient_id}_{study_id}"
                # if checkpoint and (checkpoint.get('patient_id') == patient_id and checkpoint.get('study_id') >= study_id):
                #     logger.info(f"Skipping already processed combination {unique_id}")
                #     continue
                
                # logger.info(f"-------------------------Processing patient {patient_id}-----------------------------")
                print(f"-------------------------Processing patient {patient_id}-----------------------------")
                subfolder_path = os.path.join(patient_folder, study_id)
                # print(subfolder_path)
                # with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                suv_block_lst = process_patient_folder(subfolder_path=subfolder_path, block_size = block_size)
                test_loss, test_accuracy = get_accuracy(suv_block_lst,model, transform, criterion, device)
                row_data = [patient_id, study_id, test_accuracy, test_loss]
                csvwriter.writerow(row_data)
            # print(suv_block)

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
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    return test_loader
# def validate_model_by_pos_region(model, test_loader):
#     test_model

def main(data_directory, modality, block_size = (3, 3, 3)):
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
    model = load_model("model_after_training_2024-03-07_14-38-49.pth", device)
    criterion = nn.BCEWithLogitsLoss()
    output_csv = "all_study_test_accuracy_loss.csv"
    get_region_coorindates(data_directory, output_csv, model, transform, criterion, device)


if __name__ == "__main__":
    # data_directory = "/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data"
    print('start!')
    # data_directory = "/Volumes/T7 Shield/IBM/FDG-PET-CT-Lesions"
    data_directory = "/home/ubuntu/jupyter-sandy/3D_ResNet/FDG-PET-CT-Lesions"
    modality = 'pet'
    main(data_directory, modality, block_size=(3, 3, 3))
    print('done!')

    



    