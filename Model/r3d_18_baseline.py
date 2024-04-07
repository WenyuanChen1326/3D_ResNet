import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
from torchvision.transforms import Compose, Lambda
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import argparse
import logging
import sys
from datetime import datetime
import numpy as np
from torchvision.transforms import Compose, Lambda
sys.path.append("./")
from src.utils import scale_up_block

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the minimum logging level

# Create a file handler that logs messages to a file

file_handler = logging.FileHandler(filename = 'r3d_18_baseline.log', mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Create a stream handler that logs messages to the console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    torch.save(state, filename)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
class PETCTDataset(Dataset):
    def __init__(self, csv_file, root_dir, split_type='train', neg_sampling_ratio=1.0, transform=None, load_all_test_samples=False):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the H5 files.
            split_type (string): One of 'train', 'val', or 'test' to denote the split.
            neg_sampling_ratio (float): Ratio of negative to positive samples to use for training.
            load_all_test_samples (bool): Whether to load all samples during test time.
        """
        self.petct_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split_type = split_type
        self.neg_sampling_ratio = neg_sampling_ratio
        self.transform = transform
        self.load_all_test_samples = load_all_test_samples

        # Filter the dataframe for the split
        self.split_frame = self.petct_frame[self.petct_frame['split'] == split_type]

        # Load blocks based on split type
        if self.split_type == 'test' and self.load_all_test_samples:
            # For test split, load all positive and negative blocks if load_all_test_samples is True
            self.all_blocks, self.all_labels = self.load_all_test_blocks()
        else:
            # For non-test splits or test without load_all_test_samples, only load positive blocks for dynamic sampling
            self.positive_blocks = self.load_all_blocks(block_type='positive')
            # self.negative_blocks_cache = []

    def load_all_test_blocks(self):
        blocks = []
        labels = []
        for _, row in self.split_frame.iterrows():
            for block_type in ['positive', 'negative']:
                file_path = os.path.join(self.root_dir, row['patient_id'], row['study_id'],row['study_id'] + '_blocks.hdf5')
                try:
                    with h5py.File(file_path, 'r') as hdf5_file:
                        blocks.extend(hdf5_file[block_type][:])
                        labels.extend([1 if block_type == 'positive' else 0] * hdf5_file[block_type][:].shape[0])
                except FileNotFoundError:
                    pass
        return blocks, labels
    
    def load_all_blocks(self, block_type):
        blocks = []
        for _, row in self.split_frame.iterrows():
            file_path = os.path.join(self.root_dir, row['patient_id'], row['study_id'], row['study_id'] + '_blocks.hdf5')
            try:
                with h5py.File(file_path, 'r') as hdf5_file:
                    blocks.extend(hdf5_file[block_type][:])
                        # file_loaded = True
            except FileNotFoundError:
                pass
            # with h5py.File(file_path, 'r') as hdf5_file:
            #     blocks.extend(hdf5_file[block_type][:])
        return blocks

    def __len__(self):
        if self.split_type == 'test' and self.load_all_test_samples:
            return len(self.all_blocks)
        else:
            return int(len(self.positive_blocks) *(1 + self.neg_sampling_ratio))

    def __getitem__(self, idx):
        if self.split_type == 'test' and self.load_all_test_samples:
            sample = self.all_blocks[idx]
            label = self.all_labels[idx]
        else:
            # Existing logic for training/validation or test without load_all_test_samples
            if idx < len(self.positive_blocks):
                sample = self.positive_blocks[idx]
                label = 1
            else:
                # Dynamically sample a negative block by selecting a random study
                study_row = self.split_frame.sample(n=1).iloc[0]
                patient_id = study_row['patient_id']
                study_id = study_row['study_id']
                # Construct the file path for the negative blocks for the selected patient and study
                neg_file_path = os.path.join(self.root_dir, patient_id, study_id, study_id + '_blocks.hdf5')
                try:
                    with h5py.File(neg_file_path, 'r') as hdf5_file:
                        # Load negative blocks for the selected patient and study
                        neg_blocks = hdf5_file['negative'][:]
                        # Randomly select one negative block
                        neg_idx = random.randint(0, len(neg_blocks) - 1)
                        sample = neg_blocks[neg_idx]
                        label = 0
                except FileNotFoundError:
                    logging.info(f"No negative blocks found for patient {patient_id} in study {study_id}")
                    pass
        
        if self.transform:
            sample = self.transform(sample)
        sample = np.expand_dims(sample, axis=0)
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        # return neg_block, torch.tensor(0)  # 0 as the label for negative

# Usage:
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device,current_run_dir, patience=5, multiple_GPUs = False):
    assert current_run_dir is not None
    print(f"Model type before training: {type(model)}")
    logging.info(f'multiple_GPUs is {multiple_GPUs}')
    logging.info('training start!')
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []


    best_valid_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    # Training loop
    for epoch in tqdm(range(args.start_epoch, num_epochs)):
        logging.info(f"epoch {epoch} starts")
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        # temp = 0
        for inputs, labels in tqdm(train_loader):
            # logging.info(inputs.shape)
            # assert inputs.dim() == 5 and inputs.size(1) == 1
            # Assuming inputs is a batch of 3D data with shape [batch_size, C, D, H, W]
            # where C is channels, D is depth, H is height, W is width.
            # inputs = inputs.repeat(1, 3, 1, 1, 1)
            # inputs = inputs.unsqueeze(1)  # Adds a channel dimension
            # inputs = inputs.repeat(1, 3, 1, 1, 1)
            inputs = torch.cat([inputs]*3, dim=1) 
            # print(inputs.shape)

            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.to(device), labels.to(device)
            # labels = labels.float().unsqueeze(1)

            # print(labels.shape)
            # print(labels)

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # print("Outputs shape:", outputs.shape)
            # print(outputs)
            # print("Outputs shape:",outputs.squeeze(-1).shape)


            loss = criterion(outputs.squeeze(-1), labels.float())
            # loss = criterion(outputs, labels) 
            # print(outputs.squeeze(-1))
            # print(f"label is {labels.float()}")
            # print(f"loss is {loss.item()}")
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print(f'running loss is {running_loss}')

            predicted = torch.round(torch.sigmoid(outputs.squeeze(-1)))
            # Convert logits to probabilities
            # probabilities = torch.softmax(outputs, dim=1)

            # Get the predicted class indices based on the higher probability
            # predicted = torch.argmax(probabilities, dim=1)
            # print(f'predicted {predicted}')
            # print(f'labels {labels}')
            # predicted = predicted.unsqueeze(1)  # Add a dimension to match labels' shape
            correct_preds = (predicted == labels).float()
            correct_train += correct_preds.sum().item()

            # print(f'train prediction shape{predicted.shape}')
            # print(f'((predicted == labels).sum().item()){((predicted == labels).sum().item())}')
            # print()
            
            # correct_train += ((predicted == labels).sum().item())
            total_train += labels.size(0)
            assert correct_train <= total_train
            # print(f"label_size_0 {labels.size(0)}")
            # print(f'total train within epoch {total_train}')

            # temp +=1
            # if temp >=4:
            #     break 
        
        # print(f'train loader{len(train_loader)}')
        # print(f'total train {total_train}')
        # print(f'correct train {correct_train}')
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        assert correct_train <= total_train


        # Validation loop after each epoch
        model.eval()  # Set model to evaluation mode
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():  # No need to track gradients during validation
            logging.info(f"valid starts for epoch {epoch} starts")
            # temp = 0
            for inputs, labels in tqdm(valid_loader):
                inputs = torch.cat([inputs]*3, dim=1) 
                # print(inputs.shape)
                inputs, labels = inputs.to(device), labels.to(device)
                # labels = labels.float().unsqueeze(1)
                outputs = model(inputs)
                valid_loss += criterion(outputs.squeeze(-1), labels.float()).item()
                # valid_loss += criterion(outputs, labels).item()
                predicted = torch.round(torch.sigmoid(outputs.squeeze(-1)))  # Binary classification
                correct_valid += (predicted == labels).sum().item()
                # print(predicted)
                # Convert logits to probabilities
                # probabilities = torch.softmax(outputs, dim=1)

                # Get the predicted class indices based on the higher probability
                # predicted= torch.argmax(probabilities, dim=1)
                # predicted = predicted.unsqueeze(1)  # Add a dimension to match labels' shape
                # correct_preds = (predicted == labels).float()
                # correct_valid += correct_preds.sum().item()
                # print(f'valid prediction shape{predicted.shape}')
                # print(f'((predicted == labels).sum().item()){((predicted == labels).sum().item())}')
                total_valid += labels.size(0)
                assert correct_valid <= total_valid
                # correct_valid += ((predicted == labels).sum().item())
                # temp +=1
                # if temp >=4:
                #     break
        # print(f'valid loader{len(valid_loader)}')
        # print(f'total valid {total_valid}')
        # print(f'correct valid {correct_valid}')
        assert correct_valid <= total_valid

        epoch_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(epoch_valid_loss)
        valid_accuracy = 100 * correct_valid / total_valid
        valid_accuracies.append(valid_accuracy)
        #  Early stopping logic

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = model.state_dict()  # Save the best model state
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logging.info(f'Validation loss did not improve for {epochs_no_improve}/{patience} epochs.')

        if epoch% 3 == 0:
            checkpoint_filename = f'{current_run_dir}/checkpoint_epoch_{epoch}.pth.tar'

            # if multiple_GPUs:
            print(f"Model type before error: {type(model)}")
            if isinstance(model, nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            save_checkpoint({
                'epoch': epoch + 1,
                # 'state_dict': model.state_dict(),
                'state_dict': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'valid_losses': valid_losses,
                'valid_accuracies': valid_accuracies,
                'current_run_dir': current_run_dir
            }, filename=checkpoint_filename)  
            logging.info(f'checking point saving at epoch {epoch}')

        # Check if early stopping should be triggered
        if epochs_no_improve >= patience:
            logging.info('Early stopping triggered.')
            model.load_state_dict(best_model_state)  # Restore the best model state
            break  # Break out of the loop

        logging.info(f"Epoch {epoch}:")
        logging.info(f" Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
        logging.info(f" Validation Loss: {epoch_valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, train_accuracies, valid_losses, valid_accuracies
def test_model(model, test_loader, criterion, device, current_run_dir):
    logging.info("Testing Starts!")
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    # correct = 0
    # total = 0
    all_labels = []
    all_predictions = []
    temp = 0

    with torch.no_grad():  # No gradients need to be calculated
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = torch.cat([inputs]*3, dim=1)  # Adjust input dimensions if necessary
            inputs, labels = inputs.to(device), labels.to(device)
            # print(inputs[0])
            # print(labels)
            outputs = model(inputs)

            # loss = criterion(outputs.squeeze(-1), labels.float())
            loss = criterion(outputs.squeeze(-1), labels.float()) 

            test_loss += loss.item()

            predicted = torch.round(torch.sigmoid(outputs.squeeze(-1)))
            # Convert logits to probabilities
            # probabilities = torch.softmax(outputs, dim=1)

            # Get the predicted class indices based on the higher probability
            # predicted = torch.argmax(probabilities, dim=1)

            # correct += (predicted == labels).sum().item()
            # total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            if len(all_predictions) % 1000 == 0:
                report_and_plot_metrics(all_labels, all_predictions, 'Test', current_run_dir, full = False)
                logging.info(f'Test Loss: {test_loss:.4f}')

            # temp +=1
            # if temp >= 5:
            #     break

    # test_loss /= len(test_loader)
    # test_accuracy = 100 * correct / total
    report_and_plot_metrics(all_labels, all_predictions, 'Test', current_run_dir, full = True)
    # print(f"the sum of predictions is {np.sum(all_predictions)}")
    # print(f"the sum of labels is {np.sum(all_labels)}")

    # Additional metrics can be calculated here using all_labels and all_predictions
    # For example:
    # from sklearn.metrics import precision_score, recall_score, f1_score
    # precision = precision_score(all_labels, all_predictions)
    # recall = recall_score(all_labels, all_predictions)
    # f1 = f1_score(all_labels, all_predictions)
    logging.info(f'Test Loss: {test_loss:.4f}')
    # logging.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    # print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')

    # return all_predictions, test_loss, test_accuracy  #, precision, recall, f1
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.tight_layout()
    plt.ylabel('True label', fontsize = 14)
    plt.xlabel('Predicted label', fontsize = 14)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.show()
    return figure

def compute_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, precision, recall, f1

def report_and_plot_metrics(labels, predictions, phase,current_run_dir, full = False):
    # Compute metrics
    accuracy, precision, recall, f1 = compute_metrics(labels, predictions)

    # Log metrics
    logging.info(f'{phase} Size: {len(labels)}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, class_names=['Negative', 'Positive'])

    # Save figure
    cur_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if full:
        plt.savefig(f'{current_run_dir}/{phase}_confusion_matrix_{cur_time}_full.png')
    else:
        plt.savefig(f'{current_run_dir}/{phase}_confusion_matrix_{cur_time}_size_{len(labels)}.png')

def save_plots(train_losses, train_accuracies, valid_losses, valid_accuracies, num_epochs, block_size, current_run_dir):
    # Plot training and validation losses
    loss_plot_filename = f'{current_run_dir}/block_size_{block_size}_training_validation_loss.png'
    accuracy_plot_filename = f'{current_run_dir}/block_size_{block_size}_training_validation_accuracy.png'
    plt.figure(figsize=(10, 7))
    # num_epochs = min(num_epochs, len(train_losses))
    epochs_to_plot = min(num_epochs, len(train_losses))
    plt.plot(range(1, epochs_to_plot + 1), train_losses[:epochs_to_plot], label='Training Loss')
    # print(valid_losses)
    # print(valid_accuracies)
    # print(train_accuracies)
    plt.plot(range(1, epochs_to_plot + 1), valid_losses[:epochs_to_plot], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_plot_filename)
    plt.close()

    # Plot training and validation accuracies
    plt.figure(figsize=(10, 7))
    plt.plot(range(1, epochs_to_plot + 1), train_accuracies[:epochs_to_plot], label='Training Accuracy')
    plt.plot(range(1, epochs_to_plot + 1), valid_accuracies[:epochs_to_plot], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(accuracy_plot_filename)
    plt.close()
    logging.info(f'Plots saved: {loss_plot_filename} and {accuracy_plot_filename}')
def main(args):
    mean = 3.272275845858922
    std = 3.7406388529889547
    transform = Compose([
        Lambda(lambda x: scale_up_block(x, new_resol=[112, 112, 112])),
        Lambda(lambda x: (x - mean) / std)
    ])
    # read data
    train_dataset = PETCTDataset(csv_file='./Data/Data_Split/data_with_splits.csv',
                                root_dir='./Data/Processed_Block_block_size3',
                                split_type='train',
                                neg_sampling_ratio=1.05, transform=transform)  # Adjust the ratio as needed
    valid_dataset = PETCTDataset(csv_file='./Data/Data_Split/data_with_splits.csv',
                                root_dir='./Data/Processed_Block_block_size3',
                                split_type='val',
                                neg_sampling_ratio=1.05, transform=transform)  # Adjust the ratio as needed
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # Adjust batch size as needed
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    # print(len(test_dataset))

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    logging.info(f"Using device: {device}")

    # Initialize the model
    model = r3d_18(weights=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    # model.fc = nn.Linear(num_features, 2)

    multiple_GPUs = False
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        multiple_GPUs =  True
        # model = nn.DataParallel(model)
        model = nn.DataParallel(model).to(device)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if not args.test:
        current_run_dir = None
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Train the model
        # Check for checkpoint and load it if available
        if args.resume:
            if os.path.isfile(args.resume):
                print(f"=> loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume)

                args.start_epoch = checkpoint['epoch']
                # keys = list(checkpoint['state_dict'].keys())
                # if not keys[0].startswith('module.') and multiple_GPUs:
                #     # Add 'module.' prefix to align with the DataParallel model keys
                #     new_state_dict = {'module.' + k: v for k, v in checkpoint['state_dict'].items()}
                # elif keys[0].startswith('module.') and not multiple_GPUs:
                #     # Remove 'module.' prefix to align with the non-DataParallel model keys
                #     new_state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
                # else:
                #     new_state_dict = checkpoint['state_dict']
                # if multiple_GPUs:
                #     model.module.load_state_dict(new_state_dict)
                # else:
                #     model.load_state_dict(new_state_dict)
                keys = list(checkpoint['state_dict'].keys())
                if not (keys[0].startswith('module.') and multiple_GPUs):
                    # Add 'module.' prefix to align with the DataParallel model keys
                    new_state_dict = {'module.' + k: v for k, v in checkpoint['state_dict'].items()}
                elif (keys[0].startswith('module.') and not multiple_GPUs):
                    # Remove 'module.' prefix to align with the non-DataParallel model keys
                    new_state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
                else:
                    new_state_dict = checkpoint['state_dict']
                model.load_state_dict(new_state_dict)
                optimizer.load_state_dict(checkpoint['optimizer'])
                train_losses = checkpoint['train_losses']
                train_accuracies = checkpoint['train_accuracies']
                valid_losses = checkpoint['valid_losses']
                valid_accuracies = checkpoint['valid_accuracies']
                current_run_dir = checkpoint['current_run_dir']
                print(f"=> loaded checkpoint '{args.resume}' (epoch {args.start_epoch})")
        else:
            args.start_epoch = 0
            print(f"=> no checkpoint found at '{args.resume}'")
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            current_run_dir = f"./{current_time}"
            os.makedirs(current_run_dir, exist_ok=True)
        model, train_losses, train_accuracies, valid_losses, valid_accuracies = train_model(model, train_loader, valid_loader, 
                                            criterion, optimizer, args.epochs, device, current_run_dir, args.patience, multiple_GPUs)

        # Save the model
        # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_filename = f'{current_run_dir}/trained_model_block_{args.block_size}_{current_time}.pth'
        if multiple_GPUs:
            torch.save(model.module.state_dict(), model_filename)
        else:
            torch.save(model.state_dict(), model_filename)
        # Test the model
        # print(device)
        # test_model(model, test_loader,criterion, device)

        # Save training/validation plots
        save_plots(train_losses, train_accuracies, valid_losses, valid_accuracies, args.epochs, args.block_size,current_run_dir)

    else:
        test_dataset = PETCTDataset(csv_file='./Data/Data_Split/data_with_splits.csv',
                                root_dir='./Data/Processed_Block_block_size3',
                                split_type='test',
                                neg_sampling_ratio=1, transform=transform, load_all_test_samples=args.test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        # Initialize the base model

        model = r3d_18(pretrained=False)  # Set to False since you're loading custom weights
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)  # Adjust according to your model's architecture

        # Check if multiple GPUs should be used
        multiple_GPUs = False
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            multiple_GPUs = True
            model = nn.DataParallel(model)
        path = '/home/ubuntu/jupyter-sandy/3D_ResNet/2024-04-03_18-23-20/checkpoint_epoch_9.pth.tar'
        checkpoint = torch.load(path)
        # start_epoch = checkpoint['epoch']
        keys = list(checkpoint['state_dict'].keys())
        if not (keys[0].startswith('module.') and multiple_GPUs):
            # Add 'module.' prefix to align with the DataParallel model keys
            new_state_dict = {'module.' + k: v for k, v in checkpoint['state_dict'].items()}
        elif (keys[0].startswith('module.') and not multiple_GPUs):
            # Remove 'module.' prefix to align with the non-DataParallel model keys
            new_state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
        else:
            new_state_dict = checkpoint['state_dict']
        model.load_state_dict(new_state_dict)
        model.to(device)

        # Load model state
        '''path = '/path/to/your/model.pth'
        checkpoint = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        # If the model was trained with DataParallel, its state keys will contain the 'module.' prefix
        # Adjust the keys if necessary
        if multiple_GPUs and not list(checkpoint.keys())[0].startswith('module.'):
            # Model saved without DataParallel but loading on multiple GPUs
            checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
        elif not multiple_GPUs and list(checkpoint.keys())[0].startswith('module.'):
            # Model saved with DataParallel but loading on single GPU
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

        # Load the adjusted checkpoint
        model.load_state_dict(checkpoint)

        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)'''        

        
        # path = "/home/ubuntu/jupyter-sandy/3D_ResNet/model_after_training_2024-03-07_14-38-49.pth"
        # path = '/home/ubuntu/jupyter-sandy/3D_ResNet/2024-04-03_09-06-03/trained_model_block_(3, 3, 3)_2024-04-03_09-06-03.pth'
        # path = '/home/ubuntu/jupyter-sandy/3D_ResNet/2024-04-03_17-48-44/trained_model_block_(3, 3, 3)_2024-04-03_17-57-03.pth'
        # model.load_state_dict(torch.load(path))
        # model = nn.DataParallel(model).to(device)  # Wrap with DataParallel if testing on multiple GPUs
        # model.load_state_dict(torch.load(path))


        # Adjust for the DataParallel prefix
      

        # Load the adjusted state dictionary into the model
        # model.load_state_dict(new_state_dict)
        current_run_dir = '2024-04-03_18-23-20'
        # current_run_dir_path = f'2024-04-03_18-23-20/epoch_{start_epoch}'
        # path += "/Results"
        # current_run_dir = os.makedirs(current_run_dir_path, exist_ok=True)

        test_model(model, test_loader,criterion, device, current_run_dir)

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(description='Train and evaluate a 3D ResNet model.')
    parser.add_argument('--epochs', type=int, default= 30, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default= 24, help='Batch size for training and evaluation.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument("--patience", type = int, default= 10, help = "patience for early stop")
    parser.add_argument("--test",type = bool, default = False, help= "if True, we load a model and test on full datasets")
    parser.add_argument("--block_size", type = tuple,default = (3,3,3), help = 'The block size' )
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start training from, useful for resuming training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    
    main(args)