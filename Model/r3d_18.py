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
import argparse
import logging
import sys
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Lambda
sys.path.append("./")
from src.utils import scale_up_block
# import sys
# sys.path.append('..')
# from dataloader import *
# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename = 'r3d_18.log', filemode= 'a')

# Create a logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the minimum logging level

# Create a file handler that logs messages to a file
file_handler = logging.FileHandler(filename = 'r3d_18.log', mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Create a stream handler that logs messages to the console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

#load data
class CustomDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform  # Can include scaling as part of the transformations

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)

        sample = np.expand_dims(sample, axis=0)

        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
def load_data():
    pet_train_features = np.load('./Data/train_features.npy', allow_pickle = True)
    pet_valid_features = np.load('./Data/valid_features.npy', allow_pickle = True)
    pet_test_features = np.load('./Data/test_features.npy', allow_pickle = True)

    ct_train_features = np.load('./Data/train_ct_features.npy', allow_pickle = True)
    ct_valid_features = np.load('./Data/valid_ct_features.npy', allow_pickle = True)
    ct_test_features = np.load('./Data/test_ct_features.npy', allow_pickle = True)

    train_labels = np.load('./Data/train_labels.npy', allow_pickle = True)
    valid_labels = np.load('./Data/valid_labels.npy', allow_pickle = True)
    test_labels = np.load('./Data/test_labels.npy', allow_pickle = True)
    return {'pet_train_features': pet_train_features,
            'pet_valid_features': pet_valid_features,
            'pet_test_features': pet_test_features,
            'ct_train_features': ct_train_features,
            'ct_valid_features': ct_valid_features,
            'ct_test_features': ct_test_features,
            'train_labels': train_labels,
            'valid_labels': valid_labels,
            'test_labels': test_labels
    }


def create_balanced_loaders(data, num_samples_train, num_samples_valid, num_samples_test):
    """
    Create DataLoader objects for training, validation, and test datasets with balanced classes.

    Args:
    - data (dict): A dictionary containing the features and labels for training, validation, and test sets.
    - num_samples_train (int): Number of samples per class in the training set.
    - num_samples_valid (int): Number of samples per class in the validation set.
    - num_samples_test (int): Number of samples per class in the test set.


    Returns:
    - Tuple of DataLoader objects for the training, validation, and test datasets.
    """

    # Utility function to select balanced data
    def select_balanced_data(features, labels, num_samples):
        class_0_indices = np.where(labels == 0)[0]
        class_1_indices = np.where(labels == 1)[0]
        num_samples = min(len(class_0_indices), len(class_1_indices), num_samples)
        selected_indices_0 = np.random.choice(class_0_indices, num_samples, replace=False)
        selected_indices_1 = np.random.choice(class_1_indices, num_samples, replace=False)
        selected_indices = np.concatenate((selected_indices_0, selected_indices_1))
        np.random.shuffle(selected_indices)  # Shuffle to mix classes
        return features[selected_indices], labels[selected_indices]

    # Select balanced training data
    selected_train_features, selected_train_labels = select_balanced_data(data['pet_train_features'], data['train_labels'], num_samples_train)

    # Select balanced validation data
    selected_valid_features, selected_valid_labels = select_balanced_data(data['pet_valid_features'], data['valid_labels'], num_samples_valid)

    # Select balanced test data
    selected_test_features, selected_test_labels = select_balanced_data(data['pet_test_features'], data['test_labels'], num_samples_test)
    print(selected_train_features.shape, selected_train_labels.sum())
    print(selected_valid_features.shape, selected_valid_labels.sum())
    print(selected_test_features.shape, selected_test_labels.sum())


    mean = np.mean(selected_train_features)  # Calculate the mean of the training data
    std = np.std(selected_train_features)  

    transform = Compose([
        Lambda(lambda x: scale_up_block(x, new_resol=[112, 112, 112])),
        Lambda(lambda x: (x - mean) / std) 
        # Add other transforms here as needed
    ])

    # Create custom datasets
    train_dataset = CustomDataset(selected_train_features, selected_train_labels, transform=transform)
    valid_dataset = CustomDataset(selected_valid_features, selected_valid_labels, transform=transform)
    test_dataset = CustomDataset(selected_test_features, selected_test_labels, transform=transform)

    return train_dataset, valid_dataset, test_dataset


def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, patience=5):
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    best_valid_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        logging.info(f"epoch {epoch} starts")
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
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

            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            # print("Outputs shape:", outputs.shape)
            # print("Outputs shape:",outputs.squeeze(-1).shape)

            # print("Labels shape:", labels.shape)
            loss = criterion(outputs.squeeze(-1), labels.float())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = torch.round(torch.sigmoid(outputs.squeeze(-1)))
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        # print(f"Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}, Training Accuracy:{}")

        # Validation loop after each epoch
        model.eval()  # Set model to evaluation mode
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():  # No need to track gradients during validation
            for inputs, labels in tqdm(valid_loader):
                inputs = torch.cat([inputs]*3, dim=1) 
                # inputs = inputs.repeat(1, 3, 1, 1, 1)  # Repeat channels for validation inputs
                # print(inputs.shape)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                valid_loss += criterion(outputs.squeeze(-1), labels.float()).item()
                predicted = torch.round(torch.sigmoid(outputs.squeeze(-1)))  # Binary classification
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()
        # accuracy = 100 * correct / total
        # print(f'Epoch {epoch+1}, Validation Accuracy: {accuracy:.2f}%')
        epoch_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(epoch_valid_loss)
        valid_accuracy = 100 * correct_valid / total_valid
        valid_accuracies.append(valid_accuracy)
        # logging.info(f"Epoch {epoch+1}, Validation Loss: {epoch_valid_loss:.4f}, Accuracy: {valid_accuracy:.2f}%")
        #  Early stopping logic
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = model.state_dict()  # Save the best model state
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logging.info(f'Validation loss did not improve for {epochs_no_improve}/{patience} epochs.')

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
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs.squeeze(-1), labels.float())
            test_loss += loss.item()

            predicted = torch.round(torch.sigmoid(outputs.squeeze(-1)))
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

    return test_loss, test_accuracy  #, precision, recall, f1

def save_plots(train_losses, train_accuracies, valid_losses, valid_accuracies, num_epochs):
    # Plot training and validation losses
    plt.figure(figsize=(10, 7))
    # num_epochs = min(num_epochs, len(train_losses))
    epochs_to_plot = min(num_epochs, len(train_losses))
    plt.plot(range(1, epochs_to_plot + 1), train_losses[:epochs_to_plot], label='Training Loss')
    plt.plot(range(1, epochs_to_plot + 1), valid_losses[:epochs_to_plot], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_validation_loss.png')
    plt.close()

    # Plot training and validation accuracies
    plt.figure(figsize=(10, 7))
    plt.plot(range(1, epochs_to_plot + 1), train_accuracies[:epochs_to_plot], label='Training Accuracy')
    plt.plot(range(1, epochs_to_plot + 1), valid_accuracies[:epochs_to_plot], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('training_validation_accuracy.png')
    plt.close()

    logging.info('Plots saved: "training_validation_loss.png" and "training_validation_accuracy.png"')

# Example usage:
# save_plots(train_losses, train_accuracies, valid_losses, valid_accuracies, num_epochs)

def main(args):
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load the data
    data = load_data()

    train_dataset, valid_dataset, test_dataset  = create_balanced_loaders(data, args.train_data_size_per_class, 
                                                                          args.valid_data_size_per_class,
                                                                          args.test_data_size_per_class)
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    # Initialize the model
    model = r3d_18(weights=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if not args.test:
    # Train the model
        model, train_losses, train_accuracies, valid_losses, valid_accuracies = train_model(model, train_loader, valid_loader, criterion, optimizer, args.epochs, device, args.patience)
        
        # Save the model
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_filename = f'model_after_training_{current_time}.pth'
        torch.save(model.state_dict(), model_filename)
        # Test the model
        # print(device)
        test_model(model, test_loader,criterion, device)

        # Save training/validation plots
        save_plots(train_losses, train_accuracies, valid_losses, valid_accuracies, args.epochs)

    else:
        path = "/home/ubuntu/jupyter-sandy/3D_ResNet/model_after_training_2024-03-07_14-38-49.pth"
        model.load_state_dict(torch.load(path))
        test_model(model, test_loader,criterion, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a 3D ResNet model.')
    parser.add_argument('--epochs', type=int, default= 50, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and evaluation.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument("--patience", type = int, default= 5, help = "patience for early stop")
    parser.add_argument('--train_data_size_per_class', type= float,default = np.inf, help = 'train_data_size_per_class')
    parser.add_argument('--valid_data_size_per_class', type= float,default = np.inf, help = 'valid_data_size_per_class')
    parser.add_argument('--test_data_size_per_class', type=float,default = np.inf, help = 'test_data_size_per_class')
    parser.add_argument("--test",type = bool, default = False, help= "if True, we load a model and test on full datasets")
    
    # Add other command line arguments as needed
    args = parser.parse_args()
    logging.info(f"working with train_data_size_per_class: {args.train_data_size_per_class}")
    logging.info(f"working with valid_data_size_per_class: {args.valid_data_size_per_class}")
    logging.info(f"working with test_data_size_per_class: {args.test_data_size_per_class}")
    main(args)