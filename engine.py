import torch
from tqdm.auto import tqdm
import numpy as np
from model import *
from datasets import create_datasets, create_data_loaders
from utils import *


plot = True

# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Computation device: {device}\n")

# training
def train(model, epoch, trainloader, optimizer_S, optimizer_C, 
          lam, L_S, L_C, num_classes):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    if plot:
        all_features, all_labels = [], []

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        # forward pass
        features, outputs = model(image)
        # calculate the loss
        loss1 = L_C(features, labels)
        loss2 = L_S(outputs, labels)
        loss = lam*loss1 + loss2
        optimizer_S.zero_grad()
        optimizer_C.zero_grad()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer_S.step()
        for param in L_C.parameters():
            param.grad.data *= (1./lam)
        optimizer_C.step()

        if plot:
            all_features.append(features.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())
        
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        train_running_loss += loss.item()
    
    if plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, prefix=True)
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc



# validation
def validate(model, epoch, testloader, lam, L_S, L_C, num_classes):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    if plot:
        all_features, all_labels = [], []
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            features, outputs = model(image)
            # calculate the loss
            loss1 = L_C(features, labels)
            loss2 = L_S(outputs, labels)
            loss = lam*loss1 + loss2
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        if plot:
            all_features.append(features.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())
    
    if plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, prefix=False)
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc