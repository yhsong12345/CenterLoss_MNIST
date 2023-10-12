import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from engine import *
from model import *
from datasets import create_datasets, create_data_loaders
from utils import *
import os




# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=100,
    help='number of epochs to train our network for')
parser.add_argument('-lr', '--learning_rate', type=float, 
                    default=0.001, help='learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=128,
                    help='Batch Size')
parser.add_argument('-al', '--alpha', type=float, help='alpha', default=0.5)
parser.add_argument('-lam', '--lambda', type=float, help='lambda', default=1)
parser.add_argument('-d', '-sav_dir', type=str, dest='save_dir', help='directory', default='outputs')
args = vars(parser.parse_args())


# learning_parameters 
lr = args['learning_rate']
epochs = args['epochs']
BATCH_SIZE = args['batch_size']
d = args['save_dir']
a = args['alpha']
lam = args['lambda']
print_freq = 50




# get the training, validation and test_datasets
train_dataset, valid_dataset, test_dataset = create_datasets()
# get the training and validaion data loaders
train_loader, valid_loader, _ = create_data_loaders(
    train_dataset, valid_dataset, test_dataset, BATCH_SIZE
)



# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

Model = LeNetspp()
# print(Model)

# build the model
model = Model.to(device)
print(model)
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")

# loss function
L_S = nn.CrossEntropyLoss()
L_C = CenterLoss(10, 2).to(device)

# if h:
#     model.half()
#     criterion.half()

# optimizer
optimizer_S = optim.SGD(model.parameters(), lr=lr,
                      momentum=0.9, weight_decay=0.0001)

optimizer_C = optim.SGD(L_C.parameters(), lr=a)

lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer_S,
                                                    step_size=20, gamma=0.5)
lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer_C,
                                                    step_size=150, gamma=0.5)



# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

# start the training
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, epoch, train_loader, 
                                            optimizer_S, optimizer_C, lam, 
                                            L_S, L_C, num_classes=10)
    valid_epoch_loss, valid_epoch_acc = validate(model, epoch, valid_loader,  
                                                lam, L_S, L_C, num_classes=10)
    lr_scheduler1.step()
    lr_scheduler2.step()
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss}, validation acc: {valid_epoch_acc:.3f}")
    # save the best model till now if we have the least loss in the current epoch
    print('-'*50)
    
print('TRAINING COMPLETE')