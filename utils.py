import torch
import matplotlib.pyplot as plt
import os
import pandas as pd



            

plt.style.use('ggplot')


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_acc=float(0)
    ):
        self.best_valid_acc = best_valid_acc
        
    def __call__(
        self, current_valid_acc,
        epoch, model, optimizer, criterion
    ):
        path = f'./outputs/LeNetspp'
        if current_valid_acc > self.best_valid_acc:
            self.best_valid_acc = current_valid_acc
            print(f"\nBest validation acc: {self.best_valid_acc}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'{path}/best_model.pt')


def save_model(epoch, model, optimizer, criterion):
    print(f'Saving final model...')
    path = f'./outputs/LeNetspp'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        }, f'{path}/model{epoch}.pt')



def save_data(train_acc, valid_acc, train_loss, valid_loss):
    print("Saving losses and accuracies")
    path = f'./outputs/LeNetspp'
    data = {'train_accuracy': train_acc, 'valid_accuracy':valid_acc,
            'train_loss': train_loss, 'valid_loss': valid_loss}
    df = pd.DataFrame(data=data)
    df.to_excel(f'{path}/LeNetsppresult.xlsx')

    


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    path = f'./outputs/LeNetspp'
    plt.figure(figsize=(10,7))
    plt.plot(
        train_acc, color='red', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validation accuracy'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{path}/accuracy.png')


    plt.figure(figsize=(10,7))
    plt.plot(
        train_loss, color='green', linestyle='--',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='orange', linestyle='--',
        label='validation loss'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{path}/loss.png')


def plot_features(features, labels, num_classes, epoch, prefix):
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for i in range(num_classes):
        plt.scatter(
            features[labels==i, 0],
            features[labels==i, 1],
            c = colors[i],
            s=.1
        )
    plt.legend(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'], loc='upper right')
    if prefix:
        dir = './plots/train'
        if not os.path.exists(dir):
            os.mkdir(dir)
        plt.savefig(f'{dir}/{epoch+1}.png')
    else:
        dir ='./plots/test'
        if not os.path.exists(dir):
            os.mkdir(dir)
        plt.savefig(f'{dir}/{epoch+1}.png')