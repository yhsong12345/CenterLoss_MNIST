import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import os






parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', nargs='+', default='Resnet20' ,type=str, help='Models')
args = vars(parser.parse_args())


ta = []
tl = []
va = []
vl = []

m = args['model']


for i in range(len(m)):
    path = f'./outputs/{m[i]}/{m[i]}result.xlsx'

    df = pd.read_excel(path, engine='openpyxl', sheet_name='Sheet1')
    train_accuracy = df['train_accuracy']
    train_loss = df['train_loss']
    valid_accuracy = df['valid_accuracy']
    valid_loss = df['valid_loss']

    ta.append(np.array(train_accuracy))
    tl.append(np.array(train_loss))
    va.append(np.array(valid_accuracy))
    vl.append(np.array(valid_loss))


plt.figure(figsize=(25,15))
plt.subplot(1, 2, 1)


for i in range(len(m)):
    col = (np.random.random(), np.random.random(), np.random.random())
    plt.plot(
        tl[i], color=col, linestyle='-', label=f'{m[i]}'
    )

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train Loss')



plt.subplot(1, 2, 2)


for i in range(len(m)):
    col = (np.random.random(), np.random.random(), np.random.random())
    plt.plot(
        ta[i], color=col, linestyle='-', label=f'{m[i]}'
    )

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train Accuracy')
plt.savefig(f'./plots/Train{m}')




plt.figure(figsize=(25,15))
plt.subplot(1, 2, 1)

for i in range(len(m)):
    col = (np.random.random(), np.random.random(), np.random.random())
    plt.plot(
        va[i], color=col, linestyle='-', label=f'{m[i]}'
    )

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Test Accuracy')




plt.subplot(1, 2, 2)


for i in range(len(m)):
    col = (np.random.random(), np.random.random(), np.random.random())
    plt.plot(
        vl[i], color=col, linestyle='-', label=f'{m[i]}'
    )

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Test Loss')
plt.savefig(f'./plots/Test{m}')


print('PLOTTING COMPLETE')