import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from model import *
from datasets import create_datasets, create_data_loaders
from utils import *
import torch.backends.cudnn as cudnn
import torchsummary
import torchvision



# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=200,
    help='number of epochs to train our network for')
parser.add_argument('-lr', '--learning_rate', type=float, 
                    default=0.1, help='learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=128,
                    help='Batch Size')
parser.add_argument('-al', '--alpha', type=int, help='alpha', default=0.5)
parser.add_argument('-lam', '--lambda', type=int, help='lambda', default=1)
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

# # get the training, validation and test_datasets
# train_dataset, valid_dataset, test_dataset = create_datasets(s)
# # get the training and validaion data loaders
# train_loader, valid_loader, _ = create_data_loaders(
#     train_dataset, valid_dataset, test_dataset, BATCH_SIZE
# )

# p = os.getcwd()
# path = p + f'/outputs/{m}'
# os.mkdir(path)


# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

model = LeNetspp()
model.to(device)

torchsummary.summary(model, (1, 28, 28))
# print(model)

# Model = SelectModel(m)

# # build the model
# model = Model.to(device)
# print(model)
# # total parameters and trainable parameters
# total_params = sum(p.numel() for p in model.parameters())
# print(f"{total_params:,} total parameters.")
# total_trainable_params = sum(
#     p.numel() for p in model.parameters() if p.requires_grad)
# print(f"{total_trainable_params:,} training parameters.\n")
# # optimizer
# optimizer = optim.SGD(model.parameters(), lr=lr,
#                       momentum=0.9, weight_decay=0.0001)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                     milestones=[100, 150])
# # loss function
# criterion = nn.CrossEntropyLoss()
# # initialize SaveBestModel class
# save_best_model = SaveBestModel()


## training
# def train(model, trainloader, optimizer, criterion):
#     model.train()
#     print('Training')
#     train_running_loss = 0.0
#     train_running_correct = 0
#     counter = 0
#     for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
#         counter += 1
#         image, labels = data
#         image = image.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         # forward pass
#         outputs = model(image)
#         # calculate the loss
#         loss = criterion(outputs, labels)
#         train_running_loss += loss.item()
#         # calculate the accuracy
#         _, preds = torch.max(outputs.data, 1)
#         train_running_correct += (preds == labels).sum().item()
#         # backpropagation
#         loss.backward()
#         # update the optimizer parameters
#         optimizer.step()
    
#     # loss and accuracy for the complete epoch
#     epoch_loss = train_running_loss / counter
#     epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
#     return epoch_loss, epoch_acc



# # validation
# def validate(model, testloader, criterion):
#     model.eval()
#     print('Validation')
#     valid_running_loss = 0.0
#     valid_running_correct = 0
#     counter = 0
#     with torch.no_grad():
#         for i, data in tqdm(enumerate(testloader), total=len(testloader)):
#             counter += 1
            
#             image, labels = data
#             image = image.to(device)
#             labels = labels.to(device)
#             # forward pass
#             outputs = model(image)
#             # calculate the loss
#             loss = criterion(outputs, labels)
#             valid_running_loss += loss.item()
#             # calculate the accuracy
#             _, preds = torch.max(outputs.data, 1)
#             valid_running_correct += (preds == labels).sum().item()
        
#     # loss and accuracy for the complete epoch
#     epoch_loss = valid_running_loss / counter
#     epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
#     return epoch_loss, epoch_acc

# # lists to keep track of losses and accuracies
# train_loss, valid_loss = [], []
# train_acc, valid_acc = [], []

# # start the training
# for epoch in range(epochs):
#     print(f"[INFO]: Epoch {epoch+1} of {epochs}")
#     train_epoch_loss, train_epoch_acc = train(model, train_loader, 
#                                             optimizer, criterion)
#     valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
#                                                 criterion)
#     lr_scheduler.step()
#     train_loss.append(train_epoch_loss)
#     valid_loss.append(valid_epoch_loss)
#     train_acc.append(train_epoch_acc)
#     valid_acc.append(valid_epoch_acc)
#     print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
#     print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
#     save_model(m, epoch, model, optimizer, criterion)
#     # save the best model till now if we have the least loss in the current epoch
#     save_best_model(
#         m, valid_epoch_loss, epoch, model, optimizer, criterion
#     )
#     print('-'*50)
    
# # save the trained model weights for a final time
# # save_model(m, epochs, model, optimizer, criterion)
# save_data(m, train_acc, valid_acc, train_loss, valid_loss)
# # save the loss and accuracy plots
# save_plots(m, train_acc, valid_acc, train_loss, valid_loss)
# print('TRAINING COMPLETE')


# def main():
#     global args, best_prec1
#     args = parser.parse_args()


#     # Check the save_dir exists or not
#     if not os.path.exists(args.save_dir):
#         save_dir = args.save_dir + f"/{m}"
#         os.makedirs(save_dir)

#     save_dir = args.save_dir + f"/{m}"

#     # model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
#     # model.cuda()

#     Model = SelectModel(m)

#     # build the model
#     model = Model.to(device)
#     print(model)
#     # total parameters and trainable parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"{total_params:,} total parameters.")
#     total_trainable_params = sum(
#             p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"{total_trainable_params:,} training parameters.\n")

#     # # optionally resume from a checkpoint
#     # if args.resume:
#     #     if os.path.isfile(args.resume):
#     #         print("=> loading checkpoint '{}'".format(args.resume))
#     #         checkpoint = torch.load(args.resume)
#     #         args.start_epoch = checkpoint['epoch']
#     #         best_prec1 = checkpoint['best_prec1']
#     #         model.load_state_dict(checkpoint['state_dict'])
#     #         print("=> loaded checkpoint '{}' (epoch {})"
#     #               .format(args.evaluate, checkpoint['epoch']))
#     #     else:
#     #         print("=> no checkpoint found at '{}'".format(args.resume))

#     cudnn.benchmark = True


#     # define loss function (criterion) and optimizer
#     criterion = nn.CrossEntropyLoss().cuda()

#     if half:
#         model.half()
#         criterion.half()

#     optimizer = torch.optim.SGD(model.parameters(), lr,
#                                 momentum=0.9, weight_decay=0.0001)

#     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                         milestones=[100, 150])

#     # if args.arch in ['resnet1202', 'resnet110']:
#     #     # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
#     #     # then switch back. In this setup it will correspond for first epoch.
#     #     for param_group in optimizer.param_groups:
#     #         param_group['lr'] = args.lr*0.1



#     validate(valid_loader, model, criterion)


#     for epoch in range(epochs):

#         # train for one epoch
#         print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
#         train(train_loader, model, criterion, optimizer, epoch)
#         lr_scheduler.step()

#         # evaluate on validation set
#         prec1 = validate(valid_loader, model, criterion)

#         # remember best prec@1 and save checkpoint
#         is_best = prec1 > best_prec1
#         best_prec1 = max(prec1, best_prec1)

#         if epoch > 0 and epoch % 10 == 0:
#             save_checkpoint({
#                 'epoch': epoch + 1,
#                 'state_dict': model.state_dict(),
#                 'best_prec1': best_prec1,
#             }, is_best, filename=os.path.join(save_dir, 'checkpoint.pt'))

#         save_checkpoint({
#             'state_dict': model.state_dict(),
#             'best_prec1': best_prec1,
#         }, is_best, filename=os.path.join(save_dir, 'model.pt'))

# def train(train_loader, model, criterion, optimizer, epoch):
#     """
#         Run one train epoch
#     """
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()

#     # switch to train mode
#     model.train()

#     end = time.time()
#     for i, (input, target) in enumerate(train_loader):

#         # measure data loading time
#         data_time.update(time.time() - end)

#         target = target.cuda()
#         input_var = input.cuda()
#         target_var = target
#         if half:
#             input_var = input_var.half()

#         # compute output
#         output = model(input_var)
#         loss = criterion(output, target_var)

#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         output = output.float()
#         loss = loss.float()
#         # measure accuracy and record loss
#         prec1 = accuracy(output.data, target)[0]
#         losses.update(loss.item(), input.size(0))
#         top1.update(prec1.item(), input.size(0))

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % print_freq == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#                       epoch, i, len(train_loader), batch_time=batch_time,
#                       data_time=data_time, loss=losses, top1=top1))



# def validate(val_loader, model, criterion):
#     """
#     Run evaluation
#     """
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()

#     # switch to evaluate mode
#     model.eval()

#     end = time.time()
#     with torch.no_grad():
#         for i, (input, target) in enumerate(val_loader):
#             target = target.cuda()
#             input_var = input.cuda()
#             target_var = target.cuda()

#             if half:
#                 input_var = input_var.half()

#             # compute output
#             output = model(input_var)
#             loss = criterion(output, target_var)

#             output = output.float()
#             loss = loss.float()

#             # measure accuracy and record loss
#             prec1 = accuracy(output.data, target)[0]
#             losses.update(loss.item(), input.size(0))
#             top1.update(prec1.item(), input.size(0))

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if i % print_freq == 0:
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#                           i, len(val_loader), batch_time=batch_time, loss=losses,
#                           top1=top1))

#     print(' * Prec@1 {top1.avg:.3f}'
#           .format(top1=top1))

#     return top1.avg




# if __name__ == '__main__':
#     main()