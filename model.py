import argparse
import cv2
import json
import numpy as np
import os
import pandas as pd
import scipy.misc
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torchvision
import torchvision.models as models

from PIL import Image
from averagemeter import *
from models import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.utils.data import sampler
from torchvision import datasets
from torchvision import transforms

# GLOBAL CONSTANTS
INPUT_SIZE = 224
NUM_CLASSES = 185

USE_CUDA = torch.cuda.is_available()
best_prec1 = 0
classes = []

#TODO 
#   1) if you stop half way trough an epoc, save last average values
#   2) make predicted_labels include the same timestamp as the data and save them in the same folder

# ARGS Parser
# when you don't want to pass a checkpoint: --resume ""
parser = argparse.ArgumentParser(description='PyTorch LeafSnap Training')
parser.add_argument('--resume', required = True, type=str, metavar='PATH',
                    help='path to latest checkpoint (type '' for none)')
parser.add_argument('--modelid', required = True, type=int, metavar='MODEL_ID',
                    help='1(resnet18), 2(VGG16), 3(resnet101), 4(densenet121)')
args = parser.parse_args()
MODEL_ID = args.modelid 

# Model selection function 
def selectModel(MODEL_ID):
    if MODEL_ID == 1:
        BATCH_SIZE = 128
        NUM_EPOCHS = 100
        LEARNING_RATE = 1e-1 #start from learning rate after 40 epochs
        ALPHA = 6
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, NUM_CLASSES)
        modelName = "resnet18_decay"
    elif MODEL_ID == 2:
        # BATCH_SIZE = 128
        # NUM_EPOCHS = 250
        # LEARNING_RATE = 1e-5 #start from learning rate after 40 epochs
        # ALPHA = 6
        model = models.VGG('VGG16')
        # model.fc = nn.Linear(512, NUM_CLASSES)
        modelName = "VGG16"
    elif MODEL_ID == 3:
        # BATCH_SIZE = 128
        # NUM_EPOCHS = 250
        # LEARNING_RATE = 1e-5 #start from learning rate after 40 epochs
        # ALPHA = 6
        model = models.resnet101()
        model.fc = nn.Linear(2048, NUM_CLASSES)
        modelName = "resnet101"
    else:
        BATCH_SIZE = 8
        # NUM_EPOCHS = 250
        # LEARNING_RATE = 1e-5 #start from learning rate after 40 epochs
        # ALPHA = 6
        model = models.densenet121()
        # model.fc = nn.Linear(512, NUM_CLASSES)
        modelName = "densenet121"
    return model, modelName, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, ALPHA

# Create data file with header
def createHeadertxt_train(modelName, INPUT_SIZE, filename):
    with open(filename, 'a') as a:
        a.write('#Epoch\t\t   Time\t\t            Data\t\t   Loss\t\t\t\t   Prec@1\t   Prec@5 \n')

def createHeadertxt_dev(modelName, INPUT_SIZE, filename):
    with open(filename, 'a') as a:
        a.write('#Epoch\t\t    Time\t\t    Loss\t\t   Prec@1\t\t   Prec@5 \n')
    
# Training method which trains model for 1 epoch
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    
    # for i, (input, target) in enumerate(train_loader):
    for i, data in enumerate(train_loader):
        (input,target),(path,_) = data
        # measure data loading time
        if USE_CUDA:
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5), path=path, minibatch = i)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  '\Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)) 

        if i == (len(train_loader)-1):
            with open(filename_train, 'a') as a:
                    a.write('{0}\t'
                            '{batch_time.avg:16.3f}\t'
                            '{data_time.avg:16.3f}\t'
                            '{loss.avg:16.4f}\t'
                            '{top1.avg:16.3f}\t'
                            '{top5.avg:16.3f}\n'.format(
                                epoch, batch_time=batch_time,
                                data_time=data_time, loss=losses, top1=top1, top5=top5))

# Validation method
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    class_correct = list(0. for i in range(NUM_CLASSES))
    class_total = list(0. for i in range(NUM_CLASSES))

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    for i, (input, target) in enumerate(val_loader):
        if USE_CUDA:
            input = input.cuda(async=True)
            target = target.cuda(async=True)
        with torch.no_grad(): 
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))
        if i == (len(val_loader)-1):    
            with open(filename_dev, 'a') as a:
                    a.write('{0}\t'
                            '{batch_time.avg:16.3f}\t'
                            '{loss.avg:16.4f}\t'
                            '{top1.avg:16.3f}\t'
                            '{top5.avg:16.3f}\n'.format(
                                epoch, batch_time=batch_time,
                                loss=losses, top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print('\n[INFO] Saved Model to model_best.pth.tar')
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LEARNING_RATE*0.1** (epoch // ALPHA)
    if (lr <= 1e-4): # cap the learning rate to be larger than e-4
        lr = 1e-4
    print('\n[Learning Rate] {:0.6f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,), path = None, minibatch = None):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    ## save mislabeled data
    if path:
        filename = [os.path.basename(p) for p in path]
        true_label = [os.path.basename(os.path.dirname(p)) for p in path]
        pred_label = [classes[p] for p in pred[0]]
        data = np.array([filename, true_label, pred_label])
        out = pd.DataFrame(data.T,columns =['filename', 'true_label','pred_label'])
        out.index.name = 'index'
        out['correct?'] = out['pred_label']==out['true_label']
        out_file = 'predicted_labels.csv'

        if os.path.isfile(out_file):
            if minibatch==0: # if first minibatch, overwrite existing file
                out.to_csv(out_file)
            else:
                df = pd.read_csv(out_file, index_col = 0)
                df = pd.concat([df,out],axis = 0, ignore_index = True)
                df.to_csv(out_file)
        else: # if file does not exist, make file
            out.to_csv(out_file)
    
    return res

class MyImageFolder(datasets.ImageFolder): #return image path and loader
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index), self.imgs[index]

###############################################################################

print('\n[INFO] Creating Model')
model, modelName, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, ALPHA = selectModel(MODEL_ID)

criterion = nn.CrossEntropyLoss()
if USE_CUDA:
    model = torch.nn.DataParallel(model).cuda()
    criterion = criterion.cuda()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                      momentum=0.9, weight_decay=1e-4, nesterov=True)
#optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), 
#                       eps=1e-08, weight_decay=1e-4)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
        print("=> Loaded model Prec1 = %0.2f%%"%best_prec1)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print('\n[INFO] Reading Training and Testing Dataset')
traindir = os.path.join('dataset', 'train_%d'%INPUT_SIZE)
testdir = os.path.join('dataset', 'test_%d'%INPUT_SIZE)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
data_train = MyImageFolder(traindir, transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]))
data_test = datasets.ImageFolder(testdir, transforms.Compose([
            transforms.ToTensor(),
            normalize]))
classes = data_train.classes
classes_test = data_test.classes

train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4) 

print('\n[INFO] Preparing txt files to save epoch data')
timestamp_string = time.strftime("%Y%m%d-%H%M%S") 
filename_train = './dataAndPlots/' + timestamp_string + '_train' + '_' + modelName + '_' + str(INPUT_SIZE) + '.txt'
filename_dev = './dataAndPlots/' + timestamp_string + '_dev' + '_' + modelName + '_' + str(INPUT_SIZE) + '.txt'
createHeadertxt_train(modelName, INPUT_SIZE, filename_train)
createHeadertxt_dev(modelName, INPUT_SIZE, filename_dev)

print('\n[INFO] Training Started')
for epoch in range(1, NUM_EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch)
    # evaluate on validation set
    prec1 = validate(val_loader, model, criterion)

    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best)
    print('\n[INFO] Saved Model to leafsnap_model.pth')    
    torch.save(model, 'leafsnap_model.pth')

print('\n[DONE]')
