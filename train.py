import torch
import torch.nn as nn
import time
from torchvision import datasets, transforms
import torchvision.models as models
import torch
import torch.nn as nn
import time
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import Dataset
import os
import pdb
from PIL import Image
import argparse
import csv
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np

from custom_model import ResNet, BasicBlock
import cv2
from PIL import Image
parser = argparse.ArgumentParser(description='PyTorch  Training')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test model on final test set')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--train', dest='train', action='store_true',
                    help='normal train')
parser.add_argument('--all-train', dest='all_train', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--single-evaluate', dest='single_evaluate', action='store_true',
                    help='evaluate one image')


class AverageMeter(object):
    """Computes and stores the average and current"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def resnet18(num_classes):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[2, 2, 2, 2],
                   num_classes=num_classes,
                   grayscale=True)
    return model


def train_val_dataset(dataset, val_split=0.10):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def load_model(model_path, model_sr, optimizer_sr=None):
    try:
        checkpoint = torch.load(model_path)
        model_sr.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
    except TypeError:
        load_dict = torch.load(model_path).state_dict()
        model_dict = model_sr.state_dict()
        model_dict.update(load_dict)
        model_sr.load_state_dict(model_dict) 
        epoch = None
    if optimizer_sr:
        optimizer_sr.load_state_dict(checkpoint['optimizer_state_dict'])   
    return model_sr


def adjust_learning_rate(optimizer, orig_lr, epoch, total_epoch):

    first = total_epoch * 2/3
    second = total_epoch* 5/6
    third = total_epoch

    if epoch < first:
        lr = orig_lr
    elif epoch < second:
        lr = orig_lr * 0.1
    elif epoch < third:
        lr = orig_lr * 0.01
    # print("epoch:  {}, learning-rate: {}".format(str(epoch), str(lr)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    torch.manual_seed(0)
    global args
    args = parser.parse_args()
    model = resnet18(13065)

    train_dir = "/home/jwang/handwriting_data_all/cleaned_data"
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    #Applying Transformation
    train_transforms = transforms.Compose([
                                    transforms.Resize(100),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.RandomResizedCrop(size=90, scale=(0.9, 1.0)),
                                    transforms.Resize(100),
                                    transforms.RandomAffine(degrees=15, fillcolor=255, translate=(0.1, 0.1), scale=(0.95, 1.15)),
                                    transforms.RandomRotation(degrees=15, fill=255),
                                    transforms.ToTensor(),
                                    ])

    val_transforms = transforms.Compose([                                    
                                    transforms.Resize(100),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    ])

    dataset_original = datasets.ImageFolder(train_dir)
    train_val_dict = train_val_dataset(dataset_original)
    trainsubset= train_val_dict["train"]
    valsubset = train_val_dict["val"]

    trainset = DatasetFromSubset(trainsubset, train_transforms)
    valset = DatasetFromSubset(valsubset, val_transforms)

    # train_debug_set = Subset(trainset, torch.arange(1000))
    # trainset= train_debug_set
    # val_debug_set = Subset(valset, torch.arange(1000))
    # valset =val_debug_set

    #Data Loading
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=100)

    #debug: Check dataset transformations >
    # from torchvision.utils import save_image
    # iterator = iter(train_loader)
    # for i in range(15):
    #     x_batch, y_batch = iterator.next()
    #     save_image(x_batch, 'img{}.png'.format(str(i)))
    # pdb.set_trace()
    #debug: Check dataset transformations <

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), 0.01,
                                momentum=0.9,
                                weight_decay=0.001)

    log = []

    if args.test:
        print("!!!!!!!!!!!!test!!!!!!!!!!!!")
        model.train()
        model = load_model("/home/jwang/ming/trained/all_train_model_cls.pth", model)
        model.eval()
        test_result_name = "final_all_train_test_result.csv"
        result_list = final_test(final_test_loader, model)
        listtocsv(result_list, test_result_name)

    elif args.evaluate:
        print("!!!!!!!!!!!!evaluate!!!!!!!!!!!!")
        model.train()
        model = load_model("/home/jwang/ming/trained/model_cls.pth", model)
        model.eval()
        top1 = validate(val_loader, model, criterion)
        print(" val top1: ", top1)


    elif args.train:
        print("!!!!!!!!!!!!!!!!!normal train !!!!!!!!!!!!!!!!!!!")
        total_epoch = 6
        model_sr_out_path = "/home/jwang/CHhandwriting_char_recognition/" + "model_cls.pth"
        for epoch in range(0, total_epoch):
            print("===============epoch:{} ==================".format(epoch))
            log_tmp = []
            adjust_learning_rate(optimizer, 0.01, epoch, total_epoch)

            # train for one epoch
            top1, loss = train(train_loader, model, criterion, optimizer, epoch)
            log_tmp += [top1, loss ]
            print("train top1:", top1, "  train loss: ", loss)

            top1, loss = validate(val_loader, model, criterion)
            log_tmp += [top1, loss ]

            print(" val top1:", top1, "  val loss: ", loss)
            log.append(log_tmp)
            np.savetxt(os.path.join("/home/jwang/CHhandwriting_char_recognition/", 'log.txt'), log)


        torch.save(model, model_sr_out_path)

    elif args.all_train:
        print("!!!!!!!!!!!!!all train!!!!!!!!!!!!!!!!!")
        total_epoch = 30
        model_out_path = "/home/jwang/ming/" + "all_train_model_cls.pth"
        for epoch in range(0, total_epoch):
            adjust_learning_rate(optimizer, 0.01, epoch, total_epoch)

            # train for one epoch
            top1 = train(all_train_loader, model, criterion, optimizer, epoch)
            print("train top1:", top1)
            # top1 = validate(val_loader, model, criterion)
            # print(" val top1:", top1)
        torch.save(model, model_out_path)

    elif args.single_evaluate:
        image = cv2.imread("example_0.png")
        image = val_transforms(Image.fromarray(image))
        image = torch.unsqueeze(image, 0)
        model.train()
        model = load_model("/home/jwang/CHhandwriting_char_recognition/0403_trained/model_cls.pth", model)
        model.eval()

        pred = single_evaluate(image, model)
        print("class: ", pred)


def train_val_dataset(dataset, val_split=0.10):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    train_losses, test_losses = [], []
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        images = images.cuda()
        target = target.cuda()
        data_time.update(time.time() - end)

        output, __ = model(images)

        loss = criterion(output, target)
     

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print("top1 acc: ", top1.avg.cpu().data.numpy())
            print("train loss: ", losses.avg)
            
    return top1.avg, losses.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute output
            images = images.cuda()
            target = target.cuda()
            output, __ = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
#             top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 200 == 0:
                print("val top1 acc: ", top1.avg.cpu().data.numpy())
                print("val loss: ", loss.item())

    print("val accuracy: ", top1.avg)

    return top1.avg, losses.avg

def single_evaluate(image, model):
    model.eval()

    with torch.no_grad():
        image = image.cuda()
        output, __ = model(image)
        pred = np.argmax(output.cpu().numpy())
        print("class: ", pred)

    return pred

if __name__ == "__main__":
    main()
