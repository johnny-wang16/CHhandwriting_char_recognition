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
import natsort
from PIL import Image
import argparse
import csv
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
parser = argparse.ArgumentParser(description='PyTorch  Training')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test model on final test set')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--all-train', dest='all_train', action='store_true',
                    help='evaluate model on validation set')


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(8192, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas



def resnet18(num_classes):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[2, 2, 2, 2],
                   num_classes=num_classes,
                   grayscale=True)
    return model

# def listtocsv(csv_lst, file_name):
#     print("saving to " + file_name)
#     with open(file_name, mode='w') as cls_file:
#         cls_writer = csv.writer(cls_file, delimiter=',')
#         cls_writer.writerow(csv_lst)
#     print("finished saving to " + file_name)

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
    # model = models.__dict__["resnet18"](pretrained=False)

    # # for param in model.parameters():
    # #     param.requires_grad = False
    # model.fc = nn.Sequential(nn.Linear(13065, 13065),
    #                                  nn.LogSoftmax(dim=1))
    model = resnet18(13065)

    # pdb.set_trace()
    # model.fc = nn.Sequential(nn.Linear(512, 128),
    #                                  nn.ReLU(),
    #                                  nn.Linear(128, 20),
    #                                  nn.LogSoftmax(dim=1))
    # model.fc = nn.Linear(512, 20)

    train_dir = "/home/jwang/handwriting_data_all/cleaned_data"
    # val_dir = "/home/jwang/ming/20_categories_training/20_categories_trainingn"
    model.train()
    #Applying Transformation
    # normalize = transforms.Normalize(mean=[0.8693],
    #                                      std=[0.3])
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

    # dataset_original = datasets.ImageFolder(train_dir, train_transforms)
    dataset_original = datasets.ImageFolder(train_dir)

    train_val_dict = train_val_dataset(dataset_original)
    trainsubset= train_val_dict["train"]
    valsubset = train_val_dict["val"]

    trainset = DatasetFromSubset(trainsubset, train_transforms)
    valset = DatasetFromSubset(valsubset, val_transforms)


    #Data Loading
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=3)
    #debug: Check dataset transformations >
    # from torchvision.utils import save_image
    # iterator = iter(train_loader)
    # for i in range(15):
    #     x_batch, y_batch = iterator.next()
    #     save_image(x_batch, 'img{}.png'.format(str(i)))
    # pdb.set_trace()
    #debug: Check dataset transformations <

    criterion = nn.CrossEntropyLoss()

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


    elif (not args.all_train) and (not args.evaluate) and (not args.test):
        print("!!!!!!!!!!!!!!!!!normal train !!!!!!!!!!!!!!!!!!!")
        total_epoch = 120
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
        total_epoch = 60
        model_out_path = "/home/jwang/ming/" + "all_train_model_cls.pth"
        for epoch in range(0, total_epoch):
            adjust_learning_rate(optimizer, 0.01, epoch, total_epoch)

            # train for one epoch
            top1 = train(all_train_loader, model, criterion, optimizer, epoch)
            print("train top1:", top1)

            # top1 = validate(val_loader, model, criterion)
            # print(" val top1:", top1)

        torch.save(model, model_out_path)


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
        data_time.update(time.time() - end)


        # compute output
#         print("images shape: ", images.shape)
#         print("targets shape: ", target.shape)
        output, __ = model(images)

        # if i == 1:
        #     pdb.set_trace()

        # print("output: ", output)
        loss = criterion(output, target)
        # print("outpt prob: ", output)
        # print("output prob shape: ", output.shape)
        # print("    target: ", target)
        # print("      loss: ", loss.item())

        # pdb.set_trace()
#         print(target.numpy())
        

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


        if i % 200 == 0:
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
            output = model(images)
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

        # TODO: this should also be done with the ProgressMeter
#         print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
#               .format(top1=top1, top5=top5))
    print("val accuracy: ", top1.avg)

    return top1.avg, losses.avg

# def final_test(val_loader, model):
#     prediction_list = [-1 for i in range(716)]


#     # switch to evaluate mode
#     model.eval()

#     with torch.no_grad():
#         end = time.time()
#         for i, (images, index) in enumerate(val_loader):
#             # compute output
#             output = model(images)
#             __, pred= output.topk(1, 1, True, True)
#             prediction_list[index] = pred.item()

#             # measure accuracy and record loss
#             # acc1, acc5 = accuracy(output, target, topk=(1, 5))

# #             top5.update(acc5[0], images.size(0))

#     return prediction_list

#  class CustomDataSet(Dataset):
#     def __init__(self, main_dir, transform):
#         self.main_dir = main_dir
#         self.transform = transform
#         all_imgs = os.listdir(main_dir)
#         self.total_imgs = natsort.natsorted(all_imgs)

#     def __len__(self):
#         return len(self.total_imgs)

#     def __getitem__(self, idx):
#         img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
#         image = Image.open(img_loc).convert("RGB")
#         tensor_image = self.transform(image)
#         return tensor_image, idx
if __name__ == "__main__":
    main()
