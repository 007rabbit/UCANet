import torch
import torch.nn as nn
import torch.optim as optim
import Cnnp_dnet
import numpy as np
from torchvision import datasets, transforms
import argparse
import torch.utils.data
import torch.nn.functional as F  
import time
import random

parser = argparse.ArgumentParser(description='train UCANet')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for trainning')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train ')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate of train')
parser.add_argument('--weight_decay', type=float, default=0.001, metavar='wd',
                    help='weight_deacy')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='Adam BETA paramters.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()  
torch.manual_seed(args.seed)  
if args.cuda:
    torch.cuda.manual_seed(args.seed)  
else:
    args.gpu = None
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}  
def custom_rotation(img):
    angle = random.randint(0, 3) * 90
    return transforms.functional.rotate(img, angle)

train_path2 = r"D:\haiyang\datasets\BOWS_512_512_3000" 
train_data = datasets.ImageFolder(train_path2, transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(custom_rotation),
    transforms.ToTensor(), 
]))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

valid_path2 = r"D:\haiyang\datasets\ceshitu"  
valid_data = datasets.ImageFolder(valid_path2, transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((512,512)),
    transforms.ToTensor()
]))
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, **kwargs)
model = Cnnp_dnet.Cnnp()
if args.cuda:
    model.cuda()
optimer = optim.AdamW(model.parameters(), lr=args.lr, betas=args.betas, eps=1e-08,
                     weight_decay=args.weight_decay, amsgrad=False)  


def pre_img(img,mask, flag):
    device = img.device
    img1 = img.clone().to(device)
    I1 = torch.zeros(img.shape, device=device)
    I2 = torch.zeros(img.shape, device=device)
    I1[:, :, :, :] = img1[:, :, :, :]*(mask == flag)
    I2[:, :, :, :] = img1[:, :, :, :]*(mask != flag)
    return I1, I2


def result_deal(img,mask, flag):
    device = img.device
    img1 = img.clone().to(device)
    I1 = torch.zeros(img.shape, device=device)
    I1[:, :, :, :] = img1[:, :, :, :] * (mask == flag)
    return I1


def train(epoch):
    lr_train = (optimer.state_dict()['param_groups'][0]['lr'])
    print('lr_train=', lr_train)
    model.train()  
    for idx, (data, lable) in enumerate(train_loader):
        flag = random.randint(1, 4)
        x_, y_ = np.random.randint(0, 384, size=2)
        data = ((data*255)[:, :, x_:x_ + 128, y_:y_ + 128]).cuda()
        mask = category[x_:x_ + 128, y_:y_ + 128].cuda()
        i1, img = pre_img(data,mask, flag)
        img.requires_grad = True 
        data2 = model(img)
        pre_ = result_deal(data2,mask, flag) 
        loss = F.mse_loss(pre_, i1)
        optimer.zero_grad() 
        loss.backward()
        optimer.step()

        if (idx + 1) % 50 == 0:
            print('Train Epoch: {}   [{}/{} ({:.0f}%)], loss={}'.format(
                epoch, (idx + 1) * len(data), len(train_loader.dataset),
                       100. * (idx + 1) / len(train_loader), loss))
    return loss.item()

def valid():
    model.eval()
    loss0 = 0
    for idx, (data, label) in enumerate(valid_loader):
        data = (data*255).cuda()
        flag = random.randint(1, 4)
        with torch.no_grad():
            i1, img = pre_img(data,category, flag)
            data2 = model(img)
            pre_ = result_deal(data2,category, flag) 
            loss = F.mse_loss(pre_, i1)
        loss0 += loss.item()
    print("avg-valid-loss=", loss0/(idx+1))
    return loss0/(idx+1)

category = torch.zeros((512,512), dtype=torch.long, device='cuda')
category[::2, ::2] = 1
category[::2, 1::2] = 2
category[1::2, ::2] = 3
category[1::2, 1::2] = 4
import datetime
import pandas as pd
if __name__ == '__main__':
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    t3 = time.time()
    lo = []
    for epoch in range(args.epochs):
        if epoch%400==0 :optimer.param_groups[0]['lr'] = optimer.param_groups[0]['lr']*0.5
        t1 = time.time()
        loss1 = train(epoch)
        t2=time.time()
        loss2 = valid()
        lo.append({'epoch':epoch,'train-loss':loss1,'test-loss':loss2})
        df = pd.DataFrame(lo)
        df.to_excel('train-loss.xlsx', index=False)
        if epoch>400:
            state = {'network': model.state_dict(), 'optimizer_learn_rate': optimer.param_groups[0]['lr']}
            torch.save(state, './train_param/' + str(epoch) + '_model.pth')
        t2 = time.time()
        print('total time = ', (t2-t1)//60, 'm', (t2-t1) % 60, 's')
    t4 = time.time()
    print('total time = ', (t4-t3)//3600, 'h', ((t4-t3) % 3600)//60, 'm', (t4-t3) % 60, 's')
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print("time：", formatted_datetime)


