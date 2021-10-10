import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import *

from thop import profile
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import math
from torchstat import stat
import sklearn.metrics as sm
torch.cuda.set_device(0)
n_gpu = torch.cuda.device_count()
print(n_gpu)

train_x = np.load('experiments/pamap2_F/train_x.npy')
train_y = np.load('experiments/pamap2_F/train_y_p.npy')
test_x = np.load('experiments/pamap2_F/test_x.npy')
test_y = np.load('experiments/pamap2_F/test_y_p.npy')

print("\nShape of train_x:",train_x.shape,
      "\nShape of train_y:",train_y.shape,
      "\nShape of test_x:",test_x.shape,
      "\nShape of test_y:",test_y.shape,)

train_x = np.reshape(train_x, [-1, 86, 120, 1])
test_x = np.reshape(test_x, [-1, 86, 120, 1])
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)
print("\nShape of train_x:",train_x.shape,
      "\nShape of train_y:",train_y.shape,
      "\nShape of test_x:",test_x.shape,
      "\nShape of test_y:",test_y.shape,)

# use_gpu = torch.cuda.is_available()
batchSize = 512
# num_batches = math.ceil((np.size(train_x,0)/batchSize))
# print(num_batches)
torch_dataset = Data.TensorDataset(train_x,train_y)
train_loader = Data.DataLoader(dataset=torch_dataset,batch_size=batchSize,shuffle=True,num_workers=0)
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

class Net_SC(nn.Module):
    def __init__(self):
            super(Net_SC, self).__init__()
        # message = input("Use Selective_Convolution?[Y or N]")
        # if message in ['y', 'Y']:
        #     self.layer1 = nn.Sequential(
        #         SelectiveConv2d(in_channels=1,out_channels=64,kernel_size=(9,4),stride=(3,2),padding=0,
        #                         dropout_rate=0.1,gamma=0.001,K=32,N_max='None',),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(True)
        #     )
        #     self.ca1 = ChannelAttention(64)
        #     self.sa1 = SpatialAttention()
        #     self.layer2= nn.Sequential(
        #         SelectiveConv2d(in_channels=64,out_channels=128,kernel_size=(9,4),stride=(3,2),padding=0,
        #                         dropout_rate=0.1,gamma=0.001,K=32,N_max=256),
        #         nn.BatchNorm2d(128),
        #         nn.ReLU(True)
        #     )
        #     self.ca2 = ChannelAttention(128)
        #     self.sa2 = SpatialAttention()
        #     self.layer3 = nn.Sequential(
        #         SelectiveConv2d(in_channels=128,out_channels=256,kernel_size=(9,4),stride=(3,2),padding=0,
        #                         dropout_rate=0.1,gamma=0.001,K=32,N_max=512),
        #         nn.BatchNorm2d(256),
        #         nn.ReLU(True)
        #     )
        #     self.ca3 = ChannelAttention(256)
        #     self.sa3 = SpatialAttention()
        #     self.fc = nn.Sequential(
        #         nn.Linear(2304, 18)
        #     )
        # elif message in ['n', 'N']:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=86, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(0, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # nn.Dropout(p=0.1)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(0, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # nn.Dropout(p=0.1)
            )
            # self.ca1 = ChannelAttention(64)
            # self.sa1 = SpatialAttention()
            self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 1), stride=(3, 1), padding=(0, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True),
                # nn.Dropout(p=0.1)
            )
            self.fc = nn.Sequential(
                nn.Linear(1152, 12)
            )
        # else:
        #     exit(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x.cuda())
        return x

lr_list = []
LR = 0.0001
net = Net_SC().cuda()
opt = torch.optim.Adam(net.parameters(),lr=LR,weight_decay=1e-7)
loss_func = nn.CrossEntropyLoss().cuda()
params = list(net.parameters())
scheduler = lr_scheduler.ExponentialLR(opt, gamma=0.9)
k = 0
for i in params:
    l = 1
    print("该层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("该层参数和：" + str(l))
    k = k + l
print("总参数数量和：" + str(k))
epoch_list = []
accuracy_list = []
loss_list = []

def flat(data):
    data=np.argmax(data,axis=1)
    return  data
for epoch in range(3000):
    net.train()
    for step,(x,y) in enumerate(train_loader):
        x = x.type(torch.FloatTensor)
        x,y=x.cuda(),y
        output = net(x)
        y = flat(y).cuda()
        loss = loss_func(output,y.long())
        net.zero_grad()
        opt.zero_grad()
        loss.backward()
        opt.step()

    if epoch%1 ==0:
            net.eval()
            test_x = test_x.type(torch.FloatTensor)
            test_out = net(test_x.cuda())
            pred_y = torch.max(test_out,1)[1].data.squeeze().cuda()
            scheduler.step()
            lr_list.append(opt.state_dict()['param_groups'][0]['lr'])
            accuracy = (torch.sum(pred_y == flat(test_y.float()).cuda()).type(torch.FloatTensor) / test_y.size(0)).cuda()
            print('Epoch: ', epoch,  '| test accuracy: %.6f' % accuracy,'|loss:%.6f'%loss,'| params:',str(k))
    epoch_list.append(epoch)
    accuracy_list.append(accuracy.item())
    loss_list.append(loss.item())
    cm = sm.confusion_matrix(pred_y.cpu().numpy(), flat(test_y.float()).cpu().numpy())
    if accuracy > 0.9023:
        print(cm)
        np.save('Store/Pamap2/confusion_matrix_b.npy', cm)
print('Epoch_list:', epoch_list, 'Accuracy_list:', accuracy_list, 'Loss_list:', loss_list)
# np.save('Store/Pamap2_R/confusion_matrix_b.npy',cm)
# np.save('Store/Pamap2/epoch_baseline.npy',epoch_list)
# np.save('Store/Pamap2/accuracy_baseline.npy',accuracy_list)
# np.save('Store/Pamap2/loss_baseline.npy',loss_list)

x = epoch_list
y1 = accuracy_list
y2 = loss_list
plt.plot(x,y1,label = 'Accuracy')
plt.title('BaseLine')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

model = Net_SC()
stat(model, (86, 120, 1))

# Total params: 1,471,308
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Total memory: 1.41MB
# Total MAdd: 116.55MMAdd
# Total Flops: 58.46MFlops
# Total MemR+W: 8.45MB


