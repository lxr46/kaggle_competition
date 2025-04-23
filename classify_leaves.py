import os
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,ExponentialLR
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

#标签和类别数进行对应
train = pd.read_csv("D:/Pythontest/pytorch/data/classify-leaves/train.csv")
labels = list(pd.read_csv("D:/Pythontest/pytorch/data/classify-leaves/train.csv")['label'])
labels_unique = list(set(list(labels))) #list index--labels
label_nums = []
for i in range(len(labels)):
    label_nums.append(labels_unique.index(labels[i]))
train['number'] = label_nums
# train.to_csv("./train_num_label.csv", index = 0) #记录对应关系

test = pd.read_csv("D:/Pythontest/pytorch/data/classify-leaves/test.csv")

train_data,eval_data = train_test_split(train,test_size=0.2,stratify=train['number'])


class Leaf_Dataset(Dataset):
    '''
    树叶数据集的训练集 自定义Dataset
    '''

    def __init__(self, train_csv, transform=None, test=False):
        '''
        train_path : 传入记录图像路径及其标号的csv文件
        transform : 对图像进行的变换
        '''
        super().__init__()
        self.train_csv = train_csv
        self.image_path = list(self.train_csv['image'])  # 图像所在地址记录
        self.test = test
        if not self.test:
            self.label_nums = list(self.train_csv['number'])  # 图像的标号记录
        self.transform = transform

    def __getitem__(self, idx):
        '''
        idx : 所需要获取的图像的索引
        return : image， label
        '''
        image = cv2.imread(os.path.join("D:/Pythontest/pytorch/data/classify-leaves/train.csv", self.image_path[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #         image = Image.open(os.path.join("/kaggle/input/classify-leaves", self.image_path[idx]))
        if (self.transform != None):
            image = self.transform(image=image)['image']
        if not self.test:
            label = self.label_nums[idx]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.image_path)

transforms_train = albumentations.Compose(
    [
        albumentations.Resize(320, 320),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Rotate(limit=180, p=0.7),
        albumentations.RandomBrightnessContrast(),
        albumentations.ShiftScaleRotate(
            shift_limit=0.25, scale_limit=0.1, rotate_limit=0
        ),
        albumentations.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
            max_pixel_value=255.0, always_apply=True
        ),
        ToTensorV2(p=1.0),
    ]
)
transforms_test = albumentations.Compose(
        [
            albumentations.Resize(320, 320),
            albumentations.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                max_pixel_value=255.0, always_apply=True
            ),
            ToTensorV2(p=1.0)
        ]
    )

def train_model(train_loader, valid_loader, device = torch.device("cuda:0")):
    net = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, 176)
    net = net.to(device)
    epoch = 30
    best_epoch = 0
    best_score = 0.0
    best_model_state = None
    early_stopping_round = 3
    losses = []
    optimizer = optim.Adam(net.parameters(), lr=0.0001,weight_decay=1e-5)
    loss = nn.CrossEntropyLoss(reduction='mean')
#     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min = 1e-6)
    scheduler = ExponentialLR(optimizer, gamma=0.9,verbose=True)
    for i in range(epoch):
        acc = 0
        loss_sum = 0
        net.train()
        for x, y in tqdm(train_loader):
            x = torch.as_tensor(x, dtype=torch.float)
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            loss_temp = loss(y_hat, y)
            loss_sum += loss_temp
            optimizer.zero_grad()
            loss_temp.backward()
            optimizer.step()
#             scheduler.step()
            acc += torch.sum(y_hat.argmax(dim=1).type(y.dtype) == y)
        scheduler.step()
        losses.append(loss_sum.cpu().detach().numpy() / len(train_loader))
        print( "epoch: ", i, "loss=", loss_sum.item(), "训练集准确度=",(acc/(len(train_loader)*train_loader.batch_size)).item(), end="")

        test_acc = 0
        net.eval()
        for x, y in tqdm(valid_loader):
            x = x.to(device)
            x = torch.as_tensor(x, dtype=torch.float)
            y = y.to(device)
            y_hat = net(x)
            test_acc += torch.sum(y_hat.argmax(dim=1).type(y.dtype) == y)
        print("验证集准确度", (test_acc / (len(valid_loader)*valid_loader.batch_size)).item())
        if test_acc > best_score:
            best_model_state = copy.deepcopy(net.state_dict())
            best_score = test_acc
            best_epoch = i
            print('best epoch save!')
        if i - best_epoch >= early_stopping_round:
            break
    net.load_state_dict(best_model_state)
    testset = Leaf_Dataset(test, transform = transforms_test, test = True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, drop_last=False)
    device = torch.device("cuda:0")
    predictions = []
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.to(device)
            x = torch.as_tensor(x, dtype=torch.float)
            y_hat = net(x)
            predict = torch.argmax(y_hat,dim=1).reshape(-1)
            predict = list(predict.cpu().detach().numpy())
            predictions.extend(predict)
    return predictions

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=2023)
prediction_df = pd.DataFrame()
for fold_n, (trn_idx, val_idx) in enumerate(skf.split(train, train['number'])):
    print(f'fold {fold_n} training...')
    train_data = train.iloc[trn_idx]
    eval_data = train.iloc[val_idx]
    trainset = Leaf_Dataset(train_data, transform = transforms_train)
    evalset = Leaf_Dataset(eval_data, transform = transforms_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, drop_last=False)
    eval_loader = torch.utils.data.DataLoader(evalset, batch_size=32, shuffle=False, drop_last=False)
    predictions = train_model(train_loader, eval_loader)
    prediction_df[f'fold_{fold_n}'] = predictions