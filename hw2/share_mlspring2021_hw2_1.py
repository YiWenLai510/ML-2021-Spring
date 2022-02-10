import torch
from torch.utils.data import Dataset
# from sklearn.model_selection import KFold
# from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from statistics import mode
import csv
import glob
import os
import pandas as pd

print('Loading data ...')

data_root='./timit_11/timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')

train = np.concatenate((train,train[:,39*4:39*7]),axis=1)
test = np.concatenate((test,test[:,39*4:39*7]),axis=1)

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))
def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        
        self.layer1 = nn.Linear(546, 2048)

        self.layer2 = nn.Linear(2048, 2048)
        self.layer3 = nn.Linear(2048, 2048)
        self.layer4 = nn.Linear(2048, 2048)
        self.layer5 = nn.Linear(2048, 2048)
        self.layer6 = nn.Linear(2048, 2048)
        self.layer7 = nn.Linear(2048, 2048)

        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)
        self.dropout3 = nn.Dropout(0.6)
        self.dropout4 = nn.Dropout(0.6)
        self.dropout5 = nn.Dropout(0.6)
        self.dropout6 = nn.Dropout(0.6)
        self.dropout7 = nn.Dropout(0.6)

        self.batchNorm1 = nn.BatchNorm1d(2048)
        self.batchNorm2 = nn.BatchNorm1d(2048)
        self.batchNorm3 = nn.BatchNorm1d(2048)
        self.batchNorm4 = nn.BatchNorm1d(2048)
        self.batchNorm5 = nn.BatchNorm1d(2048)
        self.batchNorm6 = nn.BatchNorm1d(2048)
        self.batchNorm7 = nn.BatchNorm1d(2048)
        
        self.layer_out = nn.Linear(2048,39)

        
        self.act_fn1 = nn.ReLU()
        self.act_fn2 = nn.ReLU()
        self.act_fn3 = nn.ReLU()
        self.act_fn4 = nn.ReLU()
        self.act_fn5 = nn.ReLU()
        self.act_fn6 = nn.ReLU()
        self.act_fn7 = nn.ReLU()

        
    def forward(self, x):

        # x = self.layer1(x)
        # x = self.batchNorm1(x)
        # x = self.act_fn(x)
        # x = self.dropout1(x)

        x = self.layer1(x)
        x = self.act_fn1(x)
        x = self.batchNorm1(x)
        x = self.dropout1(x)  

        x = self.layer2(x)
        x = self.act_fn2(x)
        x = self.batchNorm2(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.act_fn3(x)
        x = self.batchNorm3(x)
        x = self.dropout3(x)

        x = self.layer4(x)
        x = self.act_fn4(x)
        x = self.batchNorm4(x)
        x = self.dropout4(x)

        x = self.layer5(x)
        x = self.act_fn5(x)
        x = self.batchNorm5(x)
        x = self.dropout5(x)

        x = self.layer6(x)
        x = self.act_fn6(x)
        x = self.batchNorm6(x)
        x = self.dropout6(x)

        x = self.layer7(x)
        x = self.act_fn7(x)
        x = self.batchNorm7(x)
        x = self.dropout7(x)

        x = self.layer_out(x)
        return x
"""Split the labeled data into a training set and a validation set, you can modify the variable `VAL_RATIO` to change the ratio of validation data."""

def training(train_x, train_y, val_x, val_y, ramdomseed):

    BATCH_SIZE = 2048
    s = 1253*ramdomseed + 4389
    print('seed ', s)
    same_seeds(s)
    device = get_device()
    # print(f'DEVICE: {device}')
    num_epoch = 300               # number of training epoch
    learning_rate = 0.0001       # learning rate   
    model_path = './model.ckpt'

    train_set = TIMITDataset(train_x, train_y)
    val_set = TIMITDataset(val_x, val_y)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,factor=0.1)

    best_acc = 0.0
    early_stop_cnt = 0
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        model.train() # set the model to training mode
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs) 
            batch_loss = criterion(outputs, labels)
            _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            batch_loss.backward() 
            optimizer.step() 

            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()

        # validation
        if len(val_set) > 0:
            model.eval() # set the model to evaluation mode
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels) 
                    _, val_pred = torch.max(outputs, 1) 
                
                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                    val_loss += batch_loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
                ))

                # if the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), model_path)
                    print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1
                
                if early_stop_cnt >= 15:
                    print('========break with early stop========', early_stop_cnt)
                    break
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
            ))



def testing(test, model_path, num):
    device = get_device()
    # create testing dataset
    test_set = TIMITDataset(test, None)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    # create model and load weights from checkpoint
    model = Classifier().to(device)
    model.load_state_dict(torch.load(model_path))
    predict = []
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

            for y in test_pred.cpu().numpy():
                predict.append(y)
    with open('7layer_b2048.csv','w') as f:
    # with open('./KFold_csv/prediction_'+str(num)+'.csv', 'w') as f:# ./KFold_csv/
        f.write('Id,Class\n')
        for i, y in enumerate(predict):
            f.write('{},{}\n'.format(i, y))
    print('predicting in ' + str(num)+ ' Folds')

#    
VAL_RATIO = 0.05
percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))
training(train_x, train_y, val_x, val_y, 0)
print('start test ')
testing(test, './model.ckpt',0)
### Main Funciton #####
# testing(test, './model.ckpt', 0)

# skf = KFold(n_splits=5, shuffle=False, random_state=None)
# iter_cnt = 0
# for train_index, test_index in skf.split(train, train_label):

#     train_x, train_y = train[train_index], train_label[train_index]
#     val_x, val_y = train[test_index], train_label[test_index]
#     # train_x, train_y, val_x, val_y = np.concatenate((train_x,train_x[:,39*4:39*7]),axis=1), train_y, np.concatenate((val_x,val_x[:,39*4:39*7]),axis=1), val_y

#     print('===== ' + str(iter_cnt) + ' Fold ========')
#     print('Size of training set: {}'.format(train_x.shape))
#     print('Size of validation set: {}'.format(val_x.shape))

#     training(train_x, train_y, val_x, val_y, iter_cnt)
#     testing(test, './model.ckpt', iter_cnt)
#     iter_cnt += 1

# folder_name = 'KFold_csv' 
# file_type = 'csv'
# seperator =','
# df = pd.concat([pd.read_csv(f, sep=seperator,header=0).iloc[:, 1] for f in glob.glob(folder_name + "/*."+file_type)],ignore_index=True, axis=1)
# print(df)
# results = []
# ids = []
# for index, row in df.iterrows():
#     results.append(mode(row.to_list()))
#     ids.append(index)
# print(len(results))
# resultdf = pd.DataFrame(list(zip(ids,results)),columns=['Id', 'Class'])
# print(resultdf)
# resultdf.to_csv('0fold_results.csv',index=False)#header=['Id','Class']
# print('finish predict')

