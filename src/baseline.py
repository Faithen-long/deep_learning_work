#!/usr/bin/env python
# coding: utf-8

# # 期末大作业——地震数据分类

# ## 导入库

# In[1]:


import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from dataset import SeismicDataset, mean_std_norm
from network import BasicBlock, Resnet


# ## 数据准备和检查

# In[ ]:


BATCH_SIZE = 8

# 用自定义的SeismicDataset类进行数据载入和预处理
train_dataset = SeismicDataset('../data/seismic_dataset.npz', is_train=True)
valid_dataset = SeismicDataset('../data/seismic_dataset.npz', is_train=False)

# 用torch的DataLoader类加载data loader
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_data_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 查看数据类型,维度和分布
type_list = train_dataset.type_list
print(f'type list: {type_list}')
train_data_array = train_dataset.waveform_array
print(f'shape of training data: {train_data_array.shape}')
train_label_list = train_dataset.label_array

count_list = [0] * len(type_list)
for label in train_label_list:
    count_list[label] += 1
print('training data number of')
for i in range(len(type_list)):
    print(f'{type_list[i]}:\t{count_list[i]}')

# 数据可视化
train_data_loader_iter = iter(train_data_loader)
inputs, labels = next(train_data_loader_iter)
inputs = np.array(inputs[:8])
types = type_list[labels[:8]]
for (input, type_) in zip(inputs, types):
    plt.figure(figsize=(20,4))
    plt.title(type_)
    for i in range(3):
        plt.plot(input[i])
    plt.show()


# ## 设置超参数, 定义网络, 优化器和损失函数, 并初始化训练日志

# In[ ]:


# 选择设备
IS_GPU = True
DEVICE = torch.device('cpu')
if torch.cuda.is_available(): 
    DEVICE = torch.device('cuda')


# 设置超参数
EPOCHS = 10
LR = 1e-3

model = Resnet(BasicBlock, [2, 2, 2, 2])
if IS_GPU:
    model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_loss_list = []
train_accuracy_list = []
valid_loss_list = []
valid_accuracy_list = []


# ## 训练网络

# In[ ]:


# EPOCHS = 10
# LR = 1e-3
# optimizer = optim.Adam(model.parameters(), lr=LR)

start_time = datetime.now()
print(f'start time: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')

train_batch_num = len(train_data_loader)
valid_batch_num = len(valid_data_loader)

# 做epochs轮训练
for epoch in range(EPOCHS):
    print(f'epoch: {epoch+1:2d}/{EPOCHS:2d}')

    # 用训练集训练
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0
    
    for (inputs, labels) in tqdm(train_data_loader):
        if IS_GPU:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        train_accuracy += (predicted == labels).sum().item() / len(inputs)
    
    train_loss /= train_batch_num
    train_loss_list.append(train_loss)
    train_accuracy /= train_batch_num
    train_accuracy_list.append(train_accuracy)
    print(f'train: loss={train_loss:4.2f}, accuracy={100*train_accuracy:4.1f}%')

    # 验证
    model.eval()
    valid_loss = 0.0
    valid_accuracy = 0.0

    for (inputs, labels) in tqdm(valid_data_loader):
        if IS_GPU:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        valid_accuracy += (predicted == labels).sum().item() / len(inputs)
    
    valid_loss /= valid_batch_num
    valid_loss_list.append(valid_loss)
    valid_accuracy /= valid_batch_num
    valid_accuracy_list.append(valid_accuracy)
    print(f'valid: loss={valid_loss:4.2f}, accuracy={100*valid_accuracy:4.1f}%')

end_time = datetime.now()
print(f'end time: {end_time.strftime("%Y-%m-%d %H:%M:%S")}')


# In[ ]:


epoch_list = range(1, len(train_loss_list)+1)

plt.title('Loss Curve')
plt.plot(epoch_list, train_loss_list)
plt.plot(epoch_list, valid_loss_list)
plt.legend(['train loss', 'valid loss'])
plt.show()

plt.title('Accuracy Curve')
plt.plot(epoch_list, train_accuracy_list)
plt.plot(epoch_list, valid_accuracy_list)
plt.legend(['train accuracy', 'valid accuracy'])
plt.show()


# ## 对测试集数据进行预测并保存为excel文件

# In[ ]:


test_data_array = np.load('../data/seismic_data_test.npy')
test_data_array = mean_std_norm(test_data_array)

test_pred_array = np.zeros(len(test_data_array), dtype='int64')
for i, test_data in enumerate(test_data_array):
    test_data = test_data.reshape(1, 3, 15000)
    test_data = torch.Tensor(test_data).to(DEVICE)
    test_output = model(test_data)
    test_pred = int(torch.argmax(test_output, dim=1)[0].cpu().numpy())
    test_pred_array[i] = test_pred

id_list = np.arange(len(test_pred_array))
ans = np.array([id_list, test_pred_array]).T
df = pd.DataFrame(ans, columns = ['id', 'pred'])
df.to_csv('../data/result.csv', index=False)

