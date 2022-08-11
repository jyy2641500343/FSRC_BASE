import torch
import os
import numpy as np

lines = [x.strip() for x in open('train.csv', 'r').readlines()][1:]
data = []
label = []
lb = -1

wnids_list = []

for l in lines:
    name, wnid = l.split(',')
    path = name
    if wnid not in wnids_list:
        wnids_list.append(wnid)
        lb += 1
    data.append(path)
    label.append(lb)

label = np.array(label)
m_ind = []
for i in range(max(label) + 1):
    ind = np.argwhere(label == i).reshape(-1)
    ind = torch.from_numpy(ind)
    m_ind.append(ind)

x = []
for i in range(10):
    x.append(i)
print(m_ind[1])