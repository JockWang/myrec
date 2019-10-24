import numpy as np
import pickle as pkl
from torch.utils.data import DataLoader, Dataset
import random
import os
import torch

def normal(put):
    index = list(put.keys())
    mat = []
    for i in index:
        mat.append(put[i])
    mat = np.array(mat)
    for i in range(mat.shape[1]):
        min_ = mat[:, i].min()
        max_ = mat[:, i].max()
        mat[:, i] = (mat[:, i] - min_) / (max_ - min_)
    #     print(mat)
    #     print(mat.shape)
    mat[np.isnan(mat)] = 0
    #     print(mat)
    for i in range(mat.shape[0]):
        put[index[i]] = mat[i, :].tolist()
    return put


def reaData(dataset='zhihu', mode='train'):
    data = {}
    if dataset == 'zhihu':
        with open('data/members_' + mode + '.pkl', 'rb') as source:
            data['mb'] = pkl.load(source)
        with open('data/qs_feature.pkl', 'rb') as source:
            data['qs'] = pkl.load(source)
        with open('data/as_feature.pkl', 'rb') as source:
            data['as'] = pkl.load(source)
        with open('data/mb_features.pkl', 'rb') as source:
            data['mf'] = normal(pkl.load(source))
        print('Loaded dataset:', dataset, mode)
        return Dataset(data)


#         return data

class Dataset(Dataset):
    def __init__(self, dataset, dataname='zhihu'):
        self.data = dataset
        self.dname = dataname

    def __getitem__(self, index):
        if self.dname == 'zhihu':
            mbf = self.data['mf']
            meb = self.data['mb']
            qes = self.data['qs']
            ans = self.data['as']
            qs_list = meb[index]['qs'][:-1]
            as_list = meb[index]['as'][:-1]
            invate = meb[index]['qs'][-1]
            label = meb[index]['label']
            target = [0, 0]
            target[label] = 1
            member = meb[index]['id']
            qs_matrix = []
            for qs in qs_list[-9:]:
                qs_matrix.append(qes[qs])
            as_matrix = []
            for a in as_list[-9:]:
                as_matrix.append(ans[a])
            return (np.array(qs_matrix), np.array(as_matrix), np.array(qes[invate]), np.array(mbf[member])), np.array(
                target)

    #             return (np.array(qs_matrix), np.array(as_matrix), np.array(qes[invate])), label

    def __len__(self):
        return len(self.data['mb'])


def seed_torch(seed=2018):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True