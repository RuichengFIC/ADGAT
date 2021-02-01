from torch import nn
import math
import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import os



def reset_parameters(named_parameters):
    for i in named_parameters():
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        else:
            nn.init.xavier_normal_(i[1])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(trues, preds):
    trues = np.concatenate(trues,-1)
    preds = np.concatenate(preds,0)
    acc = sum(preds.argmax(-1) == trues) / len(trues)
    auc = roc_auc_score(trues,preds[:,1])
    return acc, auc

def createPath(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

