
import argparse
#from GHA import audiogram

import numpy as np
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import speechbrain as sb
from torch.utils.data import DataLoader
import os
import datetime
import torchaudio.transforms as T
from scipy.optimize import curve_fit
import wandb
from attrdict import AttrDict
import json
import csv
from scipy.stats import spearmanr
from data_handling import get_disjoint_val_set
from process_cpc2_data import get_cpc2_dataset
from transformers import WhisperProcessor
import random
from models.simple_models import ffnn_init, minerva_wrapper4
from models.ni_feat_extractors import Spec_feats, XLSREncoder_feats, XLSRFull_feats, \
    HuBERTEncoder_feats, HuBERTFull_feats, WhisperEncoder_feats, WhisperFull_feats, WhisperBase_feats
from exemplar import get_ex_set


def main():
    

    with open('thesis_models/cpc-corrects-1234.txt', 'r') as f:
        corrects = f.read()
    corrects = [i for i in corrects.split('\n')]
    # print(corrects)
    corrects = corrects[1:]

    ex_corrects = torch.load('thesis_models/003_001_1_WhisperFull_minerva_21-21-14-Jun-2024_1234/minerva-1234_r.pt')
    writeOut = [' '.join([corrects[i], str(ex_corrects[i].item())]) for i in range(len(ex_corrects))]
    writeOut = "\n".join(writeOut)
    writeOut = "\n".join(['corrects model1', writeOut])
    with open('thesis_models/cpc-rpe-1234.txt', 'w') as f:
        f.write(writeOut)

    print(writeOut)

    print(ex_corrects.min(), ex_corrects.max())

if __name__ == '__main__':
    main()