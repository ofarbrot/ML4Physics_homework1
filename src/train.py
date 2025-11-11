import torch
import numpy as np
import sklearn
import torch.nn as nn
from model import model

def TrainingAlgoritm():
    input_data, trues = torch.load('input_data.pt'), torch.load('trues')
    training_model = model()
    return None #Skal vel ikke returnere noe

def LossFunc(y_pred:torch, y:torch)->float:
    '''can use 
    import torch
    import torch.nn as nn

    criterion = nn.BCEWithLogitsLoss()
    
    During training: loss = criterion(y_pred, y)'''
    return None 

def get_data(filename: str)->torch: #Will likely not use like this, but just a start
    data = torch.load(filename)
    return data