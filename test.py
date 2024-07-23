import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class TransformerModel(torch.nn.Module):
    def __init__(self, input_dim, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=input_dim, nhead=nhead, num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward, 
                                          dropout=dropout)
        self.linear = nn.Linear(input_dim, 1)
    