import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import sys
import os
from dataset import main as load_data

# Tente usar esse bloco somente se tiver problemas de importação como eu tive !!!!
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
project_root = os.path.dirname(grandparent_dir)
sys.path.append(grandparent_dir) # Adiciona src
sys.path.append(project_root)    # Adiciona a raiz do projeto

from configs.log_config import logger

PARAMS = {
    'experimento': 'Deep_Finances_LSTM',
    'input_size': 4,          # Número de features de entrada
    'hidden_size': 50,        # Número de neurônios na camada oculta
    'num_epochs': 100,        # Número de épocas para treinamento
    'learning_rate': 0.001,   # Taxa de aprendizado
    'window_size': 5,         # Tamanho da janela deslizante
    'feature_column': 'Close' # Coluna alvo para previsão
}

logger.info("Carregando dados para treinamento...")
X_train, y_train = load_data(path='./docs/planilhas_completas/data_actions_energy_1year.csv', collumn=PARAMS['feature_column'])

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        # Definindo a arquitetura LSTM
        # input_size: número de features de entrada
        # hidden_size: quantidade de neurônios na camada oculta
        # batch_first=True para que a entrada seja (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        # Camada linear para mapear a saída da LSTM para o valor de previsão
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)
        
    def forward(self, x):
        out, _ = self.lstm(x) # Saída da LSTM e estado oculto
        last_step = out[:, -1, :] # Pegando a saída do último passo de tempo
        return self.linear(last_step) # Mapeando para a previsão final
        