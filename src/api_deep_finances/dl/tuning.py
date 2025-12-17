import optuna
import torch
import torch.optim as optim
import torch.nn as nn
import sys
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
project_root = os.path.dirname(grandparent_dir)
sys.path.append(grandparent_dir)
sys.path.append(project_root)

from dataset import main as load_data
from configs.log_config import logger_training as logger
from modelo import LSTMModel

TICKERS = [
    'NEOE3', 'AXIA6', 'EQTL3', 'RAIZ4', 'CPFE3', 'RECV3', 'AXIA3',     
    'ALUP11', 'VBBR3', 'BRAV3', 'AURE3', 'EGIE3', 'CMIG4', 'CSAN3',     
    'ENEV3', 'TAEE11', 'UGPA3', 'ENGI11', 'LIGT3', 'CPLE6', 'PRIO3',
    'PETR4'
]
N_TRIALS = int(os.getenv("N_TRIALS", 20))
EPOCHS_TUNING = int(os.getenv('EPOCHS_TUNING', 10))

def objective(trial:int, ticker:str) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    Parameters:
    - trial: An Optuna trial object.
    - ticker: Ticker symbol for which to load the data.
    Returns:
    - RMSE value on the test set after training with the given hyperparameters.
    """
    
    #Parametros a serem otimizados
    params = {
        'hidden_size': trial.suggest_int('hidden_size', 32, 128, step=16),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'window_size': trial.suggest_int('window_size', 10, 50, step=5),
        'batch_size': 32 # Fixo ou sugerido, como preferir
    }

    # Carregar dados (Com o window_size dinâmico)
    try:
        X_train, y_train, X_test, y_test = load_data(
            path='./docs/planilhas_completas/data_actions_energy_3year.csv', 
            collumn='Close', 
            ticker=ticker,
            window_size=params['window_size'] # <--- AQUI MUDA TUDO
        )
    except Exception as e:
        # Se a janela for muito grande para os dados disponíveis
        print(f"Erro ao carregar dados para {ticker}: {e}")
        raise optuna.TrialPruned()
    
    # Definindo o input_dim dinamicamente
    input_dim = X_train.shape[2] if len(X_train.shape) == 3 else 1
    # Ajustes de Tensor
    y_train = y_train.view(-1, 1)
    y_test = y_test.view(-1, 1)
    
    # Movendo para GPU se disponível
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Definição do modelo, critério e otimizador
    modelo = LSTMModel(
        input_size=input_dim, 
        hidden_size=params['hidden_size'], 
        num_layers=params['num_layers']
    ).to(device)
    
    # Critério e otimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=params['learning_rate'])
    
    # Treinamento do modelo
    for epoch in range(EPOCHS_TUNING):
        modelo.train()
        outputs = modelo(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Avaliação e Pruning no conjunto de teste
        modelo.eval()
        with torch.no_grad():
            test_out = modelo(X_test)
            test_loss = criterion(test_out, y_test)
            rmse = torch.sqrt(test_loss).item()
            
        # Reporta o resultado para o Optuna
        trial.report(rmse, epoch)

        # Se o resultado estiver muito ruim comparado aos outros trials, PARA AGORA.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return rmse

# Executando a otimização
# TICKERS{
# 0:NEOE3, 1:AXIA6, 2:EQTL3, 3:RAIZ4, 4:CPFE3, 5:RECV3, 6:AXIA3, 7:ALUP11, 8:VBBR3,
# 9:BRAV3, 10:AURE3, 11:EGIE3, 12:CMIG4, 13:CSAN3, 14:ENEV3, 15:TAEE11, 16:UGPA3,
# 17:ENGI11, 18:LIGT3, 19:CPLE6, 20:PRIO3, 21:PETR4}

ticker = f"{TICKERS[0]}.SA"
logger.info(f"--- Otimizando {ticker} ---")

# Criando o estudo
study = optuna.create_study(direction="minimize", study_name=f"Study for {ticker}")
# Executando a otimização
study.optimize(lambda trial: objective(trial, ticker), n_trials=N_TRIALS)

logger.info(f"Melhor RMSE para {ticker}: {study.best_value}")
for i in range(len(study.best_params)):
    param_name = list(study.best_params.keys())[i]
    param_value = study.best_params[param_name]
    logger.info(f"  {param_name}: {param_value}")