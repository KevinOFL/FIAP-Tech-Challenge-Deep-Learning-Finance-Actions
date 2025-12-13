import optuna
import torch
import torch.optim as optim
import torch.nn as nn
import sys
import os
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
project_root = os.path.dirname(grandparent_dir)
sys.path.append(grandparent_dir)
sys.path.append(project_root)

from dataset import main as load_data
from configs.log_config import logger
from modelo import LSTMModel

#TICKER_ALVO = "ALUP11.SA" # Escolha um ticker representativo ou aquele que está com performance ruim
N_TRIALS = 20             # Quantas combinações diferentes o Optuna vai testar
TICKERS = pd.read_csv('./docs/planilhas_completas/data_actions_energy_3year.csv')['Ticker'].unique().tolist()

def objective(trial, ticker_index=0):
    """
    Função que o Optuna vai rodar repetidamente.
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
            ticker=TICKERS[ticker_index],
            window_size=params['window_size'] # <--- AQUI MUDA TUDO
        )
    except ValueError:
        # Se a janela for muito grande para os dados disponíveis
        raise optuna.TrialPruned()
    
    # Ajustes de Tensor
    y_train = y_train.view(-1, 1)
    y_test = y_test.view(-1, 1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Definição do modelo, critério e otimizador
    modelo = LSTMModel(
        input_size=5, 
        hidden_size=params['hidden_size'], 
        num_layers=params['num_layers']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=params['learning_rate'])

    # Loop de treinamento simples
    # Não precisamos de 1000 épocas para saber se a configuração é boa. 50 ou 100 bastam.
    EPOCHS_TUNING = 50
    
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

    # O Optuna quer MINIMIZAR esse valor retornado
    return rmse

if __name__ == "__main__":
    list_actions_params = []
    
    for i, ticker in enumerate(TICKERS):
        dict_params = {}
        dict_params['ticker'] = ticker
        logger.info("=" * 50)
        logger.info(f"Otimizando hiperparâmetros para o ticker {ticker} ({i+1}/{len(TICKERS)})")
        
        # Cria o estudo
        study = optuna.create_study(direction="minimize", study_name=f"LSTM_Tuning for {ticker}")
        # Roda a otimização
        study.optimize(objective, n_trials=N_TRIALS)

        for key, value in study.best_params.items():
            dict_params[key] = value
            
        logger.info(f"Melhores hiperparâmetros para {ticker}: {study.best_params}")
        
        list_actions_params.append(dict_params)
        
    df_params = pd.DataFrame(list_actions_params)
    df_params.to_csv('./src/api_deep_finances/dl/docs/optimized_hyperparameters.csv', index=False)