import torch.optim as optim
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import sys
import os

# Usar esse bloco somente se tiver problemas de importação como eu tive !!!!
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
project_root = os.path.dirname(grandparent_dir)
sys.path.append(grandparent_dir) # Adiciona src
sys.path.append(project_root)    # Adiciona a raiz do projeto

from dataset import main as load_data
from modelo import LSTMModel
from configs.log_config import logger

PARAMS = {
    'experimento': 'Deep_Finances_LSTM',
    'input_size': 5,            # Número de features de entrada
    'hidden_size': 100,         # Número de neurônios na camada oculta
    'num_epochs': 500,          # Número de épocas para treinamento
    'learning_rate': 0.002,     # Taxa de aprendizado
    'window_size': 5,           # Tamanho da janela deslizante
    'feature_column': 'Close'   # Coluna alvo para previsão
}

logger.info("Carregando dados para treinamento...")

X_train, y_train = load_data(path='./docs/planilhas_completas/data_actions_energy_1year.csv', collumn=PARAMS['feature_column'])
y_train = y_train.view(-1, 1)  # Ajusta o shape de y_train para (batch_size, 1)

logger.info(f"Dados carregados: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

mlflow.set_experiment(PARAMS['experimento'])

logger.info("Inicializando treinamento do modelo com mlflow...")
with mlflow.start_run():
    mlflow.log_params(PARAMS)
    
    modelo = LSTMModel(input_size=PARAMS['input_size'], hidden_size=PARAMS['hidden_size'])
    
    # Move modelo e dados para GPU se disponível
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=PARAMS['learning_rate'])
    
    logger.info("Iniciando loop de treinamento...")
    for epoch in range(PARAMS['num_epochs']):
        modelo.train()
        
        outputs = modelo(X_train)           # Treinamento do modelo
        loss = criterion(outputs, y_train)  # Cálculo da loss e validação do modelo
        optimizer.zero_grad()               # Zerando os gradientes          
        loss.backward()                     # Backpropagation  
        optimizer.step()                    # Atualização dos pesos
        
        mlflow.log_metric("loss", loss.item(), step=epoch)
        
        if (epoch+1) % 10 == 0:
            logger.info(f'Epoca [{epoch+1}/{PARAMS["num_epochs"]}], Loss: {loss.item():.5f}')
            
    logger.info("Treinamento concluído. Salvando o modelo...")
    input_sample = X_train[0:1].numpy()
    mlflow.pytorch.log_model(modelo, "lstm_model", input_example=input_sample)
    
    logger.info("Modelo salvo com sucesso no MLflow.")
    
    logger.info("Salvando scalers usados no pré-processamento...")
    mlflow.log_artifact("./src/api_deep_finances/dl/scalers/scaler_target.joblib")
    mlflow.log_artifact("./src/api_deep_finances/dl/scalers/scaler_all.joblib")