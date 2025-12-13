import torch.optim as optim
import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
import torch
import sys
import os
from mlflow.models.signature import infer_signature

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
    'experimento': 'Deep_Finances_LSTM_for_energy_actions',
    'input_size': 5,                            # Número de features de entrada
    'hidden_size': 64,                          # Número de neurônios na camada oculta
    'num_layers': 2,                            # Número de camadas LSTM
    'num_epochs': 1000,                         # Número de épocas para treinamento
    'learning_rate': 0.001,                     # Taxa de aprendizado
    'window_size': 20,                          # Tamanho da janela deslizante
    'feature_column': 'Close',                  # Coluna alvo para previsão
    'patience': 250,                            # Paciência para early stopping
    'min_delta': 0.00001                        # Mudança mínima para considerar melhoria
}



tickers = pd.read_csv('./docs/planilhas_completas/data_actions_energy_3year.csv')['Ticker'].unique().tolist()

for i, ticker in enumerate(tickers):
    logger.info(f"Treinando modelo para o ticker {ticker} ({i+1}/{len(tickers)})")
    
    logger.info(f"Carregando dados para treinamento...")
    X_train, y_train, X_test, y_test = load_data(path='./docs/planilhas_completas/data_actions_energy_3year.csv', collumn=PARAMS['feature_column'], ticker=ticker, window_size=PARAMS['window_size'])
    y_train = y_train.view(-1, 1)   # Ajusta o shape de y_train para (batch_size, 1)
    y_test = y_test.view(-1, 1)     # Ajusta o shape de y_test para (batch_size, 1)

    logger.info(f"Dados carregados: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    mlflow.set_experiment(PARAMS['experimento'])

    logger.info("Inicializando treinamento do modelo com mlflow...")
    with mlflow.start_run():
        # Metadados do experimento
        mlflow.set_tag("ticker", ticker)
        mlflow.set_tag("setor", "Energia")
        mlflow.set_tag("model_type", "LSTM")
        mlflow.set_tag("optimizer", "Adam")
        mlflow.set_tag("developer", "Kevin")
        mlflow.log_params(PARAMS)
    
        modelo = LSTMModel(input_size=PARAMS['input_size'], hidden_size=PARAMS['hidden_size'], num_layers=PARAMS['num_layers'])
    
        # Move modelo e dados para GPU se disponível
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        modelo.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)
    
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(modelo.parameters(), lr=PARAMS['learning_rate'])
        
        # Learning rate scheduler para reduzir a taxa de aprendizado se a loss não melhorar por 10 épocas
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_loss_mse = float('inf')
        best_loss_rmse = float('inf')
        patience_counter = 0

        logger.info("Iniciando loop de treinamento...")
        for epoch in range(PARAMS['num_epochs']):
            modelo.train()
            
            outputs = modelo(X_train)                           # Treinamento do modelo
            loss = criterion(outputs, y_train)                  # Cálculo da loss
            rmse = torch.sqrt(loss)                             # Cálculo do RMSE   
            mae = torch.mean(torch.abs(outputs - y_train))      # Cálculo do MAE
            optimizer.zero_grad()                               # Zerando os gradientes          
            loss.backward()                                     # Backpropagation  
            optimizer.step()                                    # Atualização dos pesos
            
            
            # Avaliação no conjunto de teste
            modelo.eval()
            with torch.no_grad():
                test_outputs = modelo(X_test)
                test_loss = criterion(test_outputs, y_test)
                current_loss_mse = test_loss.item() 
                test_rmse = torch.sqrt(test_loss)
                current_loss_rmse = test_rmse.item()
                test_mae = torch.mean(torch.abs(test_outputs - y_test))
                
            scheduler.step(test_loss)                           # Atualiza o scheduler com a loss de teste atual
            
            mlflow.log_metric("Train Loss MSE", loss.item(), step=epoch)
            mlflow.log_metric("Test Loss MSE", test_loss.item(), step=epoch)
            mlflow.log_metric("Train RMSE", rmse.item(), step=epoch)
            mlflow.log_metric("Test RMSE", test_rmse.item(), step=epoch)
            mlflow.log_metric("Train MAE", mae.item(), step=epoch)
            mlflow.log_metric("Test MAE", test_mae.item(), step=epoch)

            # Coloca o modelo de volta em modo de treinamento
            modelo.train()
            
            # Early Stopping check
            if current_loss_mse < (best_loss_mse - PARAMS['min_delta']):
                best_loss_mse = current_loss_mse
                best_loss_rmse = current_loss_rmse
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= PARAMS['patience']:
                logger.info("----- Activate Early stopping -----")
                logger.info(f'Early stopping at epoch {epoch+1} and loss not upgraded for {PARAMS["patience"]} epochs. The best loss RMSE was {best_loss_rmse:.5f}.')
                break
            
            if (epoch+1) % 100 == 0:
                logger.info(f'Epoca [{epoch+1}/{PARAMS["num_epochs"]}], Train Loss RMSE: {rmse.item():.5f} - Test Loss RMSE: {test_rmse.item():.5f}')
        
        
        logger.info("Treinamento concluido. Salvando o modelo...")
        
        # Inferindo a assinatura do modelo para registro no MLflow
        input_sample_numpy = X_train[0:1].cpu().numpy()
        output_sample_numpy = modelo(X_train[0:1]).detach().cpu().numpy()
        signature = infer_signature(input_sample_numpy, output_sample_numpy)
        
        input_sample = X_train[0:1].numpy()
        
        ticker = ticker.replace('.SA','')
        
        # Salvando o modelo treinado no MLflow
        mlflow.pytorch.log_model(
            modelo,
            artifact_path='model',
            input_example=input_sample,
            registered_model_name=f"LSTMModel_{ticker}",
            signature=signature
        )

        logger.info("Modelo salvo com sucesso no MLflow.")
        
        logger.info("Salvando scalers usados no pre-processamento...")
        mlflow.log_artifact(f"./src/api_deep_finances/dl/scalers/scaler_target_{ticker}.joblib")
        mlflow.log_artifact(f"./src/api_deep_finances/dl/scalers/scaler_all_{ticker}.joblib")
        
logger.info("Treinamento de todos os modelos concluído.")