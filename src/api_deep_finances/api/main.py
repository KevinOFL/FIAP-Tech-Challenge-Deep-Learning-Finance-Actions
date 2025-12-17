import os
import sys
import mlflow
import pandas as pd
import yfinance as yf
import torch
import types
from dotenv import load_dotenv
from joblib import load
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Literal
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
project_root = os.path.dirname(grandparent_dir)
sys.path.append(grandparent_dir) 
sys.path.append(project_root)

from configs.log_config import logger_api as logger
from configs.log_config import logger_db
from src.api_deep_finances.dl.dataset import prediction_data_tratative
from src.api_deep_finances.dl.modelo import LSTMModel
from configs.database import SessionLocal, engine
from src.api_deep_finances.models import prediction_model

if 'modelo' not in sys.modules:
    fake_module = types.ModuleType('modelo') # Cria um módulo fantasma
    fake_module.LSTMModel = LSTMModel        # Coloca a classe dentro dele
    sys.modules['modelo'] = fake_module      # Registra no sistema
    logger.info("Patch aplicado: Módulo 'modelo' injetado no sys.modules")

mlruns_path = os.path.join(project_root, "mlruns")
mlflow.set_tracking_uri(f"file:///{mlruns_path}")

prediction_model.Base.metadata.create_all(bind=engine)

app = FastAPI(title="API Deep Finances", version="1.0.0")

def get_db():
    db = SessionLocal()
    logger_db.info("Sessão do banco de dados iniciada")
    try:
        yield db
        logger_db.info("Sessão do banco de dados finalizada com sucesso")
    finally:
        logger_db.info("Fechando sessão do banco de dados")
        db.close()
        
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return "<h1>API Deep Finances Online</h1>"

@app.post("/predict", response_class=JSONResponse)
async def predict(
    ticker: Literal[
    'NEOE3', 'AXIA6', 'EQTL3', 'RAIZ4', 'CPFE3', 'RECV3', 'AXIA3',     
    'ALUP11', 'VBBR3', 'BRAV3', 'AURE3', 'EGIE3', 'CMIG4', 'CSAN3',     
    'ENEV3', 'TAEE11', 'UGPA3', 'ENGI11', 'LIGT3', 'CPLE6', 'PRIO3',
    'PETR4'
    ],
    db: Session = Depends(get_db)):
    """
    Endpoint to predict the closing price of a given stock ticker.
    Saves the prediction result in the database.
    Parameters:
    - ticker: Stock ticker symbol.
    - db: Database session dependency.
    Returns:
    - JSONResponse with the predicted closing price.
    """
    input_db = {}
    ticker_SA = f"{ticker}.SA"
    logger.info(f"Recebemos o request para o ticker: {ticker}")
    
    params = pd.read_csv('./src/api_deep_finances/dl/docs/optimized_hyperparameters.csv')
    windows_size = int(params[params['ticker']==ticker_SA]['window_size'].values[0])
    path_root = './src/api_deep_finances/dl/scalers'
    scaler_target = load(f'{path_root}/scaler_target_{ticker}.joblib')
        
    try:
        model_name = f"LSTMModel_{ticker}"
        input_db['ticker'] = ticker
        input_db['model_name'] = model_name
        
        logger.info(f"Buscando versões para o modelo: {model_name}")
        
        client = mlflow.tracking.MlflowClient()
        
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"Modelo '{model_name}' não encontrado no registro do MLflow em {mlruns_path}")
        
        # Ordena as versões para pegar a mais recente
        versions.sort(key=lambda x: int(x.version))
        latest_version_obj = versions[-1]
        version_number = latest_version_obj.version
        current_stage = latest_version_obj.current_stage
        
        model_uri = f"models:/{model_name}/{version_number}"
        
        logger.info(f"Carregando versão {version_number} (Stage: {current_stage}) do URI: {model_uri}")
        input_db['model_version'] = str(version_number)
        model = mlflow.pytorch.load_model(model_uri=model_uri)
        logger.info(f"Modelo carregado com sucesso para o ticker: {ticker}")
        
        # Obter dados históricos do Yahoo Finance
        stock = yf.Ticker(ticker_SA)
        stock_wind_size = stock.history(period=os.getenv("YEAR_HISTORY"))
        df = pd.DataFrame(stock_wind_size)
        
        if df.empty:
            raise ValueError(f"Yahoo Finance retornou vazio para {ticker}")
        
        df.reset_index(inplace=True)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        datas = prediction_data_tratative(df, window_size=windows_size, ticker=ticker)
        
        model.eval()
        
        # Realizar a predição
        with torch.no_grad():
            prediction = model(datas)
            normalized_prediction = prediction.detach().cpu().numpy()
            real_value = scaler_target.inverse_transform(normalized_prediction)
            real_value = f"{real_value[0][0]:.2f}"
            input_db['prevision_price'] = float(real_value)
            logger.info(f"Previsão realizada com sucesso para o ticker {ticker}: {real_value}")
            
        # Inserir ou atualizar a previsão no banco de dados
        stmt = insert(prediction_model.Prediction).values(**input_db)
        logger_db.info(f"Inserindo/Atualizando previsão no banco de dados para o ticker {ticker}")
        db.execute(stmt)
        db.commit()
        db.close()

        return JSONResponse(
                content={
                "ticker": ticker,
                "predicted_close_price": float(real_value)
                },
                status_code=200)

    except Exception as e:
        logger.error(f"Erro ao processar a previsão para o ticker {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
        
        