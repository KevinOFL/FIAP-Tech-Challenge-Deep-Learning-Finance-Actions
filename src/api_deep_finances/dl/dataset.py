import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

def treat_outliers_capping(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Replaces outliers with lower and upper limits calculated via IQR by Ticker.
    Ideal for Deep Learning to avoid breaking the temporal sequence.
    """
    df_treated = df.copy()
    
    for ticker in df_treated['Ticker'].unique():
        # Máscara para selecionar apenas as linhas deste ticker
        mask = df_treated['Ticker'] == ticker
        
        # Dados do ticker atual
        data = df_treated.loc[mask, column]
        
        # Cálculo dos quartis e limites
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Aplica o "Capping" (clip) nos valores desse ticker
        # Tudo abaixo do lower vira lower; tudo acima do upper vira upper
        df_treated.loc[mask, column] = data.clip(lower=lower_bound, upper=upper_bound)
        
    return df_treated

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators as new features to the DataFrame.
    """
    df_fe = df.copy()
    
    # Indicadores técnicos simples
    df_fe['SMA_5'] = df_fe['Close'].rolling(window=5).mean()  # Média móvel simples de 5 dias
    df_fe['SMA_15'] = df_fe['Close'].rolling(window=15).mean()  # Média móvel simples de 15 dias
    df_fe['SMA_30'] = df_fe['Close'].rolling(window=30).mean()  # Média móvel simples de 30 dias
    
    df_fe['Variation_pct'] = df_fe['Close'].pct_change()  # Variação percentual diária
    
    # Remover linhas com NaN geradas pelos cálculos
    df_fe.dropna(inplace=True)
    
    return df_fe

def date_setup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts 'Date' column to datetime, sorts by date, and drops the 'Date' column.
    """
    df_date = df.copy()
    
    df_date['Date'] = pd.to_datetime(df_date['Date'], utc=True)
    
    df_date = df_date.sort_values(by=['Date'])
    
    df_final = df_date.drop(columns=['Date'])
    
    return df_final

def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the DataFrame using Min-Max scaling.
    """
    data_normalized = data.copy()
    
    # Um Scaler para o target e outro para todas as features
    # Assim, na hora de inverter a normalização, conseguimos recuperar o valor original do target
    # Sem a nescessidade de matriz de transformação completa
    scaler_target = MinMaxScaler(feature_range=(0, 1)) 
    scaler_target.fit(data_normalized[['Target']])
    
    scaler_all = MinMaxScaler(feature_range=(0, 1))
    data_normalized[:] = scaler_all.fit_transform(data_normalized)
    
    dump(scaler_target, './src/api_deep_finances/dl/scalers/scaler_target.joblib')
    dump(scaler_all, './src/api_deep_finances/dl/scalers/scaler_all.joblib')
    
    return data_normalized

def sliding_window(data, window_size: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates sliding windows of data for time series forecasting.
    Returns tensors for input features (X) and target values (y).
    """
    X = []
    y = []
    
    # Loop que percorre os dados criando fotos do passado
    for i in range(len(data) - window_size):
        window = data[i:(i + window_size)]
        
        target = data[i + window_size]

        target_value = target[0]
        
        X.append(window)
        y.append(target_value)
        
    return torch.tensor(np.array(X).astype(np.float32)), torch.tensor(np.array(y).astype(np.float32))
    
def main(path:str, collumn:str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Main function to process the financial data CSV.
    Reads the CSV, applies date setup, outlier treatment, and feature engineering.
    """
    # Acionando as funções de tratamento
    df = pd.read_csv(path)
    df = date_setup(df)
    df = treat_outliers_capping(df, collumn)
    df = feature_engineering(df)
    
    # Removendo colunas desnecessárias
    df = df.drop(columns=['Ticker','Dividends', 'Stock Splits', 'Volume', 'Open', 'High', 'Low'])
    
    # Ajustando o DataFrame para o formato esperado
    df.reset_index(drop=True, inplace=True) 
    df.rename(columns={'Close': 'Target'}, inplace=True)
    df = df[['Target', 'SMA_5', 'SMA_15', 'SMA_30', 'Variation_pct']]
    
    # Normalizando os dados e criando as janelas deslizantes
    df = normalize_data(df)
    data_array = df.values
    X_tensor, y_tensor = sliding_window(data_array)
    
    return X_tensor, y_tensor

# Como usar:
#    X_train, y_train = main(path='./docs/planilhas_completas/data_actions_energy_1year.csv', collumn='Close')