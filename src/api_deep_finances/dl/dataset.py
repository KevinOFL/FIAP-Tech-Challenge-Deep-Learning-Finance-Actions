import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

def treat_outliers_capping(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Treats outliers in the specified column of the DataFrame using the capping method based on the IQR.
    Parameters:
    - df: Input DataFrame.
    - column: The column name where outliers need to be treated.
    Returns:
    - DataFrame with outliers treated in the specified column.
    """
    
    df_treated = df.copy()
    # Caso não tenha a coluna Ticker, cria uma fictícia para evitar erros
    if 'Ticker' not in df_treated.columns:
        df_treated['Ticker'] = 'UNKNOWN'
        
    # Aplicando o capping para cada ticker individualmente
    for ticker in df_treated['Ticker'].unique():
        mask = df_treated['Ticker'] == ticker
        data = df_treated.loc[mask, column]
        
        # Se tiver poucos dados, o quantile pode falhar ou ser impreciso
        if len(data) > 0:
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_treated.loc[mask, column] = data.clip(lower=lower_bound, upper=upper_bound)
        
    # Remove a coluna fictícia se foi criada
    if 'UNKNOWN' in df_treated['Ticker'].values:
         if 'Ticker' in df_treated.columns and (df_treated['Ticker'] == 'UNKNOWN').all():
             df_treated.drop(columns=['Ticker'], inplace=True)

    return df_treated

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to the DataFrame for feature engineering.
    Currently adds Simple Moving Averages (SMA) and daily percentage variation.
    Parameters:
    - df: Input DataFrame with at least a 'Close' column.
    Returns:
    - DataFrame with added technical indicators.
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
    Sets up the 'Date' column as datetime and sorts the DataFrame by date.
    Parameters:
    - df: Input DataFrame with a 'Date' column.
    Returns:
    - DataFrame sorted by 'Date' with 'Date' column removed.
    """
    
    df_date = df.copy()
    
    df_date['Date'] = pd.to_datetime(df_date['Date'], utc=True)
    
    df_date = df_date.sort_values(by=['Date'])
    
    df_final = df_date.drop(columns=['Date'])
    
    return df_final

def normalize_data(data: pd.DataFrame, ticker:str) -> pd.DataFrame:
    """
    Normalizes the DataFrame using Min-Max scaling.
    Saves the scalers for target and all features separately.
    Parameters:
    - data: Input DataFrame to be normalized.
    - ticker: Ticker symbol used for naming the scaler files.
    Returns:
    - Normalized DataFrame.
    """
    
    data_normalized = data.copy()
    
    # Um Scaler para o target e outro para todas as features
    # Assim, na hora de inverter a normalização, conseguimos recuperar o valor original do target
    # Sem a nescessidade de matriz de transformação completa
    scaler_target = MinMaxScaler(feature_range=(0, 1)) 
    scaler_target.fit(data_normalized[['Target']])
    
    scaler_all = MinMaxScaler(feature_range=(0, 1))
    data_normalized[:] = scaler_all.fit_transform(data_normalized)
    
    ticker = ticker.replace('.SA','')
    
    # Salvando os scalers para uso futuro
    dump(scaler_target, f'./src/api_deep_finances/dl/scalers/scaler_target_{ticker}.joblib')
    dump(scaler_all, f'./src/api_deep_finances/dl/scalers/scaler_all_{ticker}.joblib')
    
    return data_normalized

def sliding_window(data, window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates sliding windows from the data for time series forecasting.
    Parameters:
    - data: Numpy array of shape (num_samples, num_features).
    - window_size: Size of the sliding window.
    Returns:
    - Tuple of tensors (X, y) where X is the input windows and y is the target values.
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

def prediction_data_tratative(df: pd.DataFrame, window_size: int, ticker: str) -> torch.Tensor:
    """
    Prepares the latest data for making predictions.
    Parameters:
    - df: Input DataFrame with the latest data.
    - window_size: Size of the sliding window.
    - ticker: Ticker symbol used for loading the scaler files.
    Returns:
    - Tensor ready for prediction.
    """
    
    path_root = './src/api_deep_finances/dl/scalers'
    scaler_all = load(f'{path_root}/scaler_all_{ticker}.joblib')
    
    df_tratated = df.copy()
    
    # Caso não tenha a coluna Ticker, cria uma fictícia para evitar erros
    if 'Ticker' not in df_tratated.columns:
        df_tratated['Ticker'] = ticker
        
    # Acionando as funções de tratamento
    df_tratated = date_setup(df_tratated)
    df_tratated = treat_outliers_capping(df_tratated, 'Close')
    df_tratated = feature_engineering(df_tratated)
    df_tratated = df_tratated.drop(columns=['Ticker','Dividends', 'Stock Splits', 'Volume', 'Open', 'High', 'Low'], errors='ignore')
    df_tratated.reset_index(drop=True, inplace=True)
    df_tratated.rename(columns={'Close': 'Target'}, inplace=True)
    df_tratated = df_tratated[['Target', 'SMA_5', 'SMA_15', 'SMA_30', 'Variation_pct']]
    
    # Verificando se temos dados suficientes
    if len(df_tratated) < window_size:
        raise ValueError(f"Dados insuficientes. O modelo precisa de {window_size} dias LIMPOS, mas só temos {len(df_tratated)}.")
    
    # Pegando apenas a última janela de dados
    df_final = df_tratated.tail(window_size).copy()
    
    # Normalizando os dados com o scaler salvo
    df_final[:] = scaler_all.transform(df_final)
    
    # Convertendo para tensor
    data_array = df_final.values
    tensor = torch.tensor(np.array(data_array).astype(np.float32)).unsqueeze(0)

    return tensor

def main(path:str, collumn:str, ticker:str, window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Main function to process the data and prepare training and testing datasets.
    Parameters:
    - path: Path to the CSV file.
    - collumn: Column name to be used for prediction.
    - ticker: Ticker symbol for filtering data.
    - window_size: Size of the sliding window.
    Returns:
    - Tuple of tensors (X_train, y_train, X_test, y_test).
    """
    
    # Acionando as funções de tratamento
    df = pd.read_csv(path)
    
    # Filtrando pelo ticker desejado
    df = df[df['Ticker'] == ticker].copy()
    if len(df) == 0:
        raise ValueError(f"No data found for ticker: {ticker}")
    
    df = date_setup(df)
    df = treat_outliers_capping(df, collumn)
    df = feature_engineering(df)
    
    # Removendo colunas desnecessárias
    df = df.drop(columns=['Ticker','Dividends', 'Stock Splits', 'Volume', 'Open', 'High', 'Low'], errors='ignore')
    
    # Ajustando o DataFrame para o formato esperado
    df.reset_index(drop=True, inplace=True) 
    df.rename(columns={'Close': 'Target'}, inplace=True)
    df = df[['Target', 'SMA_5', 'SMA_15', 'SMA_30', 'Variation_pct']]
    
    # Normalizando os dados e criando as janelas deslizantes
    df = normalize_data(df, ticker)
    data_array = df.values
    
    # Criando as janelas deslizantes
    X_tensor, y_tensor = sliding_window(data_array, window_size=window_size)
    
    # Dividindo em treino e teste (80% treino, 20% teste)
    train_size = int(len(X_tensor) * 0.8)
    X_train = X_tensor[:train_size]
    y_train = y_tensor[:train_size]
    X_test = X_tensor[train_size:]
    y_test = y_tensor[train_size:]
    
    return X_train, y_train, X_test, y_test