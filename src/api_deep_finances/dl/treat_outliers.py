import numpy as np
import pandas as pd

def treat_outliers_capping(df, column):
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

# Como usar:
# df_limpo = treat_outliers_capping(df_original, 'Close')