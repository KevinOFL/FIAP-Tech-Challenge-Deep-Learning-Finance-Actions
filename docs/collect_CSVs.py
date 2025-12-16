import pandas as pd
import yfinance as yf
import os
import sys
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from configs.log_config import logger

class Collect_CSVs:
    def __init__(self):
        # Ajuste para garantir que o caminho funcione independente de onde o script é rodado
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(self.base_path, "acoes")
        
        # Garante que a pasta de output existe
        os.makedirs(self.file_path, exist_ok=True)
        
        self.aux_path = os.path.join(self.base_path, "planilhas_aux", "acoes_setor_de_energia.csv")
        self.final_path = os.path.join(self.base_path, "planilhas_completas")
        os.makedirs(self.final_path, exist_ok=True)
        
    def clean_csv_aux(self, csv="docs/planilhas_aux/acoes_setor_de_energia.csv"):
        """ 
        Cleans the auxiliary CSV file by removing unnecessary columns and formatting tickers.
        """
        
        df = pd.read_csv(csv)
        
        df.dropna(inplace=True)
        df.drop(axis=1, columns=['Setor', 'Subsegmento', 'Empresa'], inplace=True)
        df = df + ".SA"
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def collect_finances_csv(self):
        """ 
        Collects financial data for tickers listed in the auxiliary CSV and saves them as individual CSV files.
        """
        
        df_tickers = self.clean_csv_aux()
        
        df_tickers = df_tickers[1:]
        
        for ticker_string in df_tickers['Ticker']:
            try:
                logger.info(f"Processando Ticker: {ticker_string}")
                
                stock = yf.Ticker(ticker_string)
                stock_3year = stock.history(period=os.getenv("YEAR_HISTORY"))
                
                if stock_3year.empty:
                    logger.warning(f"Nenhum dado encontrado para {ticker_string}")
                    continue
                
                df_datas = pd.DataFrame(stock_3year)
                df_datas["Ticker"] = ticker_string
                
                safe_ticker_name = ticker_string.replace('/', '_') 
                save_location = os.path.join(self.file_path, f"{safe_ticker_name}_3year.csv")
                df_datas.to_csv(save_location)
                
            except Exception as e:
                # yfinance pode falhar por 'KeyError' ou problemas de conexão, não 'FileNotFoundError'.
                logger.error(f"Falha ao processar {ticker_string}: {e}")
                
        logger.info("Data collection complete.")
    
    def join_csvs(self):
        """ 
        Joins individual CSV files into a single comprehensive CSV file.
        """
        
        all_files = os.listdir(self.file_path)
        csv_files = [f for f in all_files if f.endswith('_3year.csv')]
        
        if not csv_files:
            logger.warning("Nenhum arquivo CSV encontrado para unir.")
            return
        
        df_list = []
        for file in csv_files:
            file_full_path = os.path.join(self.file_path, file)
            df = pd.read_csv(file_full_path)
            df_list.append(df)
        
        combined_df = pd.concat(df_list, ignore_index=True)
        save_final = os.path.join(self.final_path, "data_actions_energy_3year.csv")
        combined_df.to_csv(save_final, index=False)
        
        logger.info(f"CSV files joined successfully at {save_final}")
    
    def main(self):
        self.collect_finances_csv()
        self.join_csvs()
        
        
collector = Collect_CSVs()
collector.main()  