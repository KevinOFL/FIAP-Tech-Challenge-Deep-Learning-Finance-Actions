import pandas as pd
import yfinance as yf
import os

class Collet_CSVs:
    def __init__(self):
        self.file_path = "docs/acoes"
        self.data = []
        
    def clean_csv_aux(self, csv="docs/planilhas_aux/acoes_setor_de_energia.csv"):
        # Load the CSV file
        df = pd.read_csv(csv, )
        
        # Example cleaning steps
        df.dropna(inplace=True)  # Remove missing values
        df.drop(axis=1, columns=['Setor', 'Subsegmento', 'Empresa'], inplace=True)
        df = df + ".SA"
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def collect_finances_csv(self):
        df_tickers = self.clean_csv_aux()
        
        df_tickers = df_tickers[1:]
        
        for ticker_string in df_tickers['Ticker']:
            try:
                print(f"Processando Ticker: {ticker_string}")
                
                stock = yf.Ticker(ticker_string)
                stock_6mon = stock.history(period="6mo")
                df_datas = pd.DataFrame(stock_6mon)
                df_datas["Ticker"] = ticker_string
                
                safe_ticker_name = ticker_string.replace('/', '_') 
                df_datas.to_csv(f"{self.file_path}/{safe_ticker_name}_6mo.csv")
                
            except Exception as e:
                # yfinance pode falhar por 'KeyError' ou problemas de conexão, não 'FileNotFoundError'.
                print(f"Falha ao processar {ticker_string}: {e}")
                
        return "Data collection complete."
    
    def join_csvs(self):
        all_files = os.listdir(self.file_path)
        csv_files = [f for f in all_files if f.endswith('_6mo.csv')]
        
        df_list = []
        for file in csv_files:
            df = pd.read_csv(f"{self.file_path}/{file}")
            df_list.append(df)
        
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df.to_csv("docs/planilhas_completas/data_actions_energy_6mo.csv", index=False)
        return "CSV files joined successfully."
    
    def main(self):
        self.collect_finances_csv()
        self.join_csvs()
        
        
if __name__ == "__main__":
    collector = Collet_CSVs()
    collector.main()  