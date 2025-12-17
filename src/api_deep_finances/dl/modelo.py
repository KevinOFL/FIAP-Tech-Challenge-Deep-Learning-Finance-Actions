import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        LSTM Model for time series forecasting.
        Parameters:
        - input_size: Number of input features.
        - hidden_size: Number of neurons in the hidden layer.
        - num_layers: Number of LSTM layers.
        """
        
        super(LSTMModel, self).__init__()
        
        # Definindo a arquitetura LSTM
        # input_size: número de features de entrada
        # hidden_size: quantidade de neurônios na camada oculta
        # batch_first=True para que a entrada seja (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        
        # Camada linear intermediária
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=32)
        
        # Adiciona não linearidade
        self.relu = nn.ReLU()
        
        # Camada linear final para saída
        self.linear_final = nn.Linear(in_features=32, out_features=1)
        
    def forward(self, x):
        """
        Forward pass of the model.
        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, input_size).
        Returns:
        - previsao: Output tensor of shape (batch_size, 1).
        """
        
        out, _ = self.lstm(x)                   # Saída da LSTM e estado oculto
        last_step = out[:, -1, :]               # Pegando a saída do último passo de tempo
        #x = self.dropout(last_step)            # Aplicando dropout
        x = self.linear1(last_step)             # Camada linear intermediária
        x = self.relu(x)                        # Função de ativação ReLU
        previsao = self.linear_final(x)         # Camada linear final
        return previsao                         # previsão final
        