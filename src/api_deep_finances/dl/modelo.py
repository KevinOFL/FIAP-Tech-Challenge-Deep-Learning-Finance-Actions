import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        
        # Definindo a arquitetura LSTM
        # input_size: número de features de entrada
        # hidden_size: quantidade de neurônios na camada oculta
        # batch_first=True para que a entrada seja (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        
        # Camada linear para mapear a saída da LSTM para o valor de previsão
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)
        
    def forward(self, x):
        out, _ = self.lstm(x)           # Saída da LSTM e estado oculto
        last_step = out[:, -1, :]       # Pegando a saída do último passo de tempo
        return self.linear(last_step)   # Mapeando para a previsão final
        