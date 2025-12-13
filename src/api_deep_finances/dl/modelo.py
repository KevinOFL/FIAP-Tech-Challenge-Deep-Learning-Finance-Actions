import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        
        # Definindo a arquitetura LSTM
        # input_size: número de features de entrada
        # hidden_size: quantidade de neurônios na camada oculta
        # batch_first=True para que a entrada seja (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        
        # Camada de dropout para evitar overfitting, desligando 20% dos neurônios durante o treinamento
        #self.dropout = nn.Dropout(p=0.2)
        
        # Camada linear intermediária
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=32)
        
        # Adiciona não linearidade
        self.relu = nn.ReLU()
        
        # Camada linear final para saída
        self.linear_final = nn.Linear(in_features=32, out_features=1)
        
    def forward(self, x):
        out, _ = self.lstm(x)                   # Saída da LSTM e estado oculto
        last_step = out[:, -1, :]               # Pegando a saída do último passo de tempo
        #x = self.dropout(last_step)            # Aplicando dropout
        x = self.linear1(last_step)             # Camada linear intermediária
        x = self.relu(x)                        # Função de ativação ReLU
        previsao = self.linear_final(x)         # Camada linear final
        return previsao                         # previsão final
        