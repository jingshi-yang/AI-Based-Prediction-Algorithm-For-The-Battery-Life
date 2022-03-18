import torch


class LSTMNet(torch.nn.Module):

    def __init__(self, input_size, hidden_dim, num_layers, is_bidirectional, n_class=1):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_dim,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  bidirectional=is_bidirectional
                                  )
        self.linear = torch.nn.Linear(hidden_dim, n_class)


    def forward(self, f_input):
        # Input: (batch_size, seq_len, input_size)
        out, _ = self.cell(f_input)
        out = out.reshape(-1, self.hidden_dim)
        out = self.linear(out)
        # Output: (batch_size, output_size)
        return out


class GRUNet(torch.nn.Module):

    def __init__(self, input_size, hidden_dim, num_layers, n_class=1):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = torch.nn.GRU(input_size=input_size,
                                 hidden_size=hidden_dim,
                                 num_layers=num_layers,
                                 batch_first=True
                                 )
        self.linear = torch.nn.Linear(hidden_dim, n_class)


    def forward(self, f_input):
        # Input: (batch_size, seq_len, input_size)
        out, _ = self.cell(f_input)
        out = out.reshape(-1, self.hidden_dim)
        out = self.linear(out)
        # Output: (batch_size, output_size)
        return out