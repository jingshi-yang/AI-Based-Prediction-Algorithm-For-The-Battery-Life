import torch


class neuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, is_bidirectional, output_size=1):
        super(neuralNetwork, self).__init__()
        self.cell = torch.nn.LSTM(input_size = input_size,
                                  hidden_size=hidden_dim,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  bidirectional=is_bidirectional
                                  )
        self.linear = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, input):
        # Input: (batch_size, seq_len, input_size)
        out, _ = self.cell(input)
        out = out.reshape(-1, self.hidden_dim)
        out = self.linear(out)
        # Output: (batch_size, output_size)
        return out