import torch
import os


def cuda(x=None):
    apply_cuda = os.environ.get('CUDA', 'false').lower() == 'true'
    if x is None:
        return apply_cuda
    
    return x.cuda() if apply_cuda else x


def randomize_input(x, randomness=0.01):
    device = 'cuda' if cuda() else 'cpu'
    return x + (torch.rand(x.shape, device=device) - 0.5) * randomness * x


def torch_rolling(t, window_size, training_mode=True):
    if window_size == 1:
        return t
    
    if window_size > t.shape[0]:
        return cuda(torch.zeros(t.shape[0], t.shape[1])) * (0 if training_mode else torch.nan)

    rolling_mean = t.T.unfold(1, window_size, 1).mean(axis=2).T
    padding = cuda(torch.zeros(window_size - 1, rolling_mean.shape[1]))
    if not training_mode:
        padding = padding * torch.nan
    return torch.cat([padding, rolling_mean])


class RollingMean(torch.nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def extra_repr(self):
        return f'length={self.length}'

    def forward(self, x):
        return torch_rolling(x, self.length, self.training)


def get_linear_block(input_size, hidden_size, bias, activation=None, dropout_p=0.5, batch_norm=False, network_sma_length=None):
    activation = activation if activation is not None else torch.nn.Identity()
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout_p), 
        torch.nn.Linear(input_size, hidden_size, bias=bias), 
        torch.nn.BatchNorm1d(hidden_size) if batch_norm else torch.nn.Identity(),
        activation,
        RollingMean(network_sma_length) if network_sma_length is not None else torch.nn.Identity(),
    )


def create_lstm_dataset(x, y=None, seq_length=128):
    X_sequences = []
    y_targets = []
    if len(x) - seq_length < 0:
        return (None, None) if y is not None else None

    for i in range(len(x) - seq_length):
        # Create a sequence of inputs of length `seq_length`
        X_sequences.append(x[i:i + seq_length])
        
        # The corresponding target is the next value in `y` after the sequence
        if y is not None:
            y_targets.append(y[i + seq_length])
    
    # Convert lists to tensors
    X_sequences = torch.stack(X_sequences)  # Shape: (num_sequences, seq_length, num_features)
    if y is not None:
        y_targets = torch.stack(y_targets)      # Shape: (num_sequences, output_features)
    
    return (X_sequences, y_targets) if y is not None else X_sequences


def mean_absolute_grads(net):
    grads = {}
    for name, param in net.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.abs().mean().item()
    return grads
