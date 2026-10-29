import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from ebt.simulate.tricks import identity
from ebt.simulate.nn.nn_loss import *
from ebt.simulate.nn.loader import *
from ebt.simulate.nn.util import *
from ebt.simulate import Model
from typing import List
import pandas as pd
import numpy as np
import inspect
import logging
import os

_LOGGER = logging.getLogger(__name__)


def _accepted_kwargs(fn):
    """(named keyword-able parameters of ``fn``, whether it also takes **kwargs)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return set(), True
    names = {p.name for p in sig.parameters.values()
             if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    return names, var_kw


def _kwargs_for(fn, hyperparameters):
    """The hyperparameters ``fn`` can take: all of them if it has **kwargs, else only the
    ones it names. Lets the factories have explicit signatures (so a typo is detectable)
    while NeuralNetModel keeps forwarding one flat hyperparameter dict."""
    names, var_kw = _accepted_kwargs(fn)
    return dict(hyperparameters) if var_kw else {k: v for k, v in hyperparameters.items() if k in names}


class FeedForwardNet(torch.nn.Module):
    def __init__(self, n_features: int, n_labels: int, n_hidden: List[int] = None, dropout_p: List[float] = None, active_f=None, bias=False, network_sma_length=1, batch_norm=False):
        super().__init__()
        n_hidden_layers = len(n_hidden) if n_hidden else 0
        dropout_p = (n_hidden_layers + 2) * [0.5] if dropout_p is None else dropout_p
        if type(dropout_p) == float:
            dropout_p = [dropout_p] * (n_hidden_layers+2)
        active_f = active_f if active_f else [torch.nn.Identity() for _ in range(n_hidden_layers + 1)]

        if n_hidden_layers == 0:
            n_first_outs = n_labels
            n_last_ins = n_features
        else:
            n_first_outs = n_hidden[0]
            n_last_ins = n_hidden[-1]

        input_layers = torch.nn.Sequential(*get_linear_block(n_features, n_first_outs, bias, active_f[0], dropout_p[0], batch_norm=batch_norm))
        modules = torch.nn.ModuleList([
            get_linear_block(n_hidden[i], n_hidden[i+1], bias, active_f[i], dropout_p[i + 1], batch_norm=batch_norm)
            for i in range(n_hidden_layers - 1)
        ])
        output_layers = get_linear_block(n_last_ins, n_labels, bias, active_f[-1], dropout_p[-1], batch_norm)
        modules.insert(0, input_layers)
        
        if n_hidden_layers > 0:
            modules.append(output_layers)
        
        if network_sma_length > 1:
            modules.append(RollingMean(network_sma_length))

        self.layers = modules        

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x
    
    
class FeedForwardNetAbs(FeedForwardNet):
    def forward(self, x):
        return super().forward(x).abs()


def _default_create_optim(params, lr=0.01):
    return torch.optim.AdamW(params, lr=lr)


def _default_create_scheduler(optim, factor=0.9, patience=1000):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=factor, patience=patience)


#: hyperparameters NeuralNetModel consumes itself (everything else goes to the factories)
_MODEL_HYPERPARAMETERS = frozenset({'n_epochs', 'save_losses', 'save_state_dicts', 'batch_size',
                                    'progress', 'progress_batch', 'torch_seed'})


class NeuralNetModel(Model):
    '''
    Walk-forward neural-net model. ``hyperparameters`` is one flat dict: the keys in
    _MODEL_HYPERPARAMETERS are read here, the rest are forwarded to ``create_net``,
    ``create_optim``, ``create_scheduler`` and ``create_loss_f`` by signature (a factory
    with **kwargs receives everything). A key that no consumer names raises ValueError when
    every factory has an explicit signature, and is logged as a warning otherwise, so a
    misspelled hyperparameter cannot be silently ignored.
    '''
    def __init__(self, 
                 create_net=FeedForwardNet,
                 create_optim=_default_create_optim,
                 create_loss_f=MSELoss.create,
                 create_scheduler=_default_create_scheduler,
                 writer=None,
                 scale_output=True,
                 **hyperparameters):
        super().__init__()
        self.create_net = create_net
        self.create_loss_f = create_loss_f
        self.create_optim = create_optim
        self.create_scheduler = create_scheduler
        self.n_epochs = hyperparameters.get('n_epochs', 20)
        self.save_losses = hyperparameters.get('save_losses', False)
        self.save_state_dicts = hyperparameters.get('save_state_dicts', False)
        self.batch_size = hyperparameters.pop('batch_size', 1_000_000)
        self.hyperparameters = hyperparameters
        self._check_hyperparameters()
        self.net = cuda(self.create_net(1, 1, **_kwargs_for(self.create_net, self.hyperparameters)))
        self.valid_feature_columns = None
        self.valid_label_columns = None
        self.saved_losses = []
        self.saved_state_dicts = []
        # tqdm wraps an iterable; identity returns it unchanged
        self.progress = tqdm if hyperparameters.get('progress', False) else identity
        self.progress_batch = tqdm if hyperparameters.get('progress_batch', False) else identity
        self.seed = hyperparameters.get('torch_seed', 1932)
        self.writer = writer
        if self.writer == True:
            self.writer = SummaryWriter(log_dir=f'runs/instance-{self.seed}')
        if isinstance(self.writer, SummaryWriter):
            _LOGGER.debug('Creating dir')
            os.makedirs(self.writer.log_dir, exist_ok=True)
            
        self.scale_output = scale_output
        self.scale = 1

    def _check_hyperparameters(self):
        factories = {'create_net': self.create_net, 'create_optim': self.create_optim,
                     'create_scheduler': self.create_scheduler, 'create_loss_f': self.create_loss_f}
        known = set(_MODEL_HYPERPARAMETERS)
        open_ended = []
        for name, fn in factories.items():
            names, var_kw = _accepted_kwargs(fn)
            known |= names
            if var_kw:
                open_ended.append(name)
        unknown = sorted(set(self.hyperparameters) - known)
        if not unknown:
            return
        if open_ended:
            _LOGGER.warning(f'{self.__class__.__name__}: hyperparameters {unknown} are not named by the model or any '
                            f'factory signature; forwarded blindly to {open_ended} (misspelled?)')
        else:
            raise ValueError(f'{self.__class__.__name__}: unknown hyperparameters {unknown}')

    def reinstatiate_writer(self):
        self.writer = SummaryWriter(self.writer.log_dir)

    def _train(self, net, loss_f, t_X, t_Y, n_epochs, t_X_val, t_Y_val, *loss_args, **loss_kwargs):
        def run_train_batch(loader, tag='train'):
            epoch_loss = 0

            for features, labels in self.progress_batch(loader):
                self.optim.zero_grad()
                yp = net(features)
                loss = loss_f(yp, labels, *loss_args, **loss_kwargs)
                loss.backward()
                self.optim.step()
                loss_v = loss.detach().clone().item()
                epoch_loss += loss_v
            
            self._log_epoch_end(epoch, epoch_loss, yp, labels, tag)
            
            return epoch_loss
        
        def run_validation_batch(loader, tag='val'):
            epoch_val_loss = 0
            with torch.no_grad():
                for features, labels in self.progress_batch(loader):
                    yp_val = net(features)
                    loss_val = loss_f(yp_val, labels, *loss_args, **loss_kwargs)
                    epoch_val_loss += loss_val.detach().clone().item()
                    break

            self._log_epoch_end(epoch, epoch_val_loss, yp_val, labels, tag)
            return epoch_val_loss

        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.optim = self.create_optim(net.parameters(), **_kwargs_for(self.create_optim, self.hyperparameters))
        self.scheduler = self.create_scheduler(self.optim, **_kwargs_for(self.create_scheduler, self.hyperparameters))

        loader = Loader(t_X, t_Y, batch_size=self.batch_size, shuffle=False)
        loader_val = Loader(t_X_val, t_Y_val, batch_size=self.batch_size, shuffle=False) if t_X_val is not None else None
        
        epoch_losses = []
        for epoch in self.progress(range(n_epochs)):
            epoch_loss = run_train_batch(loader, 'train')
            if loader_val is not None:
                run_validation_batch(loader_val, 'val')

            epoch_losses.append(epoch_loss)
            self.scheduler.step(epoch_loss)

        
        return epoch_losses

    def _log_epoch_end(self, epoch, epoch_loss, yp, labels, tag='train', *args, **kwargs):
        from torcheval.metrics import R2Score

        if self.writer is not None:
            self.writer.add_scalar(f'loss/{tag}', epoch_loss, epoch)
            self.writer.add_scalar(f'lr/{tag}', self.optim.param_groups[0]['lr'], epoch)
            self.writer.add_scalar(f'std(pred)/{tag}', yp.std().item(), epoch)
            

            labels_with_var = torch.std(labels, axis=0) > 0
            self.writer.add_scalar(f'r2/{tag}', R2Score().update(yp[:, labels_with_var], labels[:, labels_with_var]).compute().item(), epoch)
            self.writer.add_scalar(f'hitrate/{tag}', (((torch.sign(labels[labels != 0]) * torch.sign(yp[labels != 0])).mean()  + 1) / 2).item(), epoch)
            
            for k, v in mean_absolute_grads(self.net).items():
                self.writer.add_scalar(k, v, epoch)

    def _align_input_data(self, x, y):
        return x.dropna(how='all', axis=1).align(y.dropna(how='all', axis=1).dropna(how='all'), join='inner', axis=0)
    
    def find_learning_rate(self, x, y, apply_test=True, batch_size=1024, shuffle=False):
        '''
        Uses pytorch-lr-finder https://github.com/davidtvs/pytorch-lr-finder to find optimal learning rate

        Call function and then see results using lr_finder.plot()

        Use with a grain of salt.

        :param x pd.DataFrame: Same as input to fit
        :param y pd.DataFrame: Same as input to fit
        :param apply_test bool: If true, runs the range test, otherwise returns the object and data loader for the caller to run it
        '''
        from torch.utils.data import TensorDataset, DataLoader
        from torch_lr_finder import LRFinder

        x, y = self._align_input_data(x, y)
        net = self.create_net(x.shape[1], y.shape[1], **_kwargs_for(self.create_net, self.hyperparameters))
        dl = DataLoader(TensorDataset(torch.from_numpy(x.values).float(), torch.from_numpy(y.values).float()), batch_size=batch_size, shuffle=shuffle)
        lr_finder = LRFinder(net, self.create_optim(net.parameters()), self.create_loss_f(), device="cuda" if cuda() else None)
        
        if apply_test:
            lr_finder.range_test(dl, start_lr=1e-6, end_lr=1e-2, num_iter=10000)
            return lr_finder
        
        return lr_finder, dl

    def fit(self, x, y, x_val=None, y_val=None, *args, **kwargs):
        if bool(x.isna().any().any()):
            _LOGGER.warning(f'Found nans in X input to {str(self)} when running fit')
        if bool(y.isna().any().any()):
            _LOGGER.warning(f'Found nans in Y input to {str(self)} when running fit')
            
        # CPU float reduction order depends on the intra-op thread count, so a fit is
        # only bit-reproducible at a fixed count. Pin it here rather than relying on
        # each caller's env, where an inherited OMP_NUM_THREADS silently wins over
        # setdefault.
        _n_threads = int(os.environ.get('EBT_NN_TORCH_THREADS', '1'))
        if torch.get_num_threads() != _n_threads:
            torch.set_num_threads(_n_threads)

        ax, ay = self._align_input_data(x, y)
        self.ax = ax
        self.ay = ay
        ax_val, ay_val = self._align_input_data(x_val, y_val) if x_val is not None and y_val is not None else (x_val, y_val)

        if ax.empty or ay.empty:
            return self
        
        self.valid_label_columns = ay.columns

        self.valid_feature_columns = ax.columns

        self.net = cuda(self.create_net(ax.shape[1], ay.shape[1], **_kwargs_for(self.create_net, self.hyperparameters)))
        self.net.train()
        t_X = cuda(torch.tensor(ax.to_numpy()).float())
        t_y = cuda(torch.tensor(ay.to_numpy()).float())
        t_X_val = cuda(torch.tensor(ax_val.to_numpy()).float()) if ax_val is not None else ax_val
        t_y_val = cuda(torch.tensor(ay_val.to_numpy()).float()) if ay_val is not None else ay_val

        self.loss_f = self.create_loss_f(**_kwargs_for(self.create_loss_f, self.hyperparameters))
        losses = list(self._train(self.net, self.loss_f, t_X, t_y, self.n_epochs, t_X_val, t_y_val, *args, **kwargs))

        if self.save_losses:
            self.saved_losses.append(pd.Series(losses))
        if self.save_state_dicts:
            self.saved_state_dicts.append({k: v.detach().cpu() for k, v in self.net.state_dict().items()})

        if self.scale_output:
            self.scale = (self.predict(ax, ay, scale=False).shift(2) * ay).replace(0, np.nan).dropna(how='all').sum(axis=1).std()
            _LOGGER.debug(f'scale: {self.scale}')

        return self

    def predict(self, x, y, net=None, *args, **kwargs):
        if bool(x.isna().any().any()):
            _LOGGER.warning(f'Found nans in X input to {str(self)} when running predict')
        if bool(y.isna().any().any()):
            _LOGGER.warning(f'Found nans in Y input to {str(self)} when running predict')

        if self.valid_feature_columns is None:
            columns = [] if self.valid_label_columns is None else self.valid_label_columns
            return pd.DataFrame(np.nan, columns=columns, index=x.index)

        net = net or self.net
        net.eval()


        t_X = cuda(torch.tensor(x.reindex(columns=self.valid_feature_columns).fillna(0).to_numpy()).float())
        
        yp = net(t_X).detach().cpu().numpy()
        result = pd.DataFrame(data=yp, columns=self.valid_label_columns, index=x.index).dropna(how='all')
        if kwargs.get('scale', True):
            _LOGGER.debug('scaling')
            result = result / self.scale

        return result


class LSTMNet(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1, num_layers=1, dropout_lstm=0.5, bias=False,
                 active_f=None, dropout_linear=0.5, network_sma_length=None):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_lstm, bias=bias)
        self.fc = get_linear_block(hidden_size,
                                   output_size,
                                   bias=bias,
                                   activation=active_f,
                                   dropout_p=dropout_linear,
                                   network_sma_length=network_sma_length)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last hidden state
        return out


def _lstm_create_optim(params, lr=0.01):
    return torch.optim.Adam(params, lr=lr)


def _lstm_create_scheduler(optim, factor=0.9, patience=10):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=factor, patience=patience)


class LSTMNeuralNetModel(NeuralNetModel):
    def __init__(self,
                 create_net=LSTMNet,
                 create_optim=_lstm_create_optim,
                 create_loss_f=MSELoss.create,
                 create_scheduler=_lstm_create_scheduler,
                 sequence_length=10,
                 **hyperparameters,
                ):
        super().__init__(create_net, create_optim, create_loss_f, create_scheduler, **hyperparameters)
        self.sequence_length = sequence_length

    def _train(self, net, loss_f, t_X, t_Y, n_epochs, t_X_val, t_Y_val, *loss_args, **loss_kwargs):
        n_rows = t_X.shape[0]
        t_X, t_Y = create_lstm_dataset(t_X, t_Y, self.sequence_length)
        t_X_val, t_Y_val = create_lstm_dataset(t_X_val, t_Y_val, self.sequence_length) if t_X_val is not None else (None, None)
        if t_X is None:
            raise ValueError(f'{self.__class__.__name__}: {n_rows} rows is fewer than sequence_length={self.sequence_length}')

        return super()._train(net, loss_f, t_X, t_Y, n_epochs, t_X_val, t_Y_val, *loss_args, **loss_kwargs)

    def predict(self, x, y, net=None, *args, **kwargs):
        if self.valid_feature_columns is None:
            columns = [] if self.valid_label_columns is None else self.valid_label_columns
            return pd.DataFrame(np.nan, columns=columns, index=x.index)

        net = net or self.net
        net.eval()

        t_X = cuda(torch.tensor(x.reindex(columns=self.valid_feature_columns).fillna(0).to_numpy()).float())

        t_X = create_lstm_dataset(t_X, None, self.sequence_length)
        if t_X is None:
            return pd.DataFrame()

        loader = Loader(t_X, y=None , batch_size=self.batch_size, shuffle=False)
        
        yps = []
        for features in self.progress_batch(loader):
            yps.append(net(features).detach().cpu().numpy())

        result = pd.DataFrame(data=np.concatenate(yps), columns=self.valid_label_columns, index=x.index[self.sequence_length:])
        if kwargs.get('scale', True):
            result = result / self.scale
        return result

