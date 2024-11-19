"""
Author: AmirPouya Hemmasian (a.pouyahemmasian@gmail.com) (ahemmasi@andrew.cmu.edu)
"""
from einops import rearrange
import torch
from torch import nn
from utils_train import parse_csv, parse_csv_scalers

Device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Linear(nn.Linear):
    """
    Point-Wise Linear Layer for PDE data 
    where the feature/channel dimension is the second dimension (after batch)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = rearrange(x, 'b c ... -> b ... c')
        x = super().forward(x)
        x = rearrange(x, 'b ... c -> b c ...')
        return x


class Projector(nn.Module):
    """
    A simple linear projector for input and output variables.
    Code is simple and self-explanatory.

    the input/output space is a dictionary of input/output variables, each with shape (B, C, ...) where ... are spatial dimensions
    the projection space is a single tensor of shape (B, C, ...) where ... are spatial dimensions
    """
    def __init__(
            self,
            in_vars, # iterable of strings
            in_dim : int,
            out_dim : int,
            proj_dim : int,
            out_vars = None, # iterable of strings. If None, in_vars are used
            device = Device,
    ):
        self.device = device
        super().__init__()
        self.in_vars = list(in_vars)
        self.out_vars = list(out_vars) or self.in_vars
        self.in_projector = nn.ModuleDict({
            var: Linear(in_dim, proj_dim, device=device)
            for var in self.in_vars
        })
        self.out_projector = nn.ModuleDict({
            var: Linear(proj_dim, out_dim, device=device)
            for var in self.out_vars
        })

    def in_proj(self, x: dict) -> torch.FloatTensor:
        return torch.stack([self.in_projector[var](x[var]) for var in x]).sum(dim=0)

    def out_proj(self, x: torch.FloatTensor) -> dict:
        return {var: self.out_projector[var](x) for var in self.out_vars}
    
    def transfer_from(
            self,
            source_projector : nn.Module,
            transfer_in_vars : str = 'all', # comma separated variables to transfer from source to target.
            transfer_in_vars_scalers : str = None, # comma separated scalers for each variable.
            transfer_out_vars : str = 'all',
            transfer_out_vars_scalers : str = None
            ):
        """
        inputs for in_vars or out_vars:
            separate variables by comma
            If the same variable is being transferred, use it by itself
            if you want to transfer a variable to another variable, use '>'
        examples:
        u,v,w :
            u projector of source model transfered to u projector of this model
            v projector of source model transfered to v projector of this model
        u,v>w:
            u projector of source model transfered to u projector of this model
            v projector of source model transfered to w projector of this model
        """
        transfer_parser = lambda x: list(x.split('>')) if '>' in x else x

        in_vars = parse_csv(transfer_in_vars, full=self.in_vars, func=transfer_parser)

        in_vars_scalers = parse_csv_scalers(transfer_in_vars_scalers)
        
        if len(in_vars_scalers) == 1:
            in_vars_scalers = in_vars_scalers * len(in_vars)
        elif len(in_vars_scalers) != len(in_vars):
            raise ValueError('number of scalers should be the same as number of variables')

        out_vars = parse_csv(transfer_out_vars, full=self.out_vars, func=transfer_parser)

        out_vars_scalers = parse_csv_scalers(transfer_out_vars_scalers)
        
        if len(out_vars_scalers) == 1:
            out_vars_scalers = out_vars_scalers * len(out_vars)
        elif len(out_vars_scalers) != len(out_vars):
            raise ValueError('number of scalers should be the same as number of variables')

        for i, source_target in enumerate(in_vars):
            if not isinstance(source_target, list): source_var, target_var = source_target, source_target
            else: source_var, target_var = source_target
            self.in_projector[target_var].weight.data = source_projector.in_projector[source_var].weight.data * in_vars_scalers[i]
            self.in_projector[target_var].bias.data = source_projector.in_projector[source_var].bias.data * in_vars_scalers[i]

        for i, source_target in enumerate(out_vars):
            if not isinstance(source_target, list): source_var, target_var = source_target, source_target
            else: source_var, target_var = source_target
            self.out_projector[target_var].weight.data = source_projector.out_projector[source_var].weight.data * out_vars_scalers[i]
            self.out_projector[target_var].bias.data = source_projector.out_projector[source_var].bias.data * out_vars_scalers[i]
    
    def set_trainability(
            self, 
            in_vars : str = None,
            out_vars : str = None,
            trainable : bool = True
            ):
        """
        each argument is comma separated list of variables, like : 'u,v,w'
        If you do not want to touch any variable, pass an empty string like ''
        """
        in_vars = parse_csv(in_vars, full=self.in_vars)
        for var in in_vars:
            self.in_projector[var].requires_grad_(trainable)

        out_vars = parse_csv(out_vars, full=self.out_vars)
        for var in out_vars:
            self.out_projector[var].requires_grad_(trainable)


class FeedForward(nn.Module):
    """
    Base code taken from:
    https://github.com/alasdairtran/fourierflow/blob/main/fourierflow/modules/feedforward.py
    Excluded LayerNorm and Dropout for simplicity.
    """
    def __init__(
            self, 
            dim: int, 
            factor: int = 2, 
            n_layers: int = 2,
            device = Device,
            ):
        super().__init__()
        self.device = device
        self.layers = []
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.extend([
                Linear(in_dim, out_dim, device=device),
                nn.ReLU(inplace=True)
            ])
        self.layers = nn.Sequential(*self.layers[:-1])

    def forward(self, x):
        return self.layers(x)
    

class Factorized_Spectral_Layer(nn.Module):
    """
    This is a general Factorized Spectral Convolution Layer for either 1D, 2D, or 3D spatial data.
    The number of spatial dimensions is implied by the length of fourier_modes.

    Base code taken from:
    https://github.com/alasdairtran/fourierflow/blob/main/fourierflow/modules/factorized_fno/mesh_2d.py
    """
    def __init__(
            self, 
            width: int,
            fourier_modes, # iterable of integers
            share_fourier = False,
            ff_factor = 2,
            ff_n_layers = 2,
            device = Device
            ):
        super().__init__()
        self.width = width
        self.fourier_modes = fourier_modes

        self.n_spatial_dims = len(fourier_modes)
        assert self.n_spatial_dims in [1, 2, 3], f'Only up to 3D supported. got {self.n_spatial_dims}D'

        if self.n_spatial_dims == 1:
            share_fourier = True
        self.share_fourier = share_fourier

        # this was there in the base code, so I'm leaving it here:
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
   
        self.fourier_weight = nn.ParameterList()
        for i in range(self.n_spatial_dims):
            n_modes = fourier_modes[i]
            weight = torch.FloatTensor(width, width, n_modes, 2)
            weight = weight.to(device)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param)
            self.fourier_weight.append(param)
            if share_fourier:
                assert all([fourier_modes[0] == n_modes for n_modes in fourier_modes]), 'number of fourier modes should be the same across all axes for them to share'
                break

        self.feedforward = FeedForward(dim=width, factor=ff_factor, n_layers=ff_n_layers, device=device)
    
    def complex_matmul(self, input, weight, spatial_dim):
        in_str = 'bi' + 'xyz'[:self.n_spatial_dims]
        weight_str = 'io' + 'xyz'[spatial_dim]
        out_str = 'bo' + 'xyz'[:self.n_spatial_dims]
        return torch.einsum(
            f'{in_str},{weight_str}->{out_str}',
            input, 
            torch.view_as_complex(weight)
            )
    
    def forward_fourier_dim(self, x, dim): # dim is the spatial axis (0 is x, 1 is y, 2 is z)

        i = dim + 2 # skipping batch and time/feature/channel
        # i is the tensor dimension corresponding to the spatial dimension dim
        n_modes = self.fourier_modes[dim]
        shape = x.shape

        ft = torch.fft.rfft(x, dim=i, norm='ortho')        
        out_ft_shape = list(shape)
        out_ft_shape[i] = shape[i]//2 + 1
        out_ft = ft.new_zeros(*out_ft_shape)

        slicer = [slice(None)] * len(shape)
        slicer[i] = slice(n_modes)
        
        out_ft[slicer] = self.complex_matmul(
            input = ft[slicer],
            weight = self.fourier_weight[dim if not self.share_fourier else 0],
            spatial_dim = dim
            )
        out = torch.fft.irfft(out_ft, n=shape[i], dim=i, norm='ortho')
        return out
    
    def forward(self, x):
        outs = []
        for dim in range(self.n_spatial_dims):
            out = self.forward_fourier_dim(x, dim)
            outs.append(out)
        outs = torch.stack(outs).sum(dim=0)
        outs = self.feedforward(outs)
        return outs
    
    def transfer_from(
            self, 
            source_layer : nn.Module,
            fourier : bool = True,
            scaler : float = None,
            ff : bool = True
            ):
        assert source_layer.n_spatial_dims in [1, self.n_spatial_dims], 'source should be either 1D or same dims'
        if scaler is None: scaler = 1.0

        if fourier:
            # len(fourier_weight) for wach model is either 1 (1D or shared) or n_spatial_dims (different for each dim)
            # cannot transfer if source has multiple and self has ones
            assert len(source_layer.fourier_weight) in [1, len(self.fourier_weight)], 'number of source fourier weights should be the same or 1'

            for i in range(len(self.fourier_weight)):
                j = 0 if len(source_layer.fourier_weight) == 1 else i

                self.fourier_weight[i].data = source_layer.fourier_weight[j].data * scaler

        if ff:
            self.feedforward.load_state_dict(source_layer.feedforward.state_dict())

    def set_trainability(
            self, 
            fourier : bool = True,
            ff : bool = True,
            trainable : bool = True
            ):
        if fourier:
            self.fourier_weight.requires_grad_(trainable)
        if ff:
            self.feedforward.requires_grad_(trainable)
    

class FFNO(nn.Module):
    """
    Base code taken from:
    https://github.com/alasdairtran/fourierflow/blob/main/fourierflow/modules/factorized_fno/mesh_2d.py

    This is a general FFNO for either 1D, 2D, or 3D spatial data.
    The number of spatial dimensions is determined by the length of fourier_modes.
    """
    def __init__(
            self,
            in_vars, # iterable of strings
            out_vars, # iterable of strings
            in_dim : int, 
            out_dim : int,

            n_layers: int,
            width: int,
            fourier_modes: tuple,
            share_fourier = False,

            ff_n_layers = 2,
            ff_factor = 2,

            device = Device
            ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.fourier_modes = fourier_modes
        self.device = device

        self.projector = Projector(
            in_vars = in_vars,
            out_vars = out_vars,
            in_dim = in_dim,
            out_dim = out_dim,
            proj_dim = width,
            device = device,
            )
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            Factorized_Spectral_Layer(
                width = width,
                fourier_modes = fourier_modes,
                share_fourier = share_fourier,
                ff_factor = ff_factor,
                ff_n_layers = ff_n_layers,
                device = device
                ) for _ in range(n_layers)
                ])

    def forward(self, x):
        x = self.projector.in_proj(x)
        for layer in self.layers:
            b = layer(x)
            x = x + b
        b = self.projector.out_proj(b)
        return b
    
    def transfer_from(
            self,
            source_ffno : nn.Module,     

            transfer_in_vars : str = 'all',
            transfer_in_vars_scalers : str = None, 
            transfer_out_vars : str = 'all', 
            transfer_out_vars_scalers : str = None, 

            transfer_fourier_layers : str = 'all',
            transfer_fourier_scalers : str = None,
            transfer_ff_layers : str = 'all',
    ):
        assert self.n_layers == source_ffno.n_layers, 'number of spectral layers should be the same'
        source_ffno.to(self.device)
        
        self.projector.transfer_from(
            source_projector = source_ffno.projector, 
            transfer_in_vars = transfer_in_vars, 
            transfer_in_vars_scalers = transfer_in_vars_scalers,
            transfer_out_vars = transfer_out_vars,
            transfer_out_vars_scalers = transfer_out_vars_scalers
            )

        transfer_fourier_layers = parse_csv(transfer_fourier_layers, full=range(self.n_layers), func=int)
        
        transfer_fourier_scalers = parse_csv_scalers(transfer_fourier_scalers)
        
        if len(transfer_fourier_scalers) == 1:
            transfer_fourier_scalers = transfer_fourier_scalers * len(transfer_fourier_layers)

        elif len(transfer_fourier_scalers) != len(transfer_fourier_layers):
            raise ValueError('number of scalers should be the same as number of layers')

        transfer_ff_layers = parse_csv(transfer_ff_layers, full=range(self.n_layers), func=int)
    
        for i in range(self.n_layers):
            self.layers[i].transfer_from(
                source_layer = source_ffno.layers[i], 
                fourier = i in transfer_fourier_layers,
                scaler = transfer_fourier_scalers[transfer_fourier_layers.index(i)],
                ff = i in transfer_ff_layers
                )
            
    def set_trainability(
            self,
            in_vars : str = 'all', # comma separated variables to set trainability.
            out_vars : str = 'all', # comma separated variables to set trainability.
            fourier_layers : str = 'all', # comma separated layers to set trainability.
            ff_layers : str = 'all', # comma separated layers to set trainability.
            trainable : bool = True # whether to set trainable or not
            ):
        
        self.projector.set_trainability(
            in_vars = in_vars,
            out_vars = out_vars,
            trainable = trainable
            )
        
        fourier_layers = parse_csv(fourier_layers, full=range(self.n_layers), func=lambda x: int(x)%self.n_layers)
        ff_layers = parse_csv(ff_layers, full=range(self.n_layers), func=lambda x: int(x)%self.n_layers)

        for i in range(self.n_layers):
            self.layers[i].set_trainability(
                fourier = i in fourier_layers,
                ff = i in ff_layers,
                trainable = trainable
                )
