# -*- coding: utf-8 -*-
"""
@author: AmirPouya Hemmasian (a.pouyahemmasian@gmail.com) (ahemmasi@andrew.cmu.edu)
"""
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

Device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_h5_data(
        data_path,
        t_start = None, # start time-step (not actual starting time) (inclusive)
        t_end = None, # end time-step (not actual starting time) (inclusive)
        dt = 1, # time-step interval when loading
        rx = None, # resolution for spatial dimensions
        dx = None, # dx of downsampling if rx is not specified.
        device = Device,
        load_now = False,
        verbose = False
):
    """
    Extracts variables and coordinates from an h5 or hdf5 file as dictionaries.
    Since the spatial resolution might be too high, there is an option to downsample it.

    Returns:
        N : int : Number of trajectories
        nt : int : Number of snapshots
        Vars : OrderedDict : Dictionary of variables
            Each item in the dictionary has {'data': Tensor, 'stat':stat_data}
            where for each variable, we store the data as well as some statistics
            which will be useful for normalizaitons later
        Coords : OrderedDict : Dictionary of coordinates
            Each item in the dictionary has {'data': , 'shape':, 'min':, 'max':}
    """
    N, nt = None, None
    Vars = {}
    Coords = {}
    
    if t_end is not None:
        t_end = t_end + 1 # to make it inclusive

    def get_d(r, r_new = None, ds_by = None):
        if r_new is not None:
            assert r%r_new == 0, 'The resolution should be divisible by the new resolution'
            d = r//r_new
        elif ds_by is not None:
            d = ds_by
        else:
            d = 1
        return d

    f = h5py.File(data_path, 'r')
    for name, obj in f.items():
        if not isinstance(obj, h5py.Dataset):
            continue
        if name.endswith('coordinate'):
            if name[0] == 't':
                data = torch.as_tensor(obj[t_start:t_end:dt], dtype=torch.float32, device=device if load_now else 'cpu')

                # WARNING: the following block is hardcoded for PDEBench data
                # because for some reason, t-coordinate usually has an extra time-step at the end
                # extra_ts = (len(data)-1)%10
                # if extra_ts > 0:
                #     data = data[:-extra_ts]

            elif name[0] in ['x', 'y', 'z']:
                d = get_d(obj.shape[0], rx, dx)
                data = torch.as_tensor(obj[::d], dtype=torch.float32, device=device if load_now else 'cpu')
            Coords[name] = {
                'data': data,
                'shape': tuple(data.shape),
                'min': data.min().item(),
                'max': data.max().item(),
                }
        else:
            slicer = [slice(None)] + [ 
                slice(t_start, t_end, dt) # time slices
                ] + [
                slice(None, None, get_d(r, rx, dx)) # space slices
                 for r in obj.shape[2:]
            ]
            if name == 'tensor':
                name = 'u'
            # data = torch.as_tensor(obj[*slicer], dtype=torch.float32, device=device if load_now else 'cpu')
            # The above line needs python >=3.11 but the line below is the equivalent for older version
            data = torch.as_tensor(obj[eval(', '.join([str(slc) for slc in slicer]))], 
                                   dtype=torch.float32, device=device if load_now else 'cpu')
            Vars[name] = {
                'data': data,
                'shape': tuple(data.shape),
                'min': data.min().item(),
                'max': data.max().item(),
                'mean': data.mean().item(),
                'std': data.std().item(),
                'meanabs': data.abs().mean().item(),
                'maxabs': data.abs().max().item(),
                'RMS': ((data**2).mean()**0.5).item()
                }

            N = N or data.shape[0]
            assert N == data.shape[0]
            nt = nt or data.shape[1]
            assert nt == data.shape[1]
    
    if verbose:
        print(50*'-')
        for coord, info in Coords.items():
            print(
                coord, 
                '| shape', info['shape'],
                *[f'| {key} {val:.6f}' for key, val in info.items() if key not in ['data', 'shape']]
                )
            print(50*'-')
        for var, info in Vars.items():
            print(
                var,
                '| shape', info['shape'],
                *[f'| {key} {val:.6f}' for key, val in info.items() if key not in ['data', 'shape']]
                )
            print(50*'-')

    f.close()
    return N, nt, Vars, Coords


class PDEDataset(Dataset):
    def __init__(
            self,
            data_path : str,
            t_start : int = None,
            t_end : int = None,
            dt : int = None,
            rx : int = None,
            dx : int = None,
            device = Device,
            load_now = True,
            verbose = False
            ):
        super().__init__()

        if verbose:
            print('Loading', data_path, '...')

        self.N, self.nt, self.Vars, self.Coords = extract_h5_data(
            data_path = data_path,
            t_start = t_start,
            t_end = t_end,
            dt = dt,
            rx = rx,
            dx = dx,
            device = device,
            load_now = load_now,
            verbose = verbose
            )
        
        if verbose:
            print('LOADED!')
            print(50*'=')

        self.device = device
        self.norm_mode = 'none'
        self.indexes = torch.arange(self.N)
        self.dt = 1
        self.in_snapshots = 1
        self.out_snapshots = 1
        self.rollout = 1
        self.skip = 0

    def normalize_manually(self, norm_cs:dict):
        """
        Normalize the data manually using the provided constant for each variable
        """
        if self.norm_mode != 'none':
            self.normalize('none')
                
        for var in self.Vars:
            self.Vars[var]['data'] /= norm_cs[var]

        self.norm_mode = 'manual'
        self.manual_norm_cs = norm_cs

    def normalize(self, mode:str='none'):
        if mode == self.norm_mode:
            # The data is already normalized in the desired way
            return
        
        if mode == 'none':
            # Reversing whatever normalization is in effect
            if self.norm_mode == 'standard':
                for var in self.Vars.keys():
                    self.Vars[var]['data'] *= self.Vars[var]['std']
                    self.Vars[var]['data'] += self.Vars[var]['mean']
            elif self.norm_mode == 'minmax':
                for var in self.Vars.keys():
                    self.Vars[var]['data'] *= self.Vars[var]['max'] - self.Vars[var]['min']
                    self.Vars[var]['data'] += self.Vars[var]['min']
            elif self.norm_mode == 'maxabs':
                for var in self.Vars.keys():
                    self.Vars[var]['data'] *= self.Vars[var]['maxabs']
            elif self.norm_mode == 'meanabs':
                for var in self.Vars.keys():
                    self.Vars[var]['data'] *= self.Vars[var]['meanabs']
            elif self.norm_mode == 'RMS':
                for var in self.Vars.keys():
                    self.Vars[var]['data'] *= self.Vars[var]['RMS']
            elif self.norm_mode == 'manual':
                for var in self.Vars.keys():
                    self.Vars[var]['data'] *= self.manual_norm_cs[var]
                self.manual_norm_cs = None
            self.norm_mode = 'none'
            return

        if self.norm_mode != 'none':
            # Reversing the current normalization
            self.norlamize('none')
        
        if mode == 'standard':
            for var in self.Vars.keys():
                self.Vars[var]['data'] -= self.Vars[var]['mean']
                self.Vars[var]['data'] /= self.Vars[var]['std']
        elif mode == 'minmax':
            for var in self.Vars.keys():
                self.Vars[var]['data'] -= self.Vars[var]['min']
                self.Vars[var]['data'] /= self.Vars[var]['max'] - self.Vars[var]['min']
        elif mode == 'absmax':
            for var in self.Vars.keys():
                self.Vars[var]['data'] /= self.Vars[var]['abs_max']
        elif mode == 'absmean':
            for var in self.Vars.keys():
                self.Vars[var]['data'] /= self.Vars[var]['abs_mean']
        elif mode == 'RMS':
            for var in self.Vars.keys():
                self.Vars[var]['data'] /= self.Vars[var]['RMS']
        else:
            raise ValueError('Invalid normalization mode')

        self.norm_mode = mode

    def config(
            self,
            subset: float = 1.0,
            where: str = 'random',
            reverse : bool = False, 
            seed : int = 0,
            frac : float = 1.0,
            frac_seed : int = 0
            ):

        chosen = np.full(self.N, True)
        
        assert subset >= 0.0
        assert subset <= 1.0

        assert where in ['random', 'top', 'bottom', 'middle']
        n_subset = round(subset*self.N)

        if where == 'random':
            np.random.seed(seed)
            cut = np.random.permutation(self.N) <= n_subset-1
        
        elif where == 'top':
            cut = np.arange(self.N) > (self.N - n_subset)

        elif where == 'bottom':
            cut = np.arange(self.N) <= n_subset
        
        elif where == 'middle':
            cut = np.arange(self.N) > (self.N//2 - n_subset//2) & np.arange(self.N) <= (self.N//2 + n_subset//2)

        if reverse:
            cut = ~cut
        
        chosen = chosen & cut

        chosen_indexes = np.arange(self.N)[chosen]

        if frac == 1.0:
            self.indexes = chosen_indexes
            return self.indexes
    
        np.random.seed(frac_seed)
        perm = np.random.permutation(len(chosen_indexes))
        self.indexes = chosen_indexes[perm[:round(frac*len(chosen_indexes))]]

        return self.indexes

    
    def save(self, path:str):
        with h5py.File(path, 'w') as f:
            for attr in self.Vars.keys():
                f.create_dataset(
                    name = attr, 
                    data = self.Vars[attr]['data'].cpu().numpy()
                    )
            for attr in self.Coords.keys():
                f.create_dataset(
                    name = attr, 
                    data = self.Coords[attr]['data'].cpu().numpy()
                    )
    
    def samples_per_traj(self) -> int:
        return self.nt - (self.in_snapshots + self.rollout*(self.skip + self.out_snapshots) - 1) * self.dt
    
    def get_index(self, i):
        traj_idx = i // self.samples_per_traj()
        t_start = i % self.samples_per_traj()
        return traj_idx, t_start
    
    def config_autoregression(
            self,
            dt : int = None, 
            in_snapshots : int = None,
            out_snapshots : int = None, 
            rollout : int = None,
            skip : int = None,
            ):
        if dt is None: dt = self.dt
        if in_snapshots is None: in_snapshots = self.in_snapshots
        if out_snapshots is None: out_snapshots = self.out_snapshots
        if rollout is None: rollout = self.rollout
        if skip is None: skip = self.skip

        assert dt > 0
        assert in_snapshots > 0
        assert out_snapshots > 0
        assert rollout >= 0
        assert (in_snapshots + rollout*(skip+out_snapshots) - 1) * dt < self.nt, 'The selected setting exceed the length of the trajectory!'
        if skip > 0 and rollout > 1:
            assert in_snapshots <= out_snapshots, 'if skip > 0, we need in_snapshots <= out_snapshots'

        self.dt = dt
        self.in_snapshots = in_snapshots
        self.out_snapshots = out_snapshots
        self.rollout = rollout
        self.skip = skip

    def __len__(self):
        return len(self.indexes)*self.samples_per_traj()

    def __getitem__(self, i : int) -> list:
        traj_idx, t_start = self.get_index(i)
        sample = {var:
            [
            self.Vars[var]['data'][
                self.indexes[traj_idx], # index of trajectory
                t_start : t_start + self.dt*self.in_snapshots : self.dt, # the selected snapshots of this trajectory
                ...] # The input chunk
                ] + [
            self.Vars[var]['data'][
                self.indexes[traj_idx], # index of trajectory
                t_start + self.dt*(self.in_snapshots + (j+1)*self.skip + j*self.out_snapshots) : t_start + self.dt*(self.in_snapshots + (j+1)*self.skip + (j+1)*self.out_snapshots) : self.dt,
                ...]
                for j in range(self.rollout) # The output chunks
                ]
        for var in self.Vars
        }

        # old code for concatenating the variables along the time dimension
        # We concatenate different variables along the time dimension
        # dim=1 contains both several time steps for all the variables
        # sample = [torch.cat(x) for x in zip(*sample)]
        sample = [{var: sample[var][j].to(self.device) for var in self.Vars} for j in range(self.rollout+1)]

        return sample
