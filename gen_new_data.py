import os
import h5py
from tqdm import tqdm
import itertools
import argparse
from copy import deepcopy
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch

# mail-02
PDEBench_data_dir = '/media/pouya/DATA1/PDE_data/PDEBench'
data_dir = '/media/pouya/DATA1/PDE_data/FPNO_data'

# others
data_dir = '/home/pouya/FPNO_data'

os.makedirs(data_dir, exist_ok=True)

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Euler_Solver_Periodic:

    def __init__(self, grid = (4,), device = Device):

        self.grid = grid
        self.device = device
        self.ndims = len(grid)
        for n in grid:
            assert n > 2, 'Grid size must be at least 4 in each dimension'
        
        self.rng = [np.arange(n) for n in grid]
        self.n = np.prod(grid)

        # indexes for explicit schemes
        # each of these corresponds to the previous or next cell for ALL vells.
        self.ctr = [slice(None) for _ in grid]
        self.prev = [[slice(None) for _ in grid] for _ in grid]
        self.next = [[slice(None) for _ in grid] for _ in grid]
        for i in range(self.ndims):
            self.prev[i][i] = (np.arange(grid[i]) - 1) % grid[i]
            self.next[i][i] = (np.arange(grid[i]) + 1) % grid[i]

        # matrices for implicit schemes
        self.ctr_A = torch.zeros((*grid, *grid), device=device)
        self.prev_A = [torch.zeros((*grid, *grid), device=device) for _ in range(self.ndims)]
        self.next_A = [torch.zeros((*grid, *grid), device=device) for _ in range(self.ndims)]
        
        for idxs in itertools.product(*self.rng):
            idxs = list(idxs)
            self.ctr_A[idxs+idxs] = 1
            for i in range(self.ndims):
                prev_idxs = deepcopy(idxs)
                prev_idxs[i] = (prev_idxs[i] - 1) % grid[i]
                self.prev_A[i][idxs+prev_idxs] = 1
                next_idxs = deepcopy(idxs)
                next_idxs[i] = (next_idxs[i] + 1) % grid[i]
                self.next_A[i][idxs+next_idxs] = 1

        for i in range(self.ndims):
            self.prev[i] = [list(idxs) if isinstance(idxs, np.ndarray) else idxs for idxs in self.prev[i]]
            self.next[i] = [list(idxs) if isinstance(idxs, np.ndarray) else idxs for idxs in self.next[i]]

    def _test(self, u:np.ndarray):
        # Sanity check that makes sure implicit and explicit schemes are consistent
        # fo an arbitrary input u
        assert u.shape == self.grid, 'u shape must match the grid shape'

        u = torch.as_tensor(u, device=self.device)

        assert np.all(
            self.ctr_A.reshape(self.n, self.n) @ u.flatten() == u[self.ctr].flatten()
        ), 'Failed for the center cell'

        for i in range(self.ndims):
            assert torch.all(
                self.prev_A[i].reshape(self.n, self.n) @ u.flatten() == u[self.prev[i]].flatten()
            ), f'Failed for previous cell in dimension {i}'
            assert torch.all(
                self.next_A[i].reshape(self.n, self.n) @ u.flatten() == u[self.next[i]].flatten()
            ), f'Failed for next cell in dimension {i}'

        print('Test passed')
            
    def config(self):
        # needs to be defined in child class
        # has to define self.A and self.dt_solve and other stuff
        raise NotImplementedError

    def explicit_stability_check(self):
        raise NotImplementedError
    
    def explicit_step(self, u:torch.Tensor) ->torch.Tensor:
        raise NotImplementedError
    
    def implicit_step(self, u:torch.Tensor) ->torch.Tensor: # the matrix A should be defined in child class.
        return torch.linalg.solve(self.A.reshape(self.n, self.n), u.flatten()).reshape(*u.shape)
    
    def config_time(self, T=1.0, dt_save=0.05):
        dt_solve = self.dt_solve
        nt_solve = round(T/dt_solve)
        save_freq = round(dt_save/dt_solve)
        assert nt_solve*dt_solve == T, f'T={T} must be a multiple of dt_solve={dt_solve}'
        assert save_freq*dt_solve == dt_save, f'dt_save={dt_save} must be a multiple of dt_solve={dt_solve}'
        self.nt_solve = nt_solve
        self.save_freq = save_freq
        self.T = T
        self.dt_save = dt_save

    def solve(self, u0:np.ndarray, explicit=False) ->np.ndarray:

        if explicit:
            self.explicit_stability_check() # needs to be defined in child class
            stepper = self.explicit_step
        else:
            stepper = self.implicit_step

        output = [u0.copy()]
        u = torch.as_tensor(u0.copy(), device=self.device)
        assert tuple(u.shape) == self.grid, 'u0 shape must match the grid shape'

        timer = tqdm(range(1, self.nt_solve+1), leave=False)
        for i in timer:
            u = stepper(u)
            if i % self.save_freq == 0:
                output.append(u.clone().cpu().numpy())

        return np.stack(output)
    

class Diffusion_Euler_Solver_Periodic(Euler_Solver_Periodic):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def config(
        self,
        nu = [0.1], # diffusion coefficient in each dimension
        dx = [2**-10], # spatial grid size in each dimension
        Fo = [0.4], # Fourier number in each dimension
        dt_solve = None,
        ):
        assert len(nu) == len(dx) == self.ndims
        self.nu = nu
        self.dx = dx
        if dt_solve is None:
            dt_solve = np.min([Fo[i]*dx[i]**2/nu[i] for i in range(self.ndims)])
        Fo = [nu[i]*dt_solve/dx[i]**2 for i in range(self.ndims)]    
        self.Fo = Fo
        self.dt_solve = dt_solve

        self.A = self.ctr_A.clone()
        for i in range(self.ndims):
            self.A -= self.Fo[i]*(self.next_A[i] - 2*self.ctr_A + self.prev_A[i])

    def explicit_stability_check(self):
        assert sum(self.Fo) < 0.5, f'Fourier condition not satisfied, it is {self.Fo}'

    def explicit_step(self, u_prev:torch.Tensor) ->torch.Tensor:
        u = u_prev.clone()
        for i in range(self.ndims):
            u += self.Fo[i]*(u[self.next[i]] - 2*u + u[self.prev[i]])

        return u


def save_ICs():
    IC_1d_path = PDEBench_data_dir + '/1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5'

    with h5py.File(IC_1d_path, 'r') as f:
        x_mesh = f['x-coordinate'][:] # (1024,)
        u0_1d = f['tensor'][:, 0] # (10000, 1024)

    IC_1d_path_new = data_dir + '/IC_1d_from_Adv0.1.hdf5'

    with h5py.File(IC_1d_path_new, 'w') as f:
        f.create_dataset(name='x-coordinate', data=x_mesh, dtype=np.float32)
        f.create_dataset(name='u0', data=u0_1d, dtype=np.float32)

    IC_2d_path = PDEBench_data_dir + '/2D/CFD/2D_Train_Rand/2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5'

    with h5py.File(IC_2d_path, 'r') as f:
        x_mesh_2d = f['x-coordinate'][:] # (128,)
        u0_2d = f['Vx'][:, 0]*3 # (10000, 128, 128)    
    
    IC_2d_path_new = data_dir + '/IC_2d_from_CFD0.1_Vx_tripled.hdf5'

    with h5py.File(IC_2d_path_new, 'w') as f:
        f.create_dataset(name='x-coordinate', data=x_mesh_2d, dtype=np.float32)
        f.create_dataset(name='u0', data=u0_2d, dtype=np.float32)


def exact_2d_adv_solver(x, y, t, u0, beta):
## Using analytical solution for 2D advection equation
# Assuming same beta in x and y directions
    X, Y = np.meshgrid(x, y, indexing='ij')
    interpolator = RegularGridInterpolator((x, y), u0, bounds_error=False, fill_value=None)
    u = np.zeros((len(t), len(x), len(y)), dtype=np.float32)
    for i in range(len(t)):
        # Shifted coordinates
        x_shifted = (X - beta * t[i]) % 1.0
        y_shifted = (Y - beta * t[i]) % 1.0
        shifted_coords = np.array([x_shifted.flatten(), y_shifted.flatten()]).T
        u[i] = interpolator(shifted_coords).reshape(X.shape)
    return u


def generate_2d_advection_datasets(beta):
    # 0.1, 0.4, 1.0, 4.0
    IC_2d_path = data_dir + '/IC_2d_from_CFD0.1_Vx_tripled.hdf5'

    with h5py.File(IC_2d_path, 'r') as f:
        x = f['x-coordinate'][::2] # (64,)
        y = f['x-coordinate'][::2] # (64,)
        t = np.linspace(0.0, 1.0, 21) # (21,)
        u0_2d = f['u0'][:, ::2, ::2] # (10000, 64, 64)

    n_samples = len(u0_2d)
    print(f'Generating 2D_Advection_Beta{beta}')
    file_path = data_dir + f'/2D_Advection_Beta{beta}.hdf5'
    f = h5py.File(file_path, 'w')
    f.create_dataset(name='x-coordinate', data=x, dtype=np.float32)
    f.create_dataset(name='y-coordinate', data=y, dtype=np.float32)
    f.create_dataset(name='t-coordinate', data=t, dtype=np.float32)
    u = f.create_dataset(name='u', shape=(n_samples, 21, 64, 64), dtype=np.float32)
    for j in tqdm(range(n_samples)):
        u[j] = exact_2d_adv_solver(x, y, t, u0_2d[j], beta)
    f.close()


def generate_1d_diffusion_datasets(nu):
    # 0.001, 0.002, 0.004, 0.008
    IC_1d_path = data_dir + '/IC_1d_from_Adv0.1.hdf5'

    with h5py.File(IC_1d_path, 'r') as f:
        x_mesh = f['x-coordinate'][:] # (1024,)
        u0_1d = f['u0'][:] # (10000, 1024)
        assert x_mesh.shape[0] == u0_1d.shape[1]
        dx = x_mesh[1] - x_mesh[0]

    diffusion_1d_solver = Diffusion_Euler_Solver_Periodic(grid=(1024,))
    
    n_samples = len(u0_1d)
    print(f'Generating 1D_Diffusion_Nu{nu}')
    diffusion_1d_solver.config(nu=[nu], dx=[dx], dt_solve=0.01)
    diffusion_1d_solver.config_time(T=1.0, dt_save=0.05)

    file_name_1d = f'1D_Diffusion_Nu{nu}.hdf5'
    file_path_1d = os.path.join(data_dir, file_name_1d)
    with h5py.File(file_path_1d, 'w') as f:

        f.create_dataset(name='t-coordinate', data=np.linspace(0.0, 1.0, 21), dtype=np.float32)
        f.create_dataset(name='x-coordinate', data=x_mesh, dtype=np.float32)
        u = f.create_dataset(name='u', shape=(n_samples, 21, 1024), dtype=np.float32)

        for i in tqdm(range(n_samples)):
            u[i] = diffusion_1d_solver.solve(u0_1d[i])


def generate_2d_diffusion_datasets(nu):
    # 0.001, 0.002, 0.004, 0.008

    IC_2d_path = data_dir + '/IC_2d_from_CFD0.1_Vx_tripled.hdf5'

    with h5py.File(IC_2d_path, 'r') as f:
        x_mesh_2d = f['x-coordinate'][::2] # (64,)
        u0_2d = f['u0'][:, ::2, ::2] # (10000, 64, 64)
        assert x_mesh_2d.shape[0] == u0_2d.shape[1] == u0_2d.shape[2]
        dx = x_mesh_2d[1] - x_mesh_2d[0]

    diffusion_2d_solver = Diffusion_Euler_Solver_Periodic(grid=(64,64))

    n_samples = len(u0_2d)
    print(f'Generating 2D_Diffusion_Nu{nu}')
    diffusion_2d_solver.config(nu=[nu, nu], dx=[dx, dx], dt_solve=0.01)
    diffusion_2d_solver.config_time(T=1.0, dt_save=0.05)

    file_name_2d = f'2D_Diffusion_Nu{nu}.hdf5'
    file_path_2d = os.path.join(data_dir, file_name_2d)
    with h5py.File(file_path_2d, 'w') as f:

        f.create_dataset(name='t-coordinate', data=np.linspace(0.0, 1.0, 21), dtype=np.float32)
        f.create_dataset(name='x-coordinate', data=x_mesh_2d, dtype=np.float32)
        f.create_dataset(name='y-coordinate', data=x_mesh_2d, dtype=np.float32)
        u = f.create_dataset(name='u', shape=(n_samples, 21, 64, 64), dtype=np.float32)

        for i in tqdm(range(n_samples)):
            u[i] = diffusion_2d_solver.solve(u0_2d[i])


if __name__ == '__main__':
    # create argument parser with args -adv2d beta1 beta2 ... -diff1d nu1 nu2 ... -diff2d nu1 nu2 ...

    parser = argparse.ArgumentParser(description='Generate new datasets for FPNO')
    parser.add_argument('-adv2d', nargs='+', type=float, help='Beta values for 2D advection')
    parser.add_argument('-diff1d', nargs='+', type=float, help='Nu values for 1D diffusion')
    parser.add_argument('-diff2d', nargs='+', type=float, help='Nu values for 2D diffusion')
    args = parser.parse_args()

    # save_ICs()

    if args.adv2d:
        for beta in args.adv2d:
            generate_2d_advection_datasets(beta)

    if args.diff1d:
        for nu in args.diff1d:
            generate_1d_diffusion_datasets(nu)

    if args.diff2d:
        for nu in args.diff2d:
            generate_2d_diffusion_datasets(nu)
