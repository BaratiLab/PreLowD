import os

os.makedirs('experiments_FFNO/cluster3', exist_ok=True)
os.makedirs('experiments_FFNO/cluster4', exist_ok=True)
os.makedirs('experiments_FFNO/cluster5', exist_ok=True)

os.makedirs('experiments_FFNO/MAIL10', exist_ok=True)
os.makedirs('experiments_FFNO/MAIL15', exist_ok=True)
os.makedirs('experiments_FFNO/MAIL25', exist_ok=True)

bash_init = '#!/bin/bash\n'

train_command = 'python Train_FFNO.py'

data_dir = '/home/pouya/FPNO_data'

# templates ############################################################################################################

def make_cmd_1d_adv(beta, gpu_index):
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_index} {train_command}'
    cmd += f' --name 1D_Advection_Beta{beta}'
    cmd += f' --data_dir {data_dir}'
    cmd += f' --data_train 1D_Advection_Beta{beta}.hdf5'
    cmd += f' --data_ndims 1'
    cmd += f' --model_configs ./configs_FFNO/pretrain.yaml'
    cmd += '\n'
    return cmd

def make_cmd_2d_adv(beta, gpu_index, seed):
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_index} {train_command}'
    cmd += f' --name 2D_Advection_Beta{beta}_seed{seed}'
    cmd += f' --data_dir {data_dir}'
    cmd += f' --data_train 2D_Advection_Beta{beta}.hdf5'
    cmd += f' --data_ndims 2'
    cmd += f' --model_configs ./configs_FFNO/downstream_seed{seed}.yaml'
    cmd += f' --transfer_from ./results/1D_Advection_Beta{beta}'
    cmd += f' --transfer_configs ./configs_FFNO_transfer/u_1to2.yaml'
    cmd += '\n'
    return cmd

def make_cmd_1d_diff(nu, gpu_index):
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_index} {train_command}'
    cmd += f' --name 1D_Diffusion_Nu{nu}'
    cmd += f' --data_dir {data_dir}'
    cmd += f' --data_train 1D_Diffusion_Nu{nu}.hdf5'
    cmd += f' --data_ndims 1'
    cmd += f' --model_configs ./configs_FFNO/pretrain.yaml'
    cmd += '\n'
    return cmd

def make_cmd_2d_diff(nu, gpu_index, seed):
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_index} {train_command}'
    cmd += f' --name 2D_Diffusion_Nu{nu}_seed{seed}'
    cmd += f' --data_dir {data_dir}'
    cmd += f' --data_train 2D_Diffusion_Nu{nu}.hdf5'
    cmd += f' --data_ndims 2'
    cmd += f' --model_configs ./configs_FFNO/downstream_seed{seed}.yaml'
    cmd += f' --transfer_from ./results/1D_Diffusion_Nu{nu}'
    cmd += f' --transfer_configs ./configs_FFNO_transfer/u_1to2.yaml'
    cmd += '\n'
    return cmd

def make_automated_tmux(pc, n_gpus):
    cmd = bash_init
    for i in range(n_gpus):
        cmd += f"tmux new-session -d -s gpu{i}\n"
        cmd += f"tmux send-keys -t gpu{i} 'conda activate PDE_py3.9_torch1.10' C-m\n"
        cmd += f"tmux send-keys -t gpu{i} 'bash experiments_FFNO/{pc}/run_gpu{i}.sh' C-m\n"
    cmd += f"tmux ls\n"
    return cmd

# cluster 4 ############################################################################################################
pc = 'cluster4' # Adv 0.1 and 0.4

# 1D_Adv_Beta0.1 and 1D_Adv_Beta0.4 on GPU 0:
for i, gpu_index in enumerate([0]):
    cmd = bash_init
    for beta in [0.1, 0.4]:
        cmd += make_cmd_1d_adv(beta, gpu_index)
    with open(f'experiments_FFNO/{pc}/run_1D_Adv_{0.1}_{0.4}.sh', 'w') as file:
        file.write(cmd)

# 2D_Adv_Beta0.1 on GPUs 0, 1, 2:
for i, gpu_index in enumerate([0, 1, 2]):
    beta = 0.1
    cmd = bash_init
    cmd += make_cmd_2d_adv(beta, gpu_index, i)
    with open(f'experiments_FFNO/{pc}/run_gpu{gpu_index}.sh', 'w') as file:
        file.write(cmd)

# 2D_Adv_Beta0.4 on GPUs 3, 4, 5:
for i, gpu_index in enumerate([3, 4, 5]):
    beta = 0.4
    cmd = bash_init
    cmd += make_cmd_2d_adv(beta, gpu_index, i)
    with open(f'experiments_FFNO/{pc}/run_gpu{gpu_index}.sh', 'w') as file:
        file.write(cmd)

# Tmux automation
cmd = make_automated_tmux(pc, 6)
with open(f'experiments_FFNO/{pc}/run_tmux.sh', 'w') as file:
    file.write(cmd)

# cluster 5 ###########################################################################################################
pc = 'cluster5' # Adv 1.0 and Diff 0.004

# 1D_Adv_Beta0.4 and 1D_Diff_Nu0.004 on GPU 0:
for i, gpu_index in enumerate([0]):
    cmd = bash_init
    for beta in [1.0]:
        cmd += make_cmd_1d_adv(beta, gpu_index)
    for nu in [0.004]:
        cmd += make_cmd_1d_diff(nu, gpu_index)
    with open(f'experiments_FFNO/{pc}/run_1D_Adv_{beta}_Diff_{nu}.sh', 'w') as file:
        file.write(cmd)

# 2D_Adv_Beta1.0 on GPUs 0, 1, 2:
for i, gpu_index in enumerate([0, 1, 2]):
    beta = 1.0
    cmd = bash_init
    cmd += make_cmd_2d_adv(beta, gpu_index, i)
    with open(f'experiments_FFNO/{pc}/run_gpu{gpu_index}.sh', 'w') as file:
        file.write(cmd)

# 2D_Diff_Nu0.004 on GPUs 3, 4, 5:
for i, gpu_index in enumerate([3, 4, 5]):
    nu = 0.004
    cmd = bash_init
    cmd += make_cmd_2d_diff(0.004, gpu_index, i)
    with open(f'experiments_FFNO/{pc}/run_gpu{gpu_index}.sh', 'w') as file:
        file.write(cmd)

# Tmux automation
cmd = make_automated_tmux(pc, 6)
with open(f'experiments_FFNO/{pc}/run_tmux.sh', 'w') as file:
    file.write(cmd)

# MAIL-10 ##############################################################################################################
pc = 'MAIL10' # Diff 0.001

# 1D_Diff_Nu0.001 on GPU 0:
for i, gpu_index in enumerate([0]):
    cmd = bash_init
    for nu in [0.001]:
        cmd += make_cmd_1d_diff(nu, gpu_index)
    with open(f'experiments_FFNO/{pc}/run_1D_Diff_{nu}.sh', 'w') as file:
        file.write(cmd)

# 2D_Diff_Nu0.001 on GPUs 0, 1, 2:
for i, gpu_index in enumerate([0, 1, 2]):
    nu = 0.001
    cmd = bash_init
    cmd += make_cmd_2d_diff(0.001, gpu_index, i)
    with open(f'experiments_FFNO/{pc}/run_gpu{gpu_index}.sh', 'w') as file:
        file.write(cmd)

# Tmux automation
cmd = make_automated_tmux(pc, 3)
with open(f'experiments_FFNO/{pc}/run_tmux.sh', 'w') as file:
    file.write(cmd)

# MAIL-15 ##############################################################################################################
pc = 'MAIL15' # Diff 0.002

# 1D_Diff_Nu0.002 on GPU 0:
for i, gpu_index in enumerate([0]):
    cmd = bash_init
    for nu in [0.002]:
        cmd += make_cmd_1d_diff(nu, gpu_index)
    with open(f'experiments_FFNO/{pc}/run_1D_Diff_Nu_{0.002}.sh', 'w') as file:
        file.write(cmd)

# 2D_Diff_Nu0.002 on GPUs 0, 1, 2:
for i, gpu_index in enumerate([0, 1, 2]):
    nu = 0.002
    cmd = bash_init
    cmd += make_cmd_2d_diff(nu, gpu_index, i)
    with open(f'experiments_FFNO/{pc}/run_gpu{gpu_index}.sh', 'w') as file:
        file.write(cmd)

# Tmux automation
cmd = make_automated_tmux(pc, 3)
with open(f'experiments_FFNO/{pc}/run_tmux.sh', 'w') as file:
    file.write(cmd)
