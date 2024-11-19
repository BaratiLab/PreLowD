import os
import itertools
from Datasets import PDEDataset
from utils_data import tqdm_files

# mail-02
PDEBench_data_dir = '/media/pouya/DATA1/PDE_data/PDEBench'
data_dir = '/media/pouya/DATA1/PDE_data/FPNO_data'

# MAIL-25
# PDEBench_small_data_dir = '/home/pouya/FPNO_data'

os.makedirs(data_dir, exist_ok=True)

# 1D Advection #########################################################################

# original data:

Adv_1D_train_dir = PDEBench_data_dir + '/1D/Advection/Train'

Adv_1D_train_files = [
    Adv_1D_train_dir + f'/1D_Advection_Sols_beta{beta}.hdf5'
    for beta in [0.1, 0.4, 1.0, 4.0]
    ]

# our data:

small_Adv_1D_files = [
    data_dir + f'/1D_Advection_Beta{beta}.hdf5'
    for beta in [0.1, 0.4, 1.0, 4.0]
    ]

def preprocess_1d_adv():
    print('Preprocessing 1D Advection ...')
    for i, file_path in enumerate(tqdm_files(Adv_1D_train_files)):
        dataset = PDEDataset(file_path, t_start=0, t_end=100, dt=5, load_now=False)
        dataset.save(small_Adv_1D_files[i])
        del dataset


if __name__ == '__main__':
    preprocess_1d_adv()
