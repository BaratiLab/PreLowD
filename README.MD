# Pretraining a Neural Operator in Lower Dimensions

This directory contains the code and documentation for [this article](https://openreview.net/forum?id=ZewaRoZehI), published in TMLR.

## Installation and requirements


1. Clone the repository: 

`git clone https://github.com/BaratiLab/PreLowD.git`

2. create a new environment. Choose the python version according to the system and compatibility of CUDA.

`conda create -n prelowd python=3.9`

`conda activate prelowd`

3. [Install PyTorch](https://pytorch.org/get-started/locally/). The command looks like this with different details according to your device and CUDA version. We used torch 1.10 and python 3.9 because we had CUDA 11.1

`conda install pytorch torchvision torchaudio pytorch-cuda=11.0 -c pytorch -c nvidia`

4. Install the required packages: 

`conda install h5py ipykernel ipywidgets matplotlib numpy pandas PyYAML scipy tqdm`

`pip install einops`

## Usage

Modify ` make_exps_bash.py` according to your available devices and their number of GPUs. Then, run the script to create bash scripts that automate the run of the specified experiments for each GPU and each device. Then, you'll have to collect the results back to a single device to merge and get the final table. You can do tha using `final_results.py`.

You can look at and use `gen_new_data.py` for data generation.

