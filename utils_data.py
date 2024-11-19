import os
from tqdm import tqdm
import pandas as pd
import h5py
from Datasets import PDEDataset

# mail-2
data_dir_parent_folder = '/media/pouya/DATA1/PDE_data'

# others
# data_dir_parent_folder = '/home/pouya'

def file_size(file_path, Unit=None):
    if os.path.isfile(file_path):
        # Get size of file in bytes
        file_size = os.path.getsize(file_path)
        # Convert bytes to human-readable format
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if file_size < 1e3 and Unit is None:
                return f"{file_size:.2f} {unit}"
            if unit == Unit:
                return file_size
            file_size /= 1e3
    else:
        return 'Not A File'
    

class tqdm_files(tqdm):
    def __init__(self, file_paths, unit='MB', *args, **kwargs):
        self.file_paths = file_paths
        self.n_files = len(file_paths)
        self.total_size = self._get_total_size()
        super().__init__(total=self.total_size, unit=unit, *args, **kwargs)

    def _get_total_size(self):
        return sum(self._get_file_size(file) for file in self.file_paths)

    def _get_file_size(self, file_path):
        return round(os.path.getsize(file_path)/1e6)  # size in MB

    def __iter__(self):
        for file_number, file_path in enumerate(self.file_paths, start=1):
            file_size = self._get_file_size(file_path)
            self.set_description(f"{file_number}/{self.n_files}) {os.path.basename(file_path)} ({file_size} MB)")
            yield file_path
            self.update(file_size)
    

def print_h5_file_content(path):
    file_type = path.split('.')[-1] 
    if file_type != 'hdf5':
        print('Not an hdf5 file')
        if file_type == 'h5':
            print('But it is an h5 file. Skipping it ...')
        return
    with h5py.File(path, 'r') as f:
        print(50*'-')
        def check_h5_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                if name.endswith('coordinate'):
                    print(name, '| dtype', obj.dtype, '| shape', obj.shape, '| range', f'({obj[0]:.4f}, {obj[-1]:.4f})')
                else:
                    print(name, '| dtype', obj.dtype, '| shape', obj.shape)
        f.visititems(check_h5_item)


def print_dir_content(directory):
    for root, dirs, files in os.walk(directory):
        print(root)
        print(70*'#')
        for dir in dirs:
            print(dir + '/')
            print(50*'=')
        for file in files:
            file_path = os.path.join(root, file)
            print(f'{file} ({file_size(file_path)})')
            print_h5_file_content(file_path)
            print(50*'=')
        print(70*'=')


def extract_data_info(data_dir, name=None):
    df = pd.DataFrame()
    print('Extracting small data info ...')
    h5_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.hdf5')]
    for file in tqdm_files(h5_files):
        dataset = PDEDataset(file, load_now=False)
        for coord in dataset.Coords:
            df = df._append({
                'dataset': os.path.basename(file),
                'var': coord,
                **{info: dataset.Coords[coord][info] for info in dataset.Coords[coord] if info!='data'}
                }, ignore_index=True)
        for var in dataset.Vars:
            df = df._append({
                'dataset': os.path.basename(file),
                'var': var,
                **{info: dataset.Vars[var][info] for info in dataset.Vars[var] if info!='data'}
                }, ignore_index=True)
        del dataset
        
    name = name or data_dir.split('/')[-1]
    df.set_index(['dataset', 'var'], inplace=True)
    df.to_excel(f'{data_dir}/{name}.xlsx')
    df.to_excel(f'{name}.xlsx')


if __name__ == '__main__':
    extract_data_info(data_dir_parent_folder+'/FPNO_data')