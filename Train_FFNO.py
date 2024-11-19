import os
import yaml
import itertools
import pickle
from collections import OrderedDict
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import torch
from torch import optim

from config_parser import parse_args, save_config
from Datasets import PDEDataset
from utils_train import set_seed, parse_csv, Relative_Lp_Loss, train_iters
from Models import FFNO
from plotting import plot_experiments, plot_training, model_namer, transfer_namer


os.makedirs('./results', exist_ok=True)
Device = 'cuda' if torch.cuda.is_available() else 'cpu'


def is_result_or_seed_col(col_name): # YOU CAN CUSTOMIZE THIS
    """
    used to exclude columns in the results dataframe when grouping
    and indexing the columns based on the parameters of the experiment

    Any columns that are not results or seeds should be ignored when
    dealing with actual experiment parameters
    """
    col = col_name.lower()
    return ('loss' in col) or ('time' in col) or ('seed' in col)


def save_table_to_excel(result_dir, result_table):
    
    result_table.to_excel(result_dir+'/table.xlsx', index=False)

    indexes = [
        col for col in result_table.columns if not is_result_or_seed_col(col)
    ]
    if len(indexes) > 0:
        result_table_indexed = result_table.set_index(indexes)
        result_table_indexed.to_excel(result_dir+'/table_indexed.xlsx')

    non_varying_cols = [
        col for col in result_table.columns 
        if result_table[col].nunique() <= 1 
        and not is_result_or_seed_col(col)
    ]

    if len(non_varying_cols) > 0:
        result_table_trimmed = result_table.drop(columns=non_varying_cols)
        result_table_trimmed.to_excel(result_dir+'/table_trimmed.xlsx', index=False)
    
        trimmed_indexes = [
            col for col in result_table_trimmed.columns
            if not is_result_or_seed_col(col)
        ]
        if len(trimmed_indexes) > 0:
            result_table_indexed_trimmed = result_table_trimmed.set_index(trimmed_indexes)
            result_table_indexed_trimmed.to_excel(result_dir+'/table_trimmed_indexed.xlsx')


def load_data(args):
    data_train_loading_config = {
        'data_path': os.path.join(args.data_dir, args.data_train),
        't_start': args.data_train_t_start,
        't_end': args.data_train_t_end,
        'dt': args.data_train_dt,
        'rx': args.data_train_rx,
        'verbose': True,
        }

    data_train = PDEDataset(**data_train_loading_config)

    if args.data_val not in [args.data_train, None, '']:
        data_val_loading_config = {
            'data_path': os.path.join(args.data_dir, args.data_val),
            't_start': args.data_val_t_start,
            't_end': args.data_val_t_end,
            'dt': args.data_val_dt,
            'rx': args.data_val_rx
            }   
        data_val = PDEDataset(**data_val_loading_config)
        val_separate = True
    else:
        data_val = data_train
        val_separate = False
    
    return data_train, data_val, val_separate


def set_up_data(data_train, data_val, args):

    # autoregressive configuration
    data_config_autoregression = {
        'dt': args.dt,
        'in_snapshots': args.in_snapshots,
        'out_snapshots': args.out_snapshots,
        'skip': args.skip,
        'rollout': args.rollout,
        }

    data_train.config_autoregression(**data_config_autoregression)
    data_val.config_autoregression(**data_config_autoregression)


def get_data_config_functions(args, val_separate):

    def config_train_data_for_training(dataset):
        dataset.config(
            subset = 0.0 if val_separate else args.val_size,
            where = args.val_mode,
            reverse = True,
            seed = args.val_seed,
            frac = args.frac_train_data,
            frac_seed = args.seed
            )
    
    def config_train_data_for_validation(dataset):
        dataset.config(
            subset = 0.0 if val_separate else args.val_size,
            where = args.val_mode,
            reverse = True,
            seed = args.val_seed,
            frac = 1.0,
            frac_seed = args.seed
        )
        
    def config_val_data_for_validation(dataset):
        dataset.config(
            subset = 0.0 if val_separate else args.val_size,
            where = args.val_mode,
            reverse = False,
            seed = args.val_seed,
            frac = 1.0,
            frac_seed = args.seed
            )
        
    return [
        config_train_data_for_training, 
        config_train_data_for_validation, 
        config_val_data_for_validation
    ]


def train_ffno(
        ffno,
        args,
        data_train,
        data_val,
        config_train_data_for_training,
        config_train_data_for_validation,
        config_val_data_for_validation,
        pretrained_ffno = ''
        ):
    
    set_up_data(data_train, data_val, args)

    if pretrained_ffno:
        ffno.transfer_from(
            source_ffno = pretrained_ffno,
            
            transfer_in_vars = args.transfer_in_vars,
            transfer_in_vars_scalers = args.transfer_in_vars_scalers,

            transfer_out_vars = args.transfer_out_vars,
            transfer_out_vars_scalers = args.transfer_out_vars_scalers,

            transfer_fourier_layers = args.transfer_fourier_layers,
            transfer_fourier_scalers = args.transfer_fourier_scalers,
            transfer_ff_layers = args.transfer_ff_layers,
        )
        ffno.set_trainability(trainable=False)
        ffno.set_trainability(
            in_vars = args.tune_in_vars,
            out_vars = args.tune_out_vars,

            fourier_layers = args.tune_fourier_layers,
            ff_layers = args.tune_ff_layers,
            trainable = True
        )

    training_result = train_iters(
            model = ffno,

            train_dataset = data_train,
            config_train_data_for_training = config_train_data_for_training,
            config_train_data_for_validation = config_train_data_for_validation,

            val_dataset = data_val,
            config_val_data_for_validation = config_val_data_for_validation,

            optimizer = getattr(optim, args.optimizer),
            loss_fn = Relative_Lp_Loss,
            loss_reduction = 'mean',
            iters = args.iters,
            batch_size = args.batch_size,

            val_iters = args.val_iters,
            val_rollouts = args.val_rollouts,
        )
    
    return ffno, training_result


def is_valid_model(
        args
        ) -> bool:
    """
    This is a hard-coded function. Each user should modify it based on their needs.
    """
    # 1D already shares fourier weights
    if args.data_ndims == 1 and not args.share_fourier:
        return False
        
    return True


def is_valid_transfer(
        args,
        ffno: FFNO,
        ) -> bool:
    """
    This is a hard-coded function. Each user should modify it based on their needs.
    """
    transfer_parser = lambda x: list(x.split('>')) if '>' in x else [x, x]
    transfer_in_vars = parse_csv(args.transfer_in_vars, full=ffno.projector.in_vars, func=transfer_parser)
    transfer_in_vars = [var_pair[1] if isinstance(var_pair, list) else var_pair for var_pair in transfer_in_vars]
    transfer_out_vars = parse_csv(args.transfer_out_vars, full=ffno.projector.out_vars, func=transfer_parser)
    transfer_out_vars = [var_pair[1] if isinstance(var_pair, list) else var_pair for var_pair in transfer_out_vars]
    tune_in_vars = parse_csv(args.tune_in_vars, full=ffno.projector.in_vars)
    tune_out_vars = parse_csv(args.tune_out_vars, full=ffno.projector.out_vars)

    all_layers = list(range(ffno.n_layers))
    layer_indexer = lambda x: int(x)%ffno.n_layers
    transfer_fourier_layers = parse_csv(args.transfer_fourier_layers, full=all_layers, func=layer_indexer)
    transfer_ff_layers = parse_csv(args.transfer_ff_layers, full=all_layers, func=layer_indexer)
    tune_fourier_layers = parse_csv(args.tune_fourier_layers, full=all_layers, func=layer_indexer)
    tune_ff_layers = parse_csv(args.tune_ff_layers, full=all_layers, func=layer_indexer)

    # For now we consider freezing both or neither
    if set(tune_in_vars) != set(tune_out_vars):
        return False
    
    # we have to transfer something
    if not set(transfer_fourier_layers + transfer_ff_layers):
        return False

    # we have to train something
    if not set(tune_fourier_layers+tune_ff_layers):
        return False
    
    # for small scale tuning, we keep it single or no layer tuning for both components
    if len(tune_fourier_layers)==1:

        if len(tune_ff_layers)>1:
            return False
        
        if all_layers[-1] in tune_fourier_layers and all_layers[-1] not in tune_ff_layers:
            return False
        
    if len(tune_ff_layers)==1:

        if len(tune_fourier_layers)>1:
            return False     
           
        if all_layers[0] in tune_ff_layers and all_layers[0] not in tune_fourier_layers:
            return False
        
    # if not everything was transferred, we onlt tune none or all
    if not (transfer_fourier_layers == transfer_ff_layers == all_layers):
        if tune_ff_layers not in [[], all_layers]:
            return False
        if tune_fourier_layers not in [[], all_layers]:
            return False

    return True


def get_exp_combinations(exp_configs_yaml):
    if not exp_configs_yaml:
        return [], []

    with open(exp_configs_yaml) as file:
        configs = OrderedDict(yaml.safe_load(file))

    hyperparams = list(configs.keys())
    combs = [dict(zip(configs.keys(), comb)) for comb in itertools.product(*configs.values())]

    return hyperparams, combs


def find_matching_pretrained_ffno(
        pretrained_ffno_dir,
        model_args
        ):
    pretrained_dir_result_table = pd.read_excel(pretrained_ffno_dir+'/table.xlsx')
    # find a row that its columns matches the content of the dict of vars(model_args)

    matching_args = {
        arg: vars(model_args)[arg]
        for arg in [
            'in_snapshots', 
            'out_snapshots', 
            'dt', 
            'skip', 
            'n_layers',
            'ffno_width', 
            'fourier_modes', 
            'seed'
            ]
    }

    matching_row = pretrained_dir_result_table[
        (pretrained_dir_result_table[matching_args.keys()] == pd.Series(matching_args)).all(axis=1)
    ]
    if len(matching_row) == 0:
        raise ValueError('No matching pretrained model found!')
    return pretrained_ffno_dir + f'/models/model_{matching_row.index[0]:05d}.pt'


def train_and_save_result(
        args,
        data_train, 
        data_val, 
        val_separate, 

        idx, 
        result_dir,
        result_table,
        pretrained_ffno_path = '',

        log_plots = True
        ):
    set_seed(args.seed)

    ffno = FFNO(
        in_vars = data_train.Vars.keys(),
        out_vars = data_train.Vars.keys(),
        in_dim = args.in_snapshots,
        out_dim = args.out_snapshots,

        n_layers = args.n_layers,
        width = args.ffno_width,
        fourier_modes = args.data_ndims*[args.fourier_modes],
        share_fourier = args.share_fourier,
        
        device = Device
        )

    ffno, training_result = train_ffno(
        ffno, 
        args, 
        data_train,
        data_val,
        *get_data_config_functions(args, val_separate),
        pretrained_ffno = torch.load(pretrained_ffno_path, map_location=Device) if pretrained_ffno_path else None    
    )

    # saving result:
    vars(args)['batch_train_time'] = training_result['batch_train_time']
    
    y_vars = []
    for r in parse_csv(args.val_rollouts, full=[1], func=int):

        vars(args)[f'batch_val_time_r{r}'] = training_result['batch_val_time'][f'r{r}']

        for val_iter in parse_csv(args.val_iters, full=[args.iters], func=int):
            vars(args)[f'train_loss_r{r}_it{val_iter}'] = training_result['val_loss_train_data'][f'r{r}'][f'it{val_iter}'].mean()
            vars(args)[f'val_loss_r{r}_it{val_iter}'] = training_result['val_loss_val_data'][f'r{r}'][f'it{val_iter}'].mean()
            y_vars.append(f'train_loss_r{r}_it{val_iter}')
            y_vars.append(f'val_loss_r{r}_it{val_iter}')

    
    for key in vars(args):
        # adding the column if it does not exist
        if key not in result_table.columns and ('time' in key or 'loss' in key):
            result_table[key] = ''

    result_table.loc[idx] = vars(args)
    save_table_to_excel(result_dir, result_table)
    
    torch.save(ffno, result_dir+f'/models/model_{idx:05d}.pt')

    with open(result_dir+f'/trainings/training_{idx:05d}.pkl', 'wb') as file:
        pickle.dump(training_result, file)

    idx += 1

    if log_plots:

        def frac_tick_formatter(val, pos=None):
            N_train = (1-args.val_size)*data_train.N
            return f'$\\frac{{{round(val*N_train)}}}{{{round(N_train)}}}$'

        plot_experiments(
            result_df = result_table,
            result_dir = result_dir,

            group_params = ['n_layers', 'ffno_width', 'fourier_modes'],

            group_namer = model_namer,
            subgroup_namer = transfer_namer,

            result_keywords = ['loss', 'time'],
            random_keywords = ['seed'],

            varying_x_vars_only = True,
            compact_group_name = True,
            compact_subgroup_name = False, 

            all_y_in_one = False, # too many for one figure here.
 
            x_vars = ['frac_train_data'],
            x_labels = ['downstram samples'],
            x_scales = ['log'],
            x_tickers = [frac_tick_formatter],
            
            y_vars = None,
            y_label = None,
            y_scales = None,
            y_tickers = None,

            save_dir = None,
            save = True,
            show = False
        )

        plot_training(
            result_df = result_table,
            result_dir = result_dir,

            group_params = None,
            result_keywords = ['loss', 'time'],

            group_namer = model_namer,
            subgroup_namer = transfer_namer,
            
            compact_group_name = True,
            compact_subgroup_name = False,
            
            smooth_window = 10,
            zoom_size = 0.1,
            save_dir = None,
            save = True,
            show = False
        )

    return result_table, idx, ffno


def main():
    main_args, default_config = parse_args(base_config_path='./configs_FFNO/_base.yaml')

    # Creating a folder for the results
    result_dir = './results/' + main_args.name

    if os.path.exists(result_dir) and len(os.listdir(result_dir)) > 0:
        print(f'Folder {result_dir} already exists and is not empty! Do you want to overwrite it?')
        if input('y/n: ') != 'y':
            return
        
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir+'/figures', exist_ok=True)
    os.makedirs(result_dir+'/models', exist_ok=True)
    os.makedirs(result_dir+'/trainings', exist_ok=True)

    save_config(default_config, result_dir+'/default_config.yaml')

    model_hyperparams, model_combs = get_exp_combinations(main_args.model_configs)

    transfer_hyperparams, transfer_combs = get_exp_combinations(main_args.transfer_configs)

    result_table = pd.DataFrame(columns = 
        model_hyperparams + ['transfer_from'] + transfer_hyperparams
        )

    data_train, data_val, val_separate = load_data(main_args)

    idx = 0
    print(f'Starting experiments for {main_args.name}')

    def descriptor(some_args, varying_params):
        list_of_hyperparams = []
        for param in varying_params:
            current_value = vars(some_args)[param]
            param_str = f'{param}:' +''.join([f'[{value}]' if value==current_value else f'({value})' for value in varying_params[param]])
            list_of_hyperparams.append(param_str)
        
        return ' | '.join(list_of_hyperparams)

    all_model_args = []
    varying_model_params = dict()
    for model_comb in model_combs:
        model_args = deepcopy(main_args)
        vars(model_args).update(model_comb)
        model_args.transfer_from = ''
        if is_valid_model(model_args):
            all_model_args.append(model_args)
            for k, v in model_comb.items():
                if k not in varying_model_params:
                    varying_model_params[k] = [v]
                elif v not in varying_model_params[k]:
                    varying_model_params[k].append(v)


    varying_model_params = {k: v for k, v in varying_model_params.items() if len(v) > 1}

    model_args_pbar = tqdm(all_model_args, leave=True)
    for model_args in model_args_pbar:
        model_args_pbar.set_description('model | ' + descriptor(model_args, varying_model_params)+' ')
        
        # training without transfer learning:
        result_table, idx, ffno = train_and_save_result(
            args = model_args,
            data_train = data_train,
            data_val = data_val,
            val_separate = val_separate,

            idx = idx,
            result_dir = result_dir,
            result_table = result_table,
            pretrained_ffno_path = '',

            log_plots = False
            )

        if not main_args.transfer_from:
            continue

        pretrained_ffno_match_path = find_matching_pretrained_ffno(
            main_args.transfer_from,
            model_args
        )

        # transfer learning experiments:

        model_args.transfer_from = pretrained_ffno_match_path

        all_transfer_args = []
        varying_transfer_params = dict()
        for transfer_comb in transfer_combs:
            transfer_args = deepcopy(model_args)
            vars(transfer_args).update(transfer_comb)
            if is_valid_transfer(transfer_args, ffno):
                all_transfer_args.append(transfer_args)
                for k, v in transfer_comb.items():
                    if k not in varying_transfer_params:
                        varying_transfer_params[k] = [v]
                    elif v not in varying_transfer_params[k]:
                        varying_transfer_params[k].append(v)


        varying_transfer_params = {k: v for k, v in varying_transfer_params.items() if len(v) > 1}

        transfer_args_pbar = tqdm(all_transfer_args, leave=False)
        for i, transfer_args in enumerate(transfer_args_pbar):
            transfer_args_pbar.set_description('transfer | ' + descriptor(transfer_args, varying_transfer_params)+' ')

            # training with transfer learning:
            result_table, idx, ffno = train_and_save_result(
                args = transfer_args,
                data_train = data_train,
                data_val = data_val,
                val_separate = val_separate,

                idx = idx,
                result_dir = result_dir,
                result_table = result_table,
                pretrained_ffno_path = pretrained_ffno_match_path,
                
                log_plots = False # True if i == len(all_transfer_args)-1 else False,
                )
            
    # one last save to make sure everything is up to date
    save_table_to_excel(result_dir, result_table)
    print('Finished experiments for ', main_args.name, '\n')

if __name__ == '__main__':
    main()
