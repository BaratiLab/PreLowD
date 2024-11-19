import os
import pandas as pd
import argparse
from plotting import transfer_namer, plot_experiments, log_tick_formatter, compare_FFNOs
import torch

"""
This code merges the dataframes from the experiments that were done across several clusters.
Then, we get the final plots for the paper.
"""

# advections:
def merge_result_tables():

    for beta in [0.1, 0.4, 1.0]:
        result_dir = f'results/2D_Advection_Beta{beta}_merged'
        os.makedirs(result_dir+'/processed', exist_ok=True)
        merged_df = pd.DataFrame()
        for seed in [0, 1, 2]:
            seed_df = pd.read_excel(f'results/2D_Advection_Beta{beta}_seed{seed}/table.xlsx')
            merged_df = pd.concat([merged_df, seed_df])
        merged_df.to_excel(result_dir+'/table.xlsx', index=False)

    # diffusions:
    for nu in [0.001, 0.002, 0.004]:
        result_dir = f'results/2D_Diffusion_Nu{nu}_merged'
        os.makedirs(result_dir+'/processed', exist_ok=True)
        merged_df = pd.DataFrame()
        for seed in [0, 1, 2]:
            seed_df = pd.read_excel(f'results/2D_Diffusion_Nu{nu}_seed{seed}/table.xlsx')
            merged_df = pd.concat([merged_df, seed_df])
        merged_df.to_excel(result_dir+'/table.xlsx', index=False)

def make_plot_for_result_subset(result_df, result_dir, group_name, y_keywords):
    y_vars = [col for col in result_df.columns if all([kw in col.lower() for kw in y_keywords])]
    plot_experiments(
        result_df = result_df,
        result_dir = result_dir,

        group_params = ['width', 'fourier_modes'],

        group_namer = lambda x: group_name,
        subgroup_namer = transfer_namer,

        # you can customize these two, but not recommended
        result_keywords = ['loss', 'time'],
        random_keywords = ['seed'],

        varying_x_vars_only = False,
        compact_group_name = False,
        compact_subgroup_name = True,
        
        all_y_in_one = True,

        # CUSTOMIZE these
        x_vars = ['frac_train_data'],
        x_labels = ['#samples'],
        x_scales = ['log'],
        x_tickers = [log_tick_formatter],

        # CUSTOMIZE these
        y_vars = y_vars,
        y_labels = None,
        y_scales = None,
        y_tickers = None,

        save_dir = None, # change if you wanna save in default place or not
        save = True, # change if you wanna save
        show = False # change if wana show
    )


def make_final_plots():

    final_table_tex = ""

    # diffusion
    for nu in [0.001, 0.002, 0.004]:
        result_dir = f'results/2D_Diffusion_Nu{nu}_merged'
        result_df = pd.read_excel(result_dir + '/table.xlsx')
        make_plot_for_result_subset(result_df, result_dir, group_name=rf'Diffusion_Nu{nu}', y_keywords=['val_loss', 'it5000'])

        paper_table_name = '_'.join(result_dir.split('_')[1:3])+'_paper_table.tex'
        with open(result_dir + f'/processed/{paper_table_name}', 'r') as f:
            final_table_tex += f.read()
            final_table_tex += '\n'

    # advection
    for beta in [0.1, 0.4, 1.0]:
        result_dir = f'results/2D_Advection_Beta{beta}_merged'
        result_df = pd.read_excel(result_dir + '/table.xlsx')
        make_plot_for_result_subset(result_df, result_dir, group_name=rf'Advection_Beta{beta}', y_keywords=['val_loss', 'it5000'])

        paper_table_name = '_'.join(result_dir.split('_')[1:3])+'_paper_table.tex'
        with open(result_dir + f'/processed/{paper_table_name}', 'r') as f:
            final_table_tex += f.read()
            final_table_tex += '\n'

    with open('results/final_tables.tex', 'w') as f:
        f.write(final_table_tex)
        

def make_model_comparison_plots():

    # advection
    for beta in [0.1, 0.4, 1.0]:
        for seed in [0, 1, 2]:
            result_dir = f'results/2D_Advection_Beta{beta}_seed{seed}'
            result_df = pd.read_excel(result_dir + '/table.xlsx').fillna('')

            # iterating through the rows, if there is a 'transfer_from' argument, compare that with the model corresponding to the row
            for i, row in result_df.iterrows():
                print(f'Advection_Beta{beta}_seed{seed}, {i}/{len(result_df)}')
                pretrained_model = row['transfer_from']
                if not pretrained_model:
                    continue

                tuned_model = result_dir + f'/models/model_{i:05d}.pt'
                
                model_1 = torch.load(pretrained_model, map_location='cpu')
                model_2 = torch.load(tuned_model, map_location='cpu')

                # compare the models
                compare_FFNOs(model_1, model_2, show=False, save=result_dir+f'/figures/model_{i:05d}_{transfer_namer(row)}')

    # diffusion

    for nu in [0.001, 0.002, 0.004]:
        for seed in [0, 1, 2]:
            result_dir = f'results/2D_Diffusion_Nu{nu}_seed{seed}'
            result_df = pd.read_excel(result_dir + '/table.xlsx').fillna('')

            # iterating through the rows, if there is a 'transfer_from' argument, compare that with the model corresponding to the row
            for i, row in result_df.iterrows():
                print(f'Diffusion_Nu{nu}_seed{seed}, {i}/{len(result_df)}')
                pretrained_model = row['transfer_from']
                if not pretrained_model:
                    continue

                tuned_model = result_dir + f'/models/model_{i:05d}.pt'

                model_1 = torch.load(pretrained_model, map_location='cpu')
                model_2 = torch.load(tuned_model, map_location='cpu')

                # compare the models
                compare_FFNOs(model_1, model_2, show=False, save=result_dir+f'/figures/model_{i:05d}_{transfer_namer(row)}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--exp_plots', action='store_true')
    parser.add_argument('--compare_plots', action='store_true')
    args = parser.parse_args()

    if args.merge:
        merge_result_tables()
    
    if args.exp_plots:
        make_final_plots()
    
    if args.compare_plots:
        make_model_comparison_plots()
