import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import os
import torch

plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['legend.loc'] = (1.02, 0)
# setting the default font size
plt.rcParams['font.size'] = 12
# setting dpi high
plt.rcParams['savefig.dpi'] = 600


def model_namer(model_dict):
        return ','.join([f'{key}_{value}' for key, value in model_dict.items()])


def transfer_namer(transfer_args): # HARD_CODED FOR MY APPLICATION

    if not transfer_args['transfer_in_vars']:
        label = 'C0(from_scratch)'
    
    elif transfer_args['tune_fourier_layers'] == 'all' and transfer_args['tune_ff_layers'] == 'all':
        label = 'C1(all)'
    
    elif transfer_args['tune_fourier_layers'] == 'all':
        label = 'C2(all_Fouriers)'
    
    elif transfer_args['tune_ff_layers'] == 'all':
        label = 'C3(all_FFs)'

    elif not transfer_args['tune_fourier_layers'] and transfer_args['tune_ff_layers']=='3,':
        label = 'C4(last_FF)'

    elif transfer_args['tune_fourier_layers'] == '0,' and not transfer_args['tune_ff_layers']:
        label = 'C5(first_Fourier)'
    
    elif transfer_args['tune_fourier_layers'] == '0,' and transfer_args['tune_ff_layers']=='0,':
        label = 'C6(first_Fourier_first_FF)'
    
    elif transfer_args['tune_fourier_layers'] == '0,' and transfer_args['tune_ff_layers']=='3,':
        label = 'C7(first_Fourier_last_FF)'
    
    elif transfer_args['tune_fourier_layers'] == '3,' and transfer_args['tune_ff_layers']=='3,':
        label = 'C8(last_Fourier_last_FF)'
    return label[:2]


def sort_legend(ax: plt.Axes):
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels)

# for example
def log_tick_formatter(val, pos=None):
    N_train = 8000
    return round(val*N_train)
    # return f'$\\frac{{{round(val*N_train)}}}{{{round(N_train)}}}$'

def label_cleaner(raw_label):
    # Diffusion_Nu0.001_r1 should be Diffusion \nu=0.001 rollout=1 where \nu is the latex symbol for nu
    label = raw_label.replace('val_loss_', '')
    label = label.replace('_it5000', '')
    label = label.replace('Beta', r', $\beta$=')
    label = label.replace('Nu', r', $\nu$=')
    label = label.replace('r1', 'r=1')
    label = label.replace('r5', 'r=5')
    label = label.replace('_', ' ')
    return label

def plot_experiments(
        result_df : pd.DataFrame, # pandas dataframe
        result_dir : str, # directory where the results are stored

        group_params : list, # iterable of strings

        group_namer = lambda x: '|'.join([f'{k}={v}' for k, v in x.items()]),
        subgroup_namer = lambda x: '|'.join([f'{k}={v}' for k, v in x.items()]),

        result_keywords = ['loss', 'time'],
        random_keywords = ['seed'],

        varying_x_vars_only = True, # makes plots make sense, so we have more than one value for x_var
        compact_group_name = True, # makes group name more compact (only using the varying parameters)
        compact_subgroup_name = False, # makes subgroup name more compact (only using the varying parameters)

        all_y_in_one = True,

        x_vars = None,
        x_labels = None,
        x_scales = None,
        x_tickers = None,

        y_vars = None,
        y_labels = None,
        y_scales = None,
        y_tickers = None,

        save_dir = None, # change if you wanna save in default directory (None) or not
        save = True, # change if you wanna save or not save
        show = False # change if wana show or not show

        ):
    
    if save_dir is None:
        save_dir = result_dir + '/processed'
    
    result_df = result_df.fillna('') # to count NaNs as a unique value
    
    result_cols = [
        col for col in result_df.columns 
        if any([kw in col.lower() for kw in result_keywords])
        ]
    random_cols = [
        col for col in result_df.columns 
        if any([kw in col.lower() for kw in random_keywords])
        ] + ['transfer_from']

    # horizontal axes variables

    if x_vars is None:
        x_vars = []
        for col in result_df.columns:
            if col not in random_cols+result_cols:
                if varying_x_vars_only:
                    if result_df[col].nunique() > 1:
                        x_vars.append(col)
                    else:
                        print(f'{col} is not varying')
                else:
                    x_vars.append(col)
        
    if x_labels is None:
        x_labels = x_vars

    if x_scales is None:
        x_scales = ['linear']*len(x_vars)

    if x_tickers is None:
        x_tickers = [None]*len(x_vars)
        
    # vertical axes variables
    if y_vars is None:
        y_vars = [
            col for col in result_df.columns 
            if col in result_cols
            ]

    if y_labels is None:
        y_labels = y_vars
    
    if y_scales is None:
        y_scales = ['linear']*len(y_vars)

    if y_tickers is None:
        y_tickers = [None]*len(y_vars)

    
    for x_var, x_label, x_scale, x_ticker in zip(x_vars, x_labels, x_scales, x_tickers):

        gain_table = pd.DataFrame(index=result_df[x_var])

        group_cols = [
            col for col in result_df.columns 
            if col in group_params
            and col not in [x_var]+random_cols+result_cols
        ]
        groups = result_df.groupby(group_cols) if group_cols else [((), result_df)]
        for group_values, group in groups:

            group_naming_dict = {
                col: val for col, val in zip(group_cols, group_values) if
                (not compact_group_name or result_df[col].nunique() > 1)
                }
            
            group_name = group_namer(group_naming_dict)
            
            subgroup_cols = [
                col for col in group.columns 
                if col not in [x_var]+random_cols+result_cols
                ]
            
            subgroup_y_stats = []
            subgroup_names = []
            subgroups = group.groupby(subgroup_cols) if subgroup_cols else [((), group)]
            n_subgroups = len(subgroups)

            for subgroup_values, subgroup in subgroups:

                subgroup_naming_dict = {
                    col: val for col, val in zip(subgroup_cols, subgroup_values) if
                    (not compact_subgroup_name or group[col].nunique() > 1)
                    }
                subgroup_name = subgroup_namer(subgroup_naming_dict)

                for col in subgroup.columns:
                    assert subgroup[col].nunique() == 1  or col in [x_var]+random_cols+result_cols, f'{col} is not a constant in {group_name}--{subgroup_name}'

                y_stat = subgroup.groupby(x_var).agg({y_var: ['mean', 'std'] for y_var in y_vars})
                y_stat = y_stat.reindex(sorted(y_stat.index))

                subgroup_names.append(subgroup_name)
                subgroup_y_stats.append(y_stat)

            """ Start of the hard-coded table generation for the paper  """
            subgroup_paper_tables = {}
            for j in range(n_subgroups):
                # selecting the full column "mean" for all super columns and all y_vars
                mn = subgroup_y_stats[j].loc[:, pd.IndexSlice[:, 'mean']]
                mn.columns = mn.columns.droplevel(1)
                std = subgroup_y_stats[j].loc[:, pd.IndexSlice[:, 'std']]
                std.columns = std.columns.droplevel(1)

                if subgroup_names[j] == 'C0':
                    C0_mn = mn.copy()
                    gain = pd.DataFrame('', index=mn.index, columns=mn.columns)
                    C0 = True
                else:
                    gain = (mn-C0_mn)/C0_mn
                    gain = gain.map(lambda x: f"({'+' if x>0 else '-'}{100*abs(x):>3.1f}%)")
                    C0 = False
                
                mn = mn.map(lambda x: f'{100*x:0>4.2f}')
                std = std.map(lambda x: f'±{100*x:.2f}')

                subgroup_paper_table = mn
                if not C0:
                    subgroup_paper_table = subgroup_paper_table + gain
                # subgroup_paper_table = subgroup_paper_table + std

                def filter_unstable_values(x):
                    if x.find('(') == -1:
                        cutoff = x.find('±')
                    else:
                        cutoff = x.find('(')
                    if float(x[:cutoff]) > 100:
                        return '>100'
                    else:
                        return x
                    
                subgroup_paper_table = subgroup_paper_table.map(filter_unstable_values)

                # rename index name  from frac_train_data to '#samples'
                subgroup_paper_table.index.name = '#samples'
                # multiplu index values by  8000 and round them
                subgroup_paper_table.index = (subgroup_paper_table.index*8000).astype(int)
                # using label_cleaner to make the column names better
                subgroup_paper_table.columns = [label_cleaner(col) for col in subgroup_paper_table.columns]
                subgroup_paper_tables[subgroup_names[j]] = subgroup_paper_table

            paper_table = pd.concat(subgroup_paper_tables, axis=1)
            # sorting the super columns
            paper_table = paper_table.reindex(sorted(paper_table.columns), axis=1)

            # creating two separate tables, one from all the second level columns r=1 but keeping the first level of columns C0 through C8
            # paper_table_r1 = paper_table.loc[:, pd.IndexSlice[:, 'r=1']]
            # paper_table_r5 = paper_table.loc[:, pd.IndexSlice[:, 'r=5']]
            # # dropping the second level of columns
            # paper_table_r1.columns = paper_table_r1.columns.droplevel(1)
            # paper_table_r5.columns = paper_table_r5.columns.droplevel(1)
            # paper_table_r1.to_latex(save_dir+f'/{group_name}_paper_table_r1.tex')
            # paper_table_r5.to_latex(save_dir+f'/{group_name}_paper_table_r5.tex')

            paper_table = paper_table.stack(level=1, future_stack=True)

            # Sort the index to maintain order
            paper_table = paper_table.sort_index()

            paper_table.index.set_names(['#samples', 'rollout'], inplace=True)

            def get_clean_caption_label(result_dir):
                caption = 'Average loss in percentage, and the relative change compared to C0 (no pretraining) in paranthesis. Negative changes indicate improvement.'
                if 'Diffusion' in result_dir:
                    nu = float(result_dir.split('_')[2][2:])
                    short_caption = '\\textbf{'+f'Results for Diffusion $\\nu={nu}$'+'.} '
                elif 'Advection' in result_dir:
                    beta = float(result_dir.split('_')[2][4:])
                    short_caption = '\\textbf{'+f'Results for Advection $\\beta={beta}$'+'.} '
                return short_caption+caption

            paper_table.to_latex(
                save_dir+f'/{group_name}_paper_table.tex',
                caption = get_clean_caption_label(result_dir),
                column_format = 'c'*len(paper_table.index.levels)+'|'+'c'*len(paper_table.columns)
                )
            paper_table.to_excel(save_dir+f'/{group_name}_paper_table.xlsx')

            # saving them separately as excel and latex

            def refine_latex(latex_file):
                with open(latex_file, 'r') as file:
                    latex = file.read()
                latex = latex.replace('%', '\%')
                latex = latex.replace('#', '\#')
                latex = latex.replace('>', '$>$')
                latex = latex.replace('\n\\begin{tabular}','\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}')
                latex = latex.replace('\end{tabular}', '\end{tabular}}')
                latex = latex.replace('\\begin{table}', '\\begin{table}[!htbp]')
                with open(latex_file, 'w') as file:
                    file.write(latex)

            # replacing % with \% in the latex files
            # refine_latex(save_dir+f'/{group_name}_paper_table_r1.tex')
            # refine_latex(save_dir+f'/{group_name}_paper_table_r5.tex')
            refine_latex(save_dir+f'/{group_name}_paper_table.tex')


            """ Finish of the hard-coded table generation for the paper """

            if all_y_in_one:
                fig, axes = plt.subplots(1, len(y_vars), figsize=(6*len(y_vars), 4))
                fig.suptitle(label_cleaner(group_name))
            else:
                figs, axes = zip(*[plt.subplots(1, 1, figsize=(6, 4)) for _ in y_vars])

            if len(y_vars) == 1:
                axes = [axes]

            for iy, (y_var, y_label, y_scale, y_ticker) in enumerate(zip(y_vars, y_labels, y_scales, y_tickers)):
                
                axes[iy].grid(linestyle='--')
                for j in range(n_subgroups):
                
                    # axes[iy].errorbar(
                    #     x = subgroup_y_stats[j].index,
                    #     y = 100*subgroup_y_stats[j][y_var]['mean'],
                    #     yerr = 100*subgroup_y_stats[j][y_var]['std'],
                    #     label = subgroup_names[j],
                    #     fmt = 'o-',
                    #     capsize = 3
                    # )
                    axes[iy].plot(
                        subgroup_y_stats[j].index,
                        100*subgroup_y_stats[j][y_var]['mean'],
                        'o-',
                        label = subgroup_names[j]
                    )

                axes[iy].set_xlabel(x_label)
                axes[iy].set_xscale(x_scale)
                axes[iy].set_xticks(subgroup_y_stats[0].index)
                # axes[iy].tick_params(axis='x', which='major', labelsize=16)
                axes[iy].set_xlim(left=0.00035)
                if x_ticker:
                    axes[iy].xaxis.set_major_formatter(FuncFormatter(x_ticker))
                
                axes[iy].set_ylabel('% '+r'$rL_2$ error')
                axes[iy].set_yscale(y_scale)
                # axes[iy].set_ylim(bottom=0, top=min(axes[iy].get_ylim()[1], 0.2 if 'advection' in result_dir.lower() else 0.02))
                ylim_dic = {
                    'Advection_Beta0.1': 5.0,
                    'Advection_Beta0.4': 10.0,
                    'Advection_Beta1.0': 10.0,
                    'Diffusion': 1.0
                    }
                def get_ylim(result_dir):
                    for key in ylim_dic:
                        if key in result_dir:
                            return ylim_dic[key]
                    return 1.00
                
                axes[iy].set_ylim(bottom=0, top=get_ylim(result_dir))
                if y_ticker:
                    axes[iy].yaxis.set_major_formatter(FuncFormatter(y_ticker))

                
                
                if not all_y_in_one:
                    if len(subgroup_names) > 1:
                        axes[iy].legend()
                        sort_legend(axes[iy])
                    axes[iy].set_title(label_cleaner(group_name))
                    if show:
                        plt.show(figs[iy])
                    if save:
                        figs[iy].savefig(save_dir+f'/{y_var}-vs-{x_var}--{group_name}.png')
                    plt.close(figs[iy])
                
                else:
                    axes[iy].set_title(label_cleaner(y_label))


            if all_y_in_one:
                if len(subgroup_names) > 1:
                    axes[-1].legend()
                    sort_legend(axes[-1])
                    
                if show:
                    plt.show(fig)
                if save:
                    fig.savefig(save_dir+f'/all-vs-{x_var}--{group_name}.png')

                plt.close(fig)

            


def plot_training(
        result_df, 
        result_dir,

        group_params = None,
        result_keywords = ['loss', 'time'],

        group_namer = lambda x: '|'.join([f'{k}={v}' for k, v in x.items()]),
        subgroup_namer = lambda x: '|'.join([f'{k}={v}' for k, v in x.items()]),

        compact_group_name = True,
        compact_subgroup_name = False,

        smooth_window = 10,
        zoom_size = 0.1,
        save_dir = None,
        save = True,
        show = False
        ):

    
    if save_dir is None:
        save_dir = result_dir + '/processed'

    result_cols = [
        col for col in result_df.columns
        if any([kw in col.lower() for kw in result_keywords])
    ]

    if group_params is None:
        group_params = [
            col for col in result_df.columns 
            if col not in result_cols
            ]

    group_cols = [col for col in result_df.columns if col in group_params]

    for group_values, group in result_df.groupby(group_cols):

        train_fig, train_axes = plt.subplots(1, 3, figsize=(18, 4))
        
        title = group_namer({
            col: value for col, value in zip(group_cols, group_values)
            if (not compact_group_name or result_df[col].nunique() > 1)
            })
        train_fig.suptitle(title)

        init_ymin = np.inf
        init_ymax = -np.inf
        final_ymin = np.inf
        final_ymax = -np.inf

        for idx, row in group.iterrows():
            with open(result_dir+f'/trainings/training_{idx:05d}.pkl', 'rb') as file:
                training_result = pickle.load(file)
            train_loss_iterations = training_result['training_losses']

            n_iters = len(train_loss_iterations)
            
            # smoothing
            train_loss_iterations_smooth = []
            for i in range(n_iters):
                window = train_loss_iterations[max(0, i+1-smooth_window):i+1]
                train_loss_iterations_smooth.append(np.mean(window))

            zoom_xlim = round(n_iters*zoom_size)

            label = subgroup_namer({
                col: value for col, value in row.items()
                if (not compact_subgroup_name or group[col].nunique() > 1)
            })
            
            train_axes[0].plot(train_loss_iterations_smooth,  label=label)
            train_axes[1].plot(train_loss_iterations_smooth,  label=label)
            train_axes[2].plot(train_loss_iterations_smooth,  label=label)

            init_ymin = min(init_ymin, np.min(train_loss_iterations_smooth[:zoom_xlim]))
            init_ymax = max(init_ymax, np.max(train_loss_iterations_smooth[:zoom_xlim]))
            final_ymin = min(final_ymin, np.min(train_loss_iterations_smooth[-zoom_xlim:]))
            final_ymax = max(final_ymax, np.max(train_loss_iterations_smooth[-zoom_xlim:]))

        for train_ax in train_axes:
            train_ax.set_xlabel('iteration')
            train_ax.grid(linestyle='--')

        train_axes[0].set_ylabel('relative L2 training loss')
        train_axes[-1].legend()
        sort_legend(train_axes[-1])

        train_axes[0].set_title('all iterations')
        
        train_axes[1].set_title('initial iterations')
        train_axes[1].set_xlim(-1, zoom_xlim)
        train_axes[1].set_ylim(init_ymin, init_ymax)


        train_axes[2].set_title('final iterations')
        train_axes[2].set_xlim(n_iters-zoom_xlim-1, n_iters)
        train_axes[2].set_ylim(final_ymin, final_ymax)
        
        if show:
            plt.show(train_fig)
        if save:
            train_fig.savefig(save_dir+f'/training-{title}.png', bbox_inches='tight')


# Functions and helpers for plotting model parameters against each other:

def get_array(T: torch.Tensor) -> np.ndarray:
    return T.detach().cpu().numpy().flatten()


def fit_line(t1, t2):
    m, b = np.polyfit(t1, t2, 1)
    r2 = np.corrcoef(t1, t2)[0,1]**2
    return m, b, r2


def plot_param_comparison(ax, T1, T2, s=5, alpha=0.5):
    t1, t2 = get_array(T1), get_array(T2)
    ax.scatter(t1, t2, s=s, alpha=alpha)
    lim = np.array([min(t1.min(), t2.min()), max(t1.max(), t2.max())])
    ax.plot(lim, lim, color='black', label='y=x')
    # fitting a line to the data and writing the r^2 value
    m, b, r2 = fit_line(t1, t2)
    ax.plot(lim, m*lim + b, color='red', linestyle='--', label='fitted line')
    ax.set_ylabel(f'$y={m:.2f}x{"+"if b>=0 else ""}{b:.2f}, R^2={r2:.2}$')
    ax.grid(linestyle='--')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal', 'box')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend()
    sort_legend(ax)
    return m, b, r2


def compare_FFNOs(ffno_1, ffno_2, show=True, save=''):

    if save:
        os.makedirs(save, exist_ok=True)
    # looping over layers and plotting fourier weight and feedforward weight and biases against each other

    # input projectors:
    n_invars = len(ffno_1.projector.in_vars)
    fig, axes = plt.subplots(n_invars, 2, figsize=(12, 6*n_invars))
    if n_invars == 1:
        axes = axes.reshape(1,2)
    fig.suptitle(f'Input projectors')
    for i, v in enumerate(ffno_1.projector.in_vars):
        plot_param_comparison(
            axes[i,0], 
            ffno_1.projector.in_projector[v].weight, 
            ffno_2.projector.in_projector[v].weight
            )
        axes[i,0].set_title(f'{v} weights')

        plot_param_comparison(
            axes[i,1], 
            ffno_1.projector.in_projector[v].bias, 
            ffno_2.projector.in_projector[v].bias
            )
        axes[i,1].set_title(f'{v} biases')

    if show:
        fig.show()
    if save:
        fig.savefig(save+'/input_projectors.png')
    plt.close(fig)

    # output projectors
    n_outvars = len(ffno_1.projector.out_vars)
    fig, axes = plt.subplots(n_outvars, 2, figsize=(12, 6*n_outvars))
    if n_outvars == 1:
        axes = axes.reshape(1,2)
    for i, v in enumerate(ffno_1.projector.out_vars):

        plot_param_comparison(
            axes[i,0], 
            ffno_1.projector.out_projector[v].weight, 
            ffno_2.projector.out_projector[v].weight
            )
        axes[i,0].set_title(f'{v} weights')

        plot_param_comparison(
            axes[i,1], 
            ffno_1.projector.out_projector[v].bias, 
            ffno_2.projector.out_projector[v].bias
            )
        axes[i,1].set_title(f'{v} biases')

    if show:
        fig.show()
    if save:
        fig.savefig(save+f'/output_projectors.png')
    plt.close(fig)


    # looping over layers
    for i in range(ffno_1.n_layers):

        layer1 = ffno_1.layers[i]
        layer2 = ffno_2.layers[i]
        # plotting fourier weight
        nf1 = len(layer1.fourier_weight)
        nf2 = len(layer2.fourier_weight)
        assert nf1 == 1        
        fig, axes = plt.subplots(1, 2, figsize=(12,6))
        fig.suptitle(f'Layer {i} Fourier Weights')

        # plotting real parts
        plot_param_comparison(
            axes[0], 
            layer1.fourier_weight[0][..., 0], 
            layer2.fourier_weight[0][..., 0]
            )
        axes[0].set_title(f'real')

        # plotting imaginary parts
        plot_param_comparison(
            axes[1], 
            layer1.fourier_weight[0][..., 1], 
            layer2.fourier_weight[0][..., 1]
            )
        axes[1].set_title(f'imag')

        if show:
            fig.show()
        if save:
            fig.savefig(save+f'/layer{i}_fourier.png')
        plt.close(fig)

        # plotting feedforward layers
        fig, axes = plt.subplots(2, 2, figsize=(12,12))
        fig.suptitle(f'Layer {i} Feedforward')

        # 1st layer weight
        plot_param_comparison(
            axes[0,0], 
            layer1.feedforward.layers[0].weight, 
            layer2.feedforward.layers[0].weight
            )
        axes[0,0].set_title('1st layer weights')

        # 1st layer bias
        plot_param_comparison(
            axes[0,1], 
            layer1.feedforward.layers[0].bias, 
            layer2.feedforward.layers[0].bias
            )
        axes[0,1].set_title('1st layer biases')

        # 2nd layer weight
        plot_param_comparison(
            axes[1,0], 
            layer1.feedforward.layers[2].weight, 
            layer2.feedforward.layers[2].weight
            )
        axes[1,0].set_title('2nd layer weights')

        # 2nd layer bias
        plot_param_comparison(
            axes[1, 1], 
            layer1.feedforward.layers[2].bias, 
            layer2.feedforward.layers[2].bias
        )
        axes[1, 1].set_title('2nd layer biases')

        if show:
            fig.show()
        if save:
            fig.savefig(save+f'/layer{i}_feedforward.png')
        plt.close(fig)
