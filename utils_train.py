from typing import List, Callable
from tqdm import tqdm
import time
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

Device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_csv(s:str, full:list, func=str):
    if not s: # if s is None or an empty string
        return []
    if 'all' in s:
        return list(full)
    return [func(i.strip()) for i in s.split(',') if i.strip()]


def parse_csv_scalers(s:str):
    if not s:
        return [1.0]
    return [float(i.strip()) for i in s.split(',') if i.strip()]


def Relative_Lp_Loss(pred, true, reduction='mean', norm_dims_start=1, p=2, keepdict=False):

    if isinstance(pred, dict) and isinstance(true, dict):
        if not keepdict:
        # Average the loss of different variables
            return torch.stack([
                Relative_Lp_Loss(pred[key], true[key], reduction, norm_dims_start, p) for key in pred
                ]).mean()
        else:
            # for reporting separate losses for different variables
            # not differentiable. DO NOT USE IN TRAINING
            return {key: Relative_Lp_Loss(pred[key], true[key], reduction, norm_dims_start, p).item() for key in pred}
    
    assert pred.shape == true.shape, 'pred and true should have the same shape'
    norm_dims = tuple(range(norm_dims_start, len(pred.shape)))
    error_norm = torch.norm(pred - true, dim=norm_dims, p=p)
    true_norm = torch.norm(true, dim=norm_dims, p=p)
    relative_lp_loss = error_norm / true_norm
    if reduction=='mean':
        return relative_lp_loss.mean()
    elif reduction=='sum':
        return relative_lp_loss.sum()


def get_next_input(prev_input, prev_output):
    """
    prev_input: (N, in_snapshots, ...)
    prev_output: (N, out_snapshots, ...)
    where ... should be the same for prev_input and prev_output
    """
    if isinstance(prev_input, dict) and isinstance(prev_output, dict):
        return {key: get_next_input(prev_input[key], prev_output[key]) for key in prev_input.keys()}
    
    in_snapshots, out_snapshots = prev_input.shape[1], prev_output.shape[1]
    if in_snapshots < out_snapshots:
        new_input = prev_output[:, -in_snapshots:, ...]#.clone() # I'm not sure if i should clone it or not
    elif in_snapshots == out_snapshots:
        new_input = prev_output
    elif in_snapshots > out_snapshots:
        new_input = torch.cat([prev_input[:, out_snapshots:, ...], prev_output], dim=1)
    return new_input


def forward_pass_loss(
        model : torch.nn.Module, 
        xs : List[torch.Tensor],
        loss_fn = Relative_Lp_Loss,
        loss_reduction = 'mean',
        keep_time = False
        ):

    rollout = len(xs) - 1

    if rollout == 0:
        # Assuming we are AutoEncoding, since each sample has only one tensor
        x = xs[0]
        x_rec = model(x)
        final_loss = loss_fn(x_rec, x, reduction=loss_reduction)

    elif rollout > 0:

        model_input = xs[0]
        losses = []
        for r in range(1, rollout+1):
            model_output = model(model_input)
            loss = loss_fn(model_output, xs[r], reduction=loss_reduction)
            losses.append(loss)
            if rollout == 1: break # to avoid unnecessary errors for simple input to output tasks
            model_input = get_next_input(model_input, model_output)

        losses = torch.stack(losses)
        if keep_time: # return the losses for each rollout step
            final_loss = losses
        else: # average over rollout steps
            final_loss = losses.mean()

    return final_loss


def train_epoch(
        model : torch.nn.Module,
        dataloader : DataLoader,
        opt : torch.optim.Optimizer,
        loss_fn = Relative_Lp_Loss,
        loss_reduction = 'mean'
        ):
    model.train()
    training_losses = []
    times = []

    batch_pbar = tqdm(dataloader, desc='train batch', leave=False, unit='batch')
    for i, xs in enumerate(batch_pbar):
        start_time = time.time()
        some_key = list(xs[0].keys())[0]
        b = xs[0][some_key].shape[0] 

        opt.zero_grad()
        loss = forward_pass_loss(
            model = model,
            xs = xs,
            loss_fn = loss_fn,
            loss_reduction = loss_reduction
        )
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time-start_time)
        if loss_reduction == 'mean':
            training_losses.append(loss.item())
        elif loss_reduction == 'sum':
            training_losses.append(loss.item()/b)
        batch_pbar.set_postfix_str(f'batch loss: {training_losses[-1]:.6f}')
    return training_losses, times


@torch.no_grad()
def validate_epoch(
    model : torch.nn.Module,
    dataloader : DataLoader,
    loss_fn = Relative_Lp_Loss,
    keep_time = True,
    data_name = 'train data'
    ):
    model.eval()
    sum_val_loss = 0.
    sum_batch_size = 0
    times = []
    batch_pbar = tqdm(dataloader, desc=f'validating on {data_name} ', leave=False, unit='batch')
    for i, xs in enumerate(batch_pbar):
        some_key = list(xs[0].keys())[0]
        b = xs[0][some_key].shape[0] 

        start_time = time.time()
        loss = forward_pass_loss(
            model = model,
            xs = xs,
            loss_fn = loss_fn,
            loss_reduction = 'sum',
            keep_time = keep_time
        )
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time-start_time)

        sum_val_loss = sum_val_loss + loss
        sum_batch_size += b
        average_loss = (sum_val_loss/sum_batch_size).mean().item()
        batch_pbar.set_postfix_str(f'loss: {average_loss:.6f}')
    
    # if keep_time, return the losses for each time step
    # if not, return the average loss over all time steps

    # the loss is averaged over all samples in the dataset
    return sum_val_loss/sum_batch_size, times


def train_epochs(
        model : torch.nn.Module,

        train_dataset : Dataset,
        config_train_data_for_training : Callable,
        config_train_data_for_validation : Callable,

        val_dataset : Dataset,
        config_val_data_for_validation : Callable,

        optimizer = optim.Adam,
        loss_fn = Relative_Lp_Loss,
        loss_reduction = 'mean',
        epochs = 100,
        batch_size = 64,

        val_freq = None
        # NEED TO IMPLEMENT VAL_ROLLOUTS
    ):

    val_freq = val_freq or epochs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    opt = optimizer(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.2, patience=5)

    training_losses = []
    val_loss_train_data = []
    val_loss_val_data = []
    train_times = []
    val_times = []

    model.train()
    config_train_data_for_training(dataset=train_dataset)

    epoch_pbar = tqdm(range(1, epochs+1), desc='epoch ', leave=False, unit='epoch')
    for epoch in epoch_pbar:

        # TRAINING 
        training_losses_during_epoch, train_batch_times = train_epoch(
            model = model,
            dataloader = train_loader,
            opt = opt,
            loss_fn = loss_fn,
            loss_reduction = loss_reduction
        )

        train_times.extend(train_batch_times)
        training_losses.extend(training_losses_during_epoch)
        avg_running_train_loss = torch.tensor(training_losses_during_epoch).mean().item()
        scheduler.step(avg_running_train_loss)
        epoch_pbar.set_postfix_str(f'avg running train loss: {avg_running_train_loss:.6f}')

        # VALIADTION 
        if epoch % val_freq == 0:
            model.eval()

            # On train data 
            config_train_data_for_validation(dataset=train_dataset)
            train_loss, val_batch_times = validate_epoch(
                model = model,
                dataloader = train_loader,
                loss_fn = loss_fn
            )
            val_loss_train_data.append(train_loss)
            val_times.extend(val_batch_times)

            # On val data 
            config_val_data_for_validation(dataset=val_dataset)
            val_loss, val_batch_times_val = validate_epoch(
                model = model,
                dataloader = val_loader,
                loss_fn = loss_fn
            )
            val_loss_val_data.append(val_loss)

            model.train()
            config_train_data_for_training(dataset=train_dataset)


    batch_train_time = np.mean(train_times)
    batch_val_time = np.mean(val_times)


    out_dict = {
        'training_losses': training_losses,
        'val_loss_train_data': val_loss_train_data,
        'val_loss_val_data': val_loss_val_data,
        'batch_train_time': batch_train_time,
        'batch_val_time': batch_val_time
    }
    return out_dict


def train_iters(
        model : torch.nn.Module,

        train_dataset : Dataset,
        config_train_data_for_training : Callable,
        config_train_data_for_validation : Callable,

        val_dataset : Dataset,
        config_val_data_for_validation : Callable,

        optimizer = optim.Adam,
        loss_fn = Relative_Lp_Loss,
        loss_reduction = 'mean',
        iters = 5000,
        batch_size = 64,

        val_iters = None,
        val_rollouts = None,

        running_avg_window = 10,
    ):

    if not val_iters:
        val_iters = [iters]
    elif isinstance(val_iters, int):
        val_iters = [val_iters]
    elif isinstance(val_iters, str):
        val_iters = parse_csv(val_iters, full=[iters], func=int)
    
    if not val_rollouts:
        val_rollouts = [1]
    elif isinstance(val_rollouts, int):
        val_rollouts = [val_rollouts]
    elif isinstance(val_rollouts, str):
        val_rollouts = parse_csv(val_rollouts, full=[1], func=int)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    opt = optimizer(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.2, patience=50)

    training_losses = []
    val_loss_train_data = {f'r{r}': {} for r in val_rollouts}
    val_loss_val_data = {f'r{r}': {} for r in val_rollouts}

    train_times = []
    val_times = {f'r{r}': [] for r in val_rollouts}

    it = 0
    iter_pbar = tqdm(range(iters), desc='train iter ', leave=False)

    model.train()
    config_train_data_for_training(dataset=train_dataset)

    train_rolllout = train_dataset.rollout

    while it < iters:
        
        # TRAINING
        for xs in train_loader:
            start_time = time.time()

            some_key = list(xs[0].keys())[0]
            b = xs[0][some_key].shape[0]
            opt.zero_grad()
            loss = forward_pass_loss(
                model = model,
                xs = xs,
                loss_fn = loss_fn,
                loss_reduction = loss_reduction
            )
            loss.backward()
            opt.step()

            torch.cuda.synchronize()
            end_time = time.time()

            train_times.append(end_time-start_time)
            if loss_reduction == 'mean':
                training_losses.append(loss.item())
            elif loss_reduction == 'sum':
                training_losses.append(loss.item()/b)

            running_avg_train_loss = np.mean(training_losses[-running_avg_window:])
            scheduler.step(running_avg_train_loss)

            iter_pbar.set_postfix_str(f'running avg train loss: {running_avg_train_loss:.6f}')
            it += 1
            iter_pbar.update(1)
            
            if it in val_iters:
                # on train data
                config_train_data_for_validation(dataset=train_dataset)
                model.eval()

                r_pbar = tqdm(val_rollouts, leave=False)
                for r in r_pbar:
                    r_pbar.set_description_str(f'validation rollout '+''.join([f'[{rr}]' if rr == r else f'({rr})' for rr in val_rollouts])+' ')

                    # On train data
                    config_train_data_for_validation(dataset=train_dataset)
                    train_dataset.config_autoregression(rollout=r)

                    train_loss, val_batch_times = validate_epoch(
                        model = model,
                        dataloader = train_loader,
                        loss_fn = loss_fn,
                        keep_time = True, # the result will be of length r (val_rollout) averaged over samples
                        data_name = 'train data'
                    )
                    val_loss_train_data[f'r{r}'][f'it{it}'] = train_loss.cpu().numpy()
                    val_times[f'r{r}'].extend(val_batch_times)

                    
                    # On val data
                    config_val_data_for_validation(dataset=val_dataset)
                    val_dataset.config_autoregression(rollout=r)

                    val_loss, batch_times_unused = validate_epoch(
                        model = model,
                        dataloader = val_loader,
                        loss_fn = loss_fn,
                        keep_time = True, # the result will be of length r (val_rollout) averaged over samples
                        data_name = 'val data'
                    )
                    val_loss_val_data[f'r{r}'][f'it{it}'] = val_loss.cpu().numpy()

                # reset the model to training mode
                model.train()
                config_train_data_for_training(dataset=train_dataset)
                train_dataset.config_autoregression(rollout=train_rolllout)


            if it >= iters:
                break

    batch_train_time = np.mean(train_times)
    # batch_val_time = np.mean(val_times)
    batch_val_time = {f'r{r}': np.mean(val_times[f'r{r}']) for r in val_rollouts}

    out_dict = {
        'training_losses': training_losses, # a long list containing the loss of each batch (iteration) during training.

        # the following two are each a dictionary
        # each item corresonds to the average validation loss for a certain rollout
        # each item's value is again a dictionary, where each key is the iteration number and the value is the loss (over rolled out steps)
        # This way, we can check if there is a jump start in early validation loss thanks to transfer learning
        'val_loss_train_data': val_loss_train_data,
        'val_loss_val_data': val_loss_val_data,

        # the following two are the average time taken for each batch during training and validation, each are a scaler in seconds.
        'batch_train_time': batch_train_time,
        # for validation, we have time for each rollout
        'batch_val_time': batch_val_time
    }
    return out_dict
