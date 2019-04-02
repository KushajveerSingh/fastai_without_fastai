import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# NOT -> ParameterModule
# NOT -> children_and_parameters
# NOT -> flatten_model
# NOT -> lr_range
# NOT -> scheduling functions
# NOT -> SmoothenValue 
# YES -> lr_find
# NOT -> plot_lr_find

# NOT TO BE MODIFIED
class ParameterModule(nn.Module):
    "Register a lone parameter 'p' in a module"
    def __init__(self, p:nn.Parameter):
        super().__init__()
        self.val = p
    def forward(self, x): 
        return x

# NOT TO BE MODIFIED
# To be used to flatten_model
def children_and_parameters(m:nn.Module):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()],[])
    for p in m.parameters():
        if id(p) not in children_p: children.append(ParameterModule(p))
    return children

# NOT TO BE MODIFIED
flatten_model = lambda m: sum(map(flatten_model,children_and_parameters(m)),[]) if len(list(m.children())) else [m]

# NOT TO BE MODIFIED
def lr_range(model, lr):
    """
    Build differential learning rate from lr. It will give you the
    Arguments:
        model :- torch.nn.Module
        lr :- float or slice
    Returns:
        Depending upon lr
    """
    if not isinstance(lr, slice): 
        return lr
    
    num_layer = len([nn.Sequential(*flatten_model(model))])
    if lr.start: 
        mult = lr.stop / lr.start
        step = mult**(1/(num_layer-1))
        res = np.array([lr.start*(step**i) for i in range(num_layer)])
    else:
        res = [lr.stop/10.]*(num_layer-1) + [lr.stop]
    
    return np.array(res)

# NOT TO BE MODIFIED
# These are the functions that would give us the values of lr. Liks for linearly
# increasing lr we would use annealing_linear.
# You can add your own custom function, for producing lr.
# By defualt annealing_exp is used for both lr and momentum
def annealing_no(start, end, pct:float):
    "No annealing, always return `start`."
    return start
def annealing_linear(start, end, pct:float):
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start + pct * (end-start)
def annealing_exp(start, end, pct:float):
    "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start * (end/start) ** pct
def annealing_cos(start, end, pct:float):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out
def do_annealing_poly(start, end, pct:float, degree):
    return end + (start-end) * (1-pct)**degree
    
# NOT TO BE MODIFIED
class Stepper():
    """
    Used to step from start, end ('vals') over 'n_iter' iterations on a schedule. 
    We will create a stepper object and then use one of the above annelaing functions,
    to step from start lr to end lr.
    """
    def __init__(self, vals, n_iter:int, func=None):
        self.start, self.end = (vals[0], vals[1]) if isinstance(vals, tuple) else (vals,0)
        self.n_iter = max(1, n_iter)
        if func is None:
            self.func = annealing_linear if isinstance(vals, tuple) else annealing_no
        else:
            self.func = func
        self.n = 0
    
    def step(self):
        "Return next value along annealed schedule"
        self.n += 1
        return self.func(self.start, self.end, self.n/self.n_iter)
        
    @property
    def is_done(self)->bool:
        "Return 'True' if schedule completed"
        return self.n >= self.n_iter

# NOT TO BE MODIFIED
class SmoothenValue():
    "Create a smooth moving average for a value (loss, etc) using `beta`."
    def __init__(self, beta:float):
        self.beta,self.n,self.mov_avg = beta,0,0
    def add_value(self, val:float)->None:
        "Add `val` to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)

# TO BE MODIFIED IN SOME CASES
def lr_find(data_loader, model, loss_fn, opt, wd:int=0, start_lr:float=1e-7, end_lr:float=10, 
            num_it:int=100, stop_div:bool=True, smooth_beta:float=0.98, use_gpu:bool=True, 
            device=torch.device('cuda'), anneal_func=annealing_exp):
    """
    The main function that you will call to plot learning_rate vs losses graph. It is
    the only function from lr_find.py that you will call. By default it will use GPU. It
    assumes your model is already on GPU if you use use_gpu.

    Arguments:-
        data_loader :- torch.utils.data.DataLoader
        model :- torch.nn.Module
        loss_fn :- torch.nn.LossFunction
        opt :- torch.optim.Optimizer
        wd :- weight decay (default=0).
        start_lr :- The learning rate from where to start in lr_find (default=1e-7)
        end_lr :- The learning rate at which to end lr_find (default=10)
        num_it :- Number of iterations for lr_find (default=100)
        stop_div :- If the loss diverges, then stop early (default=True)
        smooth_beta :- The beta value to smoothen the running avergae of the loss function (default=0.98)
        use_gpu :- True (train on GPU) else CPU
        anneal_func :- The step function you want to use (default exp)
        device :- Torch device to use for training model (default GPU)
    Returns:
        losses :- list of smoothened version of losses
        lrs :- list of all lrs that we test
    """
    model.train()

    stop = False
    flag = False
    best_loss = 0.
    iteration = 0
    losses = []
    lrs = []
    lrs.append(start_lr)

    start_lr = lr_range(model, start_lr)
    start_lr = np.array(start_lr) if isinstance(start_lr, (tuple, list)) else start_lr
    end_lr = lr_range(model, end_lr)
    end_lr = np.array(end_lr) if isinstance(end_lr, (tuple, list)) else end_lr

    sched = Stepper((start_lr, end_lr), num_it, anneal_func)
    smoothener = SmoothenValue(smooth_beta)
    epochs = int(np.ceil(num_it/len(data_loader)))

    # save model_dict
    model_state = model.state_dict()
    opt_state = opt.state_dict()

    # Set optimizer learning_rate = start_lr
    for group in opt.param_groups:
        group['lr'] = sched.start

    for i in range(epochs):
        for data in data_loader:
            opt.zero_grad()
            
            ################### TO BE MODIFIED ###################
            # Depending on your model, you will have to modify your
            # data pipeline and how you give inputs to your model.
            inputs, labels = data
            if use_gpu:
                inputs = inputs.to(device)
                labels = labels.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            #####################################################
            
            if use_gpu:
                smoothener.add_value(loss.detach().cpu())
            else:
                smoothener.add_value(loss.detach())
            smooth_loss = smoothener.smooth
            losses.append(smooth_loss)
            loss.backward()

            ################### TO BE MODIFIED ###################
            # For AdamW. If you want to use Adam, comment these lines
            for group in opt.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-wd * group['lr'], param.data)
            #####################################################

            opt.step()

            # Change lr
            new_lr = sched.step()
            lrs.append(new_lr)
            for group in opt.param_groups:
                group['lr'] = new_lr
            
            ################### TO BE MODIFIED ###################
            # You necessarily don't want to change it. But in cases
            # when you are maximizing the loss, then you will have
            # to change it.
            if iteration == 0 or smooth_loss < best_loss:
                best_loss = smooth_loss
            iteration += 1

            if sched.is_done or (stop_div and (smooth_loss > 4*best_loss or torch.isnan(loss))):
                flag = True
                break
            #####################################################

            if iteration%10 == 0:
                print(f'Iteration: {iteration}')
        
        if flag:
            break
    
    # Load state dict
    model.load_state_dict(model_state)
    opt.load_state_dict(opt_state)

    lrs.pop()

    print(f'LR Finder is complete.')

    return losses, lrs

# NOT TO BE MODIFIED
def plot_lr_find(losses, lrs, skip_start:int=10, skip_end:int=5, suggestion:bool=False, return_fig:bool=None):
    """
    It will take the losses and lrs returned by lr_find as input.

    Arguments:-
        skip_start -> It will skip skip_start lrs from the start
        skip_end -> It will skip skip_end lrs from the end
        suggestion -> If you want to see the point where the gradient changes most
        return_fig -> True then get the fig in the return statement
    """
    lrs = lrs[skip_start:-skip_end] if skip_end > 0 else lrs[skip_start:]
    losses = losses[skip_start:-skip_end] if skip_end > 0 else losses[skip_start:]
    losses = [x.item() for x in losses]

    fig, ax = plt.subplots(1, 1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

    if suggestion:
        try:
            mg = (np.gradient(np.array(losses))).argmin()
        except:
            print("Failed to compute the gradients, there might not be enough points.")
            return
        print(f"Min numerical gradient: {lrs[mg]:.2E}")
        ax.plot(lrs[mg], losses[mg], markersize=10, marker='o', color='red')
    
    if return_fig is not None:
        return fig
