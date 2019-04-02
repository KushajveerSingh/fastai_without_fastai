# fastai_without_fastai (fwf)

fwf is not made with the goal of creating a library from where you can import functions, for that you have [fastai](https://docs.fast.ai/). The motivation behind this project is to make the useful functions of fastai in PyTorch like lr_find, fit_one_cycle available. 

(If you are not familiar with Cyclic learning, look at the papers of [Leslie N. Smith](https://arxiv.org/search/cs?searchtype=author&query=Smith%2C+L+N). I highly recommend to read his papers, as you would get to know about the state-of-art methods for training neural networks in 2019 and also [fastai courses](https://course.fast.ai/), where Jeremy teaches his own best methods).

# How the repo works

There are two main folders:
1. src
2. notebooks

In the `src` folder you will find .py scripts with the function name that it implements and in the `notebooks` folder you would find notebook with the same name, which would show you how to use the functions in .py script. I also compare my results, with fastai library results in the notebooks.

In the scripts you will find something like this in the beginning
```python
# NOT -> ParameterModule
# NOT -> children_and_parameters
# NOT -> flatten_model
# NOT -> lr_range
# NOT -> scheduling functions
# NOT -> SmoothenValue 
# YES -> lr_find
# NOT -> plot_lr_find
```
It lists all the functions/classes in that script. __NOT__ means you don't need to modify that function and __YES__ means you will have to modify the function in some cases. Example of the changes you would have to do
```python
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
```

Ideally, I wrote the .py scripts such that if you want to understand all the underlying code, you can start from the beginning of the file and read the code sequentially.

# Index of all functions

1. lr_find.py
    
    * lr_find
    * plot_lr_find

2. fit_one_cycle.py
    
    TODO
    

# Some Notation
* Python type-checking
    ```python
    def func(a:int, b:bool=True, model:nn.Module=None)->int:
        pass
    ```
    To understand the code first remove all the types and you get
    ```python
    def func(a, b, model):
        pass
    ```
    Now what the initial-code meant is, `a` is of type _int_, `b` is of type _bool_ with default value=True, and `model` is of type _torch.nn.Module_ with default None.
    
    The last `->int` means, that the function returns an int.

    Python type-checking is not necessary for these functions to work, but it help's understand the arguments better.

## Blog 

I am writing a blog series on __State of the Art Methods to train Nerual Networks in 2019__. I will give all the Leslie N. Smith's techniques along with Jeremy Howard's techniques for training neural networks. (Note:- I will update this with a link, blog post is still in draft. Will be posted by this week)

# Issues

If some documentation is missing or some piece of code is not clear, open an issue and I would be happy to clarify. Also, if any bug is found, file a PR or open an issue.

# License
fastai_without_fastai is MIT licensed, as found in the LICENSE file.