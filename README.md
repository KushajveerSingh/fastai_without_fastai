# fastai_without_fastai (fwf)

fwf is not a library that you can import and use the functions directly, for that you have [fastai](https://docs.fast.ai/). The motivation behind this project was to use the useful functions of fastai in PyTorch like lr_find, fit_one_cycle (If you are not familiar with Cyclic learning, look at the papers of [Leslie N. Smith](https://arxiv.org/search/cs?searchtype=author&query=Smith%2C+L+N)).

There are detailed jupyter notebooks in the `notebook` folder. Just look up for the functions that are implemented in each notebook, and follow that notebook top-to-bottom. (The index of all functions is provided below and also in the notebook folder).

## Index



## Some Notation
* In the notebooks, there will be code blocks like
    ```python
    ########## Boiler Code ##########
    some code
    #################################
    ```
    You don't need to modify this code, if you are just interested in getting the fastai equivalent of that function. However, I highly recommend to understand them, in cases where you want to change underlying functionality.

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
    Now what the initial-code meant is, a is of type int, b is of type bool with default value=True, and model is of type torch.nn.Module with default None.
    
    The last `->int` means, that the function returns an int.

    Python type-checking is not necessary for these functions to work, but they help understand the arguments better.

## Issues
In case I missed some documentation, or some bug is found is found, file a PR or open an issue. Also, if some argument, or part of code is not clear, you can also open an issue. If you want some functions added, open an issue or open a PR.

## License
fastai_without_fastai is MIT licensed, as found in the LICENSE file.