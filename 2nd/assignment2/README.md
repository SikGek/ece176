# Assignment 2: Building Neural Networks for multi-class classification

Assignment 2 includes two parts:
1. Building your neural network model using a toy dataset
2. Training and testing your built model on our favourite CIFAR10 dataset

## Prepare Datasets

Before you start, you need to run the following command (in terminal or in notebook beginning with `!` ) to download the datasets:

```sh
# This command will download the dataset and put it in "./dataset".
python get_dataset.py
```

## Implementation

You should run all code blocks in the following jupyter notebooks and write your answers to all inline questions included in the notebooks:

1. `test_neural_net.ipynb`
2. `cifar10_classification.ipynb`

Go through the notebooks to understand the structure. These notebook will require you to first complete the following implementations.

You should implement four layers in four different python file(in "layers/"):
1. `layers/linear.py`     : Implement linear layers with arbitrary input and output dimensions
2. `layers/relu.py`       : Implement ReLU activation function, forward and backward pass
3. `layers/softmax.py`    : Implement softmax function to calculate class probabilities
4. `layers/loss_func.py`  : Implement CrossEntropy loss, forward and backward pass

You are required to go through all the modules (the ones that are already implemented for you as well) one by one to to understand the structure. This will help you when transitioning to deep learning libraries such as PyTorch. Here are some files that you should go through that are already implemented for you:

1. `layers/sequential.py`
2. `utils/trainer.py`
3. `utils/optimizer.py`

After you complete all the functions and questions, you should upload the following files to Gradescope:

1. 4 `.py` files that you have to implement (linear, relu, softmax, loss_func) in "layers" directory.
2. The two notebook source .ipynb files
3. Exported PDF of notebooks(You could export Jupyter notebook as PDF in web portal) and merge them

You should organize files like this:
1. Put all source files(4 python files and two notebooks) into a single ".zip" file then upload it on Gradescope
2. Merge exported PDF of notebooks into a single PDF file and upload it on Gradescope


## Instructions on how to use Jupyter notebooks can be found in assignment 1 README file


## Things to keep in mind:
1. Edit only the parts that are asked of you. Do not change the random seed where it is set. This might not make it possible to get similar results.
2. Try to avoid for loops in all the implementations, the assignment can be solved without any for loops (vectorized implementation)