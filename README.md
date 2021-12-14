# About the data set
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Ref: https://www.kaggle.com/zalando-research/fashionmnist

## Steps -

1. Create a conda environment using VSCode first in your respective directory/ you can clone this repository itself.

```bash
conda create --prefix ./env python=3.7 -y
```
2. Activate the conda environment

```bash
conda activate ./env
```
OR
```bash
source activate ./env
```


4. install the requirements
```bash
pip install -r requirements.txt
```

5. initialize the dvc project
```bash
dvc init
```

6. Run the ML pipeline using the command
```bash
dvc repro
```

7. View the ML pipeline setup using the command
```bash
dvc dag
```
