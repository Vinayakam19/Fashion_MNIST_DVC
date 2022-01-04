# About the data set
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.



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

7. View the results of the model using DVC interactive studio
![image](https://user-images.githubusercontent.com/45694329/145950224-e3142fac-6487-4ffd-92d6-ac5b41eea4a9.png)


7. View the ML pipeline setup using the command
```bash
dvc dag
```
![ML-Pipeline-FMNIST](https://user-images.githubusercontent.com/45694329/145949088-dc9c5937-4fee-4980-893f-0c09d2b26d47.png)

Note : 
1. dvc needs to be installed first before running dvc repro. dvc can be installed using `pip install dvc`
2. Experiment results can be viewed in the Interactive studio using the link (https://studio.iterative.ai)
3. Using Continous Machine Learning (CML) CI-CD pipelines can be created in the github (https://github.com/iterative/cml#getting-started)


