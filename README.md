# Exploration in Action Space

## Reproducing Experiments

### Setup

Run 
```shell
./scripts/setup.sh
```

### MNIST Experiment

To reproduce the results of the MNIST experiment, run

``` shell
./scripts/run_mnist_exps.sh mnist_ars
./scripts/run_mnist_exps.sh mnist_reinforce
./scripts/run_mnist_exps.sh mnist_sgd
```

To plot the results of the experiment, run

``` shell
python scripts/plot_mnist_results.py
```
### Linear Regression Experiments

To reproduce the results of the Linear regression experiments, run

``` shell
./scripts/run_linear_exps.sh linear_ars <d>
./scripts/run_linear_exps.sh linear_reinforce <d>
./scripts/run_linear_exps.sh linear_sgd <d>
./scripts/run_linear_exps.sh linear_naturalreinforce <d>
./scripts/run_linear_exps.sh linear_newton <d>
```
where `d` is the input dimensionality.

To plot the results of the experiments, run

``` shell
python scripts/plot_linear_results.py
```

## Dependencies
* Python 3.5.2
* PyTorch 0.3.1
* Numpy 1.14.0
* argparse 1.1
* ipdb
* matplotlib 2.2.2
