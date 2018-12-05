# Exploration in Action Space

## Citation

If you use this code, please cite the following paper

> Anirudh Vemula, Wen Sun, J. Andrew Bagnell. **Exploration in Action Space**. *Learning and Inference in Robotics: Integrating Structure, Priors and Models workshop. Robotics: Science and Systems (RSS). 2018*


## Reproducing Experiments

### Setup

Run 
```shell
pip install -r requirements.txt
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
### Linear Quadratic Regulator (deprecated)
*Use the LQR experiments script in this repository. This script is deprecated*
To reproduce the results of the LQR experiments, run

```shell
python reinforce/lqr_reinforce.py
python ars/run_ars_script.py
```

## Dependencies
* Python 3.5.2
* OpenAI Gym 0.10.4
* PyTorch 0.3.1
* Numpy 1.14.0
* argparse 1.1
* ipdb
* matplotlib 2.2.2
