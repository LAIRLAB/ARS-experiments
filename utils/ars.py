import numpy as np


def flatten_params(params):
    '''
    params (list): parameters of the model

    flatten_params (np array of size num_params): flattened parameters
    '''
    flatten_params = np.concatenate([x.flatten() for x in params])
    return flatten_params

def reconstruct_params(perturbed_params, param_shapes, param_num_elements):
    '''
    perturbed_params (np array of size num_params): perturbed get_parameters
    param_shapes (list): shapes of parameters in the model
    param_num_elements (list): number of parameters in the model

    reconstruct_params (list): reconstructed parameters
    '''
    reconstruct_params = []
    c = 0
    for i in range(len(param_shapes)):
        num_elements = param_num_elements[i]
        shape_params = param_shapes[i]
        extracted_params = perturbed_params[c:c+num_elements]
        reconstruct_params.append(np.reshape(extracted_params, shape_params))
        c += num_elements
    return reconstruct_params

def sample_directions(num_directions, num_params):
    '''
    num_directions (int) : number of directions to sample
    num_params (int) : size of the direction vector

    directions (2D array of size num_directionsxnum_params ): each row is a direction vector
    '''
    directions = [np.random.normal(size=num_params) for _ in range(num_directions)]
    return np.array(directions)

def perturb_parameters(params, directions, nu):
    '''
    params (np array of size num_params): parameter vector
    directions (np array of size num_directionsxnum_params) : direction vectors
    nu (int): perturbation length

    perturbed params (np array of size 2xnum_directionsxnum_params)
    '''
    assert nu > 0
    perturbed_params = np.array([params + nu * directions, params - nu * directions])
    return perturbed_params

def get_top_directions_returns(returns, directions, num_top):
    '''
    returns (np array of size num_directionsx2): returns of each direction (+ve and -ve)
    directions (np array of size num_directionsxnum_params): direction vectors
    num_top (int): number of top directions required

    top_directions (np array of size num_topxnum_params): top direction vectors
    top_returns (np array of size num_topx2): top returns
    '''
    max_returns = returns.max(axis=1)
    top_indices = np.argsort(max_returns)[-num_top:]
    return directions[top_indices, :], returns[top_indices, :]

def update_parameters(params, stepsize, top_returns, top_directions):
    '''
    params (np array of size num_params): parameter vector
    stepsize (float): step size of the update
    top_returns (np array of size num_topx2): top returns
    top_directions (np array of size num_topxnum_params): top direction vectors

    new_params (np array of size num_params): updated params
    '''
    num_top = top_directions.shape[0]
    std_returns = np.std(top_returns)
    if std_returns == 0:
        std_returns = 1.0
    diff_returns = top_returns[:, 0] - top_returns[:, 1]  # assuming that the first column is the +ve direction and the second column is the -ve direction
    params_update = np.dot(top_directions.T, diff_returns)
    new_params = params + (stepsize/(num_top*std_returns)) * params_update
    return new_params


class RunningStat(object):

    def __init__(self, shape=None):
        self._n = 0
        self._M = np.zeros(shape, dtype=np.float64)
        self._S = np.zeros(shape, dtype=np.float64)
        self._M2 = np.zeros(shape, dtype=np.float64)

    def push_batch(self, batch):
        batch_size = batch.shape[0]
        for i in range(batch_size):
            self.push(batch[i])

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape

        n1 = self._n
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            delta = x - self._M
            self._M[...] += delta / self._n
            delta2 = x - self._M
            self._S[...] += delta * delta2

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M if self._n > 1 else 0.0

    @property
    def var(self):
        ret_var = self._S / (self._n - 1) if self._n > 1 else 1.0
        if self._n > 1:
            ret_var[ret_var == 0] = 1
        return ret_var

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape
