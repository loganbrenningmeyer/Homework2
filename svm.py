from scipy.optimize import minimize
import numpy as np

def jacobian(params: np.ndarray, 
             X: np.ndarray, 
             y: np.ndarray, 
             C: float):
    '''
    Returns the Jacobian (gradient) of the primal optimization function
    - m: number of samples
    - n: number of features

    Parameters:
    - params: array of weights (params[:-1]) and the bias (params[-1]) of shape (n + 1,)
    - X: Input features vectors of shape (m, n)
    - y: Labels of shape (m, )
    - C: Regularization term (scalar)

    where: 
        => z_i=1-y_i(\mathbf{w}^T\mathbf{x}+b)
        => I_i=z_i>0

    df/dw:
        => \frac{\partial f}{\partial \mathbf{w}}=\mathbf{w}-2C\sum_{i=1}^m(y_i\mathbf{x}_iz_iI_i)

    df/db:
        => \frac{\partial f}{\partial b}=-2C\sum_{i=1}^my_iz_iI_i
    '''
    # m: Number of samples
    m = X.shape[0]

    w = params[:-1]
    b = params[-1]

    # -- Computes z_i for each sample i=1->l, returning vector of shape (m, )
    z = 1 - y * (X @ w + b)

    I = (z > 0).astype(float)

    # -- Reshape y, z, and I for matrix multiplication
    y = y.reshape(m, 1)
    z = z.reshape(m, 1)
    I = I.reshape(m, 1)

    # -- df/dw: returns vector of shape (n, )
    dfdw = w - 2 * C * np.sum(y * X * z * I, axis=0)

    # -- df/db: returns scalar
    dfdb = -2 * C * np.sum(y * z * I)

    gradient = np.concatenate([dfdw, [dfdb]])

    return gradient

def hessian(params: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
            C: float):
    '''
    Returns the Hessian of the primal optimization function
    - m: number of samples
    - n: number of features
    '''

