from scipy.optimize import minimize
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def objective(params: np.ndarray,
              X: np.ndarray,
              y: np.ndarray,
              C: float):
    '''
    The primal objective function for an SVM
    -m: number of samples
    -n: number of features

    Parameters:
    - params: array of weights (params[:-1]) and the bias (params[-1]) of shape (n + 1,)
    - X: Input features vectors of shape (m, n)
    - y: Labels of shape (m, )
    - C: Regularization term (scalar)

    Objective function:
        => \frac{1}{2}\|\mathbf{w}\|^2+C\sum_{i=1}^m(max(0,1-y_i(\mathbf{w}^T\mathbf{x}_i+b))^2)
    '''
    w = params[:-1]
    b = params[-1]

    reg_term = 0.5 * np.linalg.norm(w)**2
    hinge_loss = C * np.sum(np.maximum(0, 1 - y * (X @ w + b))**2)

    obj = reg_term + hinge_loss

    return obj

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
    dw = w - 2 * C * np.sum(y * X * z * I, axis=0)

    # -- df/db: returns scalar
    db = -2 * C * np.sum(y * z * I)

    gradient = np.concatenate([dw, [db]])

    return gradient

def hessian(params: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
            C: float):
    '''
    Returns the Hessian of the primal optimization function
    - m: number of samples
    - n: number of features

    Parameters:
    - params: array of weights (params[:-1]) and the bias (params[-1]) of shape (n + 1,)
    - X: Input features vectors of shape (m, n)
    - y: Labels of shape (m, )
    - C: Regularization term (scalar)

    where: 
        => z_i=1-y_i(\mathbf{w}^T\mathbf{x_i}+b)
        => I_i=z_i>0

    d^2f/dw^2:
        => I+2C\sum_{i=1}^m(\mathbf{x}_i\mathbf{x}_i^TI_i)

    d^2f/db^2:
        => 2C\sum_{i=1}^mI_i

    d^2f/dwdb:
        => 2C\sum_{i=1}^m\mathbf{x}_iI_i

    d^2f/dbdw:
        => 2C\sum_{i=1}^m\mathbf{x}_iI_i
    '''
    # m: number of samples
    # n: number of features
    m, n = X.shape

    w = params[:-1]
    b = params[-1]

    # -- Compute z
    z = 1 - y * (X @ w + b)

    I = (z > 0).astype(float)

    # -- Reshape y, z, and I for matrix multiplication
    y = y.reshape(m, 1)
    z = z.reshape(m, 1)
    I = I.reshape(m, 1)

    # -- d2f/dw2 (n, n)
    dww = np.identity(n) + 2 * C * (X.T @ (X * I))

    # -- d2f/db2 (scalar)
    dbb = 2 * C * np.sum(I)

    # -- d2f/dwdb (n, 1)
    dwb = 2 * C * (X.T @ I)

    # -- d2f/dbdw (1, n)
    dbw = dwb.T

    # -- Build Hessian matrix of 2nd derivatives
    hessian_mat = np.zeros(shape=(n + 1, n + 1))

    hessian_mat[:-1, :-1] = dww
    hessian_mat[:-1,  -1] = dwb.flatten()
    hessian_mat[ -1, :-1] = dbw.flatten()
    hessian_mat[ -1,  -1] = dbb

    return hessian_mat
    
def main():
    # -- Load data
    data = scipy.io.loadmat('data/training.mat')

    data1 = data['Y1']
    data2 = data['Y2']

    # -- Create input matrix X of shape (m, n)
    X_1 = np.array(list(zip(data1[0], data1[1]))).astype(float)
    X_2 = np.array(list(zip(data2[0], data2[1]))).astype(float)
    X = np.concatenate([X_1, X_2])

    # -- Create label vector y of shape (m, )
    y_1 = np.ones(shape=X_1.shape[0])
    y_2 = -np.ones(shape=X_2.shape[0])
    y = np.concatenate([y_1, y_2])

    # -- Get # samples and # features
    m, n = X.shape

    # -- Zero initialize weights and bias
    w = np.zeros(shape=n)
    b = 0.0  # Initialize bias as scalar
    theta = np.concatenate([w, [b]])

    # -- Define regularization parameter C
    C = 1.0

    # -- Minimize using scipy.optimize.minimize
    result = minimize(fun=objective,
                      x0=theta,
                      args=(X, y, C),
                      method='Newton-CG',
                      jac=jacobian,
                      hess=hessian,
                      options={'disp': True, 'maxiter': 1000})
    
    print(result)
    
    # -- Extract optimized parameters
    opt_params = result.x
    w_opt = opt_params[:-1]
    b_opt = opt_params[-1]

    # -- Plot the data points
    plt.figure(figsize=(8, 6))

    # Plot class Y1
    plt.scatter(X_1[:, 0], X_1[:, 1], color='blue', marker='o', label='Class Y1 (+1)')

    # Plot class Y2
    plt.scatter(X_2[:, 0], X_2[:, 1], color='red', marker='s', label='Class Y2 (-1)')

    # -- Plot the decision boundary
    # The decision boundary is w1*x1 + w2*x2 + b = 0
    # Solve for x2: x2 = (-w1*x1 - b) / w2

    # Create a range of values for x1
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_values = np.linspace(x1_min, x1_max, 200)
    
    # Compute corresponding x2 values
    x2_values = (-w_opt[0] * x1_values - b_opt) / w_opt[1]
    plt.plot(x1_values, x2_values, color='green', linestyle='--', label='Decision Boundary')

    # -- Formatting the plot
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()