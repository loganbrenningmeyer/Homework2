from scipy.optimize import minimize
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

def plot_confusion_matrix(y_true, y_pred, title, filename):
    '''
    Plots a confusion matrix as a heatmap and saves it as filename
    '''
    cm = metrics.confusion_matrix(y_true, y_pred)

    cm_plot = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1, 1])

    cm_plot.plot()
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')

    
def main():
    '''
    Load Data
    '''
    # -- Training Data
    data_train = scipy.io.loadmat('data/training.mat')

    data1_train = data_train['Y1']
    data2_train = data_train['Y2']

    # -- Create input matrix X of shape (m, n)
    X_1_train = np.array(list(zip(data1_train[0], data1_train[1]))).astype(float)
    X_2_train = np.array(list(zip(data2_train[0], data2_train[1]))).astype(float)
    X_train = np.concatenate([X_1_train, X_2_train])

    # -- Create label vector y of shape (m, )
    y_1_train = -np.ones(shape=X_1_train.shape[0])
    y_2_train = np.ones(shape=X_2_train.shape[0])
    y_train = np.concatenate([y_1_train, y_2_train])

    # -- Testing Data
    data_test = scipy.io.loadmat('data/testing.mat')

    data1_test = data_test['Y1']
    data2_test = data_test['Y2']

    # -- Create input matrix X of shape (m, n)
    X_1_test = np.array(list(zip(data1_test[0], data1_test[1]))).astype(float)
    X_2_test = np.array(list(zip(data2_test[0], data2_test[1]))).astype(float)
    X_test = np.concatenate([X_1_test, X_2_test])

    # -- Create label vector y of shape (m, )
    y_1_test = -np.ones(shape=X_1_test.shape[0])
    y_2_test = np.ones(shape=X_2_test.shape[0])
    y_test = np.concatenate([y_1_test, y_2_test])

    '''
    Custom SVM
    '''
    # -- Get # samples and # features
    m, n = X_train.shape

    # -- Zero initialize weights and bias
    w = np.zeros(shape=n)
    b = 0.0  # Initialize bias as scalar
    theta = np.concatenate([w, [b]])

    # -- Define regularization parameter C
    C = 1.0

    # -- Minimize using scipy.optimize.minimize
    result = minimize(fun=objective,
                      x0=theta,
                      args=(X_train, y_train, C),
                      method='Newton-CG',
                      jac=jacobian,
                      hess=hessian,
                      options={'disp': True, 'maxiter': 1000})
    
    # -- Extract optimized parameters
    opt_params = result.x
    w_opt = opt_params[:-1]
    b_opt = opt_params[-1]

    # -- Compute metrics
    y_pred_train = np.where(X_train @ w_opt + b_opt >= 0, 1, -1)
    y_pred_test = np.where(X_test @ w_opt + b_opt >= 0, 1, -1)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    print(f"Custom SVM Training Accuracy: {accuracy_train:.2f}")

    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"Custom SVM Testing Accuracy: {accuracy_test:.2f}")

    plot_confusion_matrix(y_train, y_pred_train, 'Custom SVM (Train)', 'figs/confusion_matrix/custom_svm_train_cm.png')
    plot_confusion_matrix(y_test, y_pred_test, 'Custom SVM (Test)', 'figs/confusion_matrix/custom_svm_test_cm.png')


    '''
    SVM Library
    '''
    # -- Initialize SVM using an RBF kernel
    svm = SVC(kernel='linear', C=1.0)

    # -- Train svm
    svm.fit(X_train, y_train)

    # -- Compute metrics
    y_pred_train = svm.predict(X_train)
    y_pred_test = svm.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    print(f"SVM Library Training Accuracy: {accuracy_train:.2f}")

    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"SVM Library Testing Accuracy: {accuracy_test:.2f}")

    plot_confusion_matrix(y_train, y_pred_train, 'SVM Library (Train)', 'figs/confusion_matrix/svm_library_train_cm.png')
    plot_confusion_matrix(y_test, y_pred_test, 'SVM Library (Test)', 'figs/confusion_matrix/svm_library_test_cm.png')

    '''
    Custom SVM Plots
    '''
    # -- Initialize plots
    fig, axs = plt.subplots(1, 2, figsize=(10,5))

    # -- Training Plots

    # Plot data points
    axs[0].scatter(X_1_train[:, 0], X_1_train[:, 1], color='blue', marker='o', label='Class Y1 (-1)')
    axs[0].scatter(X_2_train[:, 0], X_2_train[:, 1], color='red', marker='s', label='Class Y2 (+1)')

    # Decision Boundary

    # Create a range of values for x1 (min x value - 1 to max x value + 1)
    x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    x1_values = np.linspace(x1_min, x1_max, 200)
    
    # Compute corresponding x2 values
    x2_values = (-w_opt[0] * x1_values - b_opt) / w_opt[1]
    axs[0].plot(x1_values, x2_values, color='green', linestyle='--', label='Decision Boundary')

    # -- Testing Plots

    # Plot data points
    axs[1].scatter(X_1_test[:, 0], X_1_test[:, 1], color='blue', marker='o', label='Class Y1 (-1)')
    axs[1].scatter(X_2_test[:, 0], X_2_test[:, 1], color='red', marker='s', label='Class Y2 (+1)')

    # Decision Boundary

    # Create a range of values for x1 (min x value - 1 to max x value + 1)
    x1_min, x1_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    x1_values = np.linspace(x1_min, x1_max, 200)
    
    # Compute corresponding x2 values
    x2_values = (-w_opt[0] * x1_values - b_opt) / w_opt[1]
    axs[1].plot(x1_values, x2_values, color='green', linestyle='--', label='Decision Boundary')

    # -- Adjust the plots
    axs[0].set_xlabel('x1')
    axs[0].set_ylabel('x2')
    axs[0].set_title('Custom SVM Decision Boundary (Train)')

    axs[1].set_xlabel('x1')
    axs[1].set_ylabel('x2')
    axs[1].set_title('Custom SVM Decision Boundary (Test)')

    axs[0].legend(loc='lower right')
    axs[1].legend(loc='lower right')

    axs[0].grid()
    axs[1].grid()

    axs[0].set_aspect('equal', adjustable='box')
    axs[1].set_aspect('equal', adjustable='box')

    axs[0].set_xlim(x1_min, x1_max)
    axs[1].set_xlim(x1_min, x1_max)

    axs[0].set_ylim(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1)
    axs[1].set_ylim(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1)

    # -- Save the plot
    fig.savefig('figs/decision_boundary/custom_svm.png', bbox_inches='tight')

    '''
    SVM Library Plots
    '''
    # -- Extract weights and bias from SVM
    w_library = svm.coef_[0]
    b_library = svm.intercept_[0]

    # -- Initialize plots
    fig, axs = plt.subplots(1, 2, figsize=(10,5))

    # -- Training Plots

    # Plot data points
    axs[0].scatter(X_1_train[:, 0], X_1_train[:, 1], color='blue', marker='o', label='Class Y1 (-1)')
    axs[0].scatter(X_2_train[:, 0], X_2_train[:, 1], color='red', marker='s', label='Class Y2 (+1)')

    # Decision Boundary

    # Create a range of values for x1 (min x value - 1 to max x value + 1)
    x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    x1_values = np.linspace(x1_min, x1_max, 200)
    
    # Compute corresponding x2 values
    x2_values = (-w_library[0] * x1_values - b_library) / w_library[1]
    axs[0].plot(x1_values, x2_values, color='green', linestyle='--', label='Decision Boundary')

    # -- Testing Plots

    # Plot data points
    axs[1].scatter(X_1_test[:, 0], X_1_test[:, 1], color='blue', marker='o', label='Class Y1 (-1)')
    axs[1].scatter(X_2_test[:, 0], X_2_test[:, 1], color='red', marker='s', label='Class Y2 (+1)')

    # Decision Boundary

    # Create a range of values for x1 (min x value - 1 to max x value + 1)
    x1_min, x1_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    x1_values = np.linspace(x1_min, x1_max, 200)
    
    # Compute corresponding x2 values
    x2_values = (-w_library[0] * x1_values - b_library) / w_library[1]
    axs[1].plot(x1_values, x2_values, color='green', linestyle='--', label='Decision Boundary')

    # -- Adjust the plots
    axs[0].set_xlabel('x1')
    axs[0].set_ylabel('x2')
    axs[0].set_title('SVM Library Decision Boundary (Train)')

    axs[1].set_xlabel('x1')
    axs[1].set_ylabel('x2')
    axs[1].set_title('SVM Library Decision Boundary (Test)')

    axs[0].legend(loc='lower right')
    axs[1].legend(loc='lower right')

    axs[0].grid()
    axs[1].grid()

    axs[0].set_aspect('equal', adjustable='box')
    axs[1].set_aspect('equal', adjustable='box')

    axs[0].set_xlim(x1_min, x1_max)
    axs[1].set_xlim(x1_min, x1_max)

    axs[0].set_ylim(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1)
    axs[1].set_ylim(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1)

    # -- Save the plot
    fig.savefig('figs/decision_boundary/svm_library.png', bbox_inches='tight')


if __name__ == "__main__":
    main()