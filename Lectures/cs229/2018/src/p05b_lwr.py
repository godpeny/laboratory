import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    lwr = LocallyWeightedLinearRegression(tau=0.5)
    lwr.fit(x_train, y_train)

    # Get MSE value on the validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = lwr.predict(x_eval)
    mse = np.mean((y_eval - y_pred)**2)
    print(mse)

    # Plot validation predictions on top of training set
    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_eval, y_pred, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    # No need to save predictions

    # Plot data
    plt.show()
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m = 0
        m_1, n = self.x.shape
        m_2, _ = x.shape

        if m_1 < m_2:
            m=m_1
        else:
            m=m_2

        x = x[:m,]
        self.x = self.x[:m,]
        self.y = self.y[:m,]

        sub = self.x - np.reshape(x, (m,-1,n))  # (m,m,n)
        euclid_norm_2d = np.sum(np.square(sub), axis=2)  # np.linalg.norm(self.x - np.reshape(x, (m,-1,n)), axis=2, ord=2)**2 => (m,m)
        w = np.exp(-euclid_norm_2d/(2*self.tau**2))  # (m,m) => [x[0]-self.x, x[1]-self.x, ... x[m]-self.x]
        W = np.apply_along_axis(np.diag, axis=1, arr=w)  # (m,m,m) diagonal matrix => [0][][] = diag(x[0]-self.x), [1][][] = diag(x[1]-self.x), ... [m][][] = diag(x[m]-self.x)
        self.theta = np.linalg.inv((self.x.T @ W @ self.x)) @ self.x.T @ W @ self.y
        # (2,200) @ (200,200,200) @ (200,2) = (200,2,200) @ (200,2) = (200,2,2)
        # (200,2,2) @ (2,200) @ (200,200,200) @ (200,) =
        # (200,2,200) @ (200,200,200) @ (200,) = (200,2,200) @ (200,) =
        # (200,2)

        y_pred = np.einsum('ij,ij->i', x, self.theta)
        # (200,2) (200,2) -> [[0][a_0, b_0], [1][a_1, b_1], ... [m][a_m, b_m]], [[0][c_0, d_0], [1][c_1, d_1], ... [m][c_m, d_m]]
        # [[0][(a_0 * c_0) + (b_0 * d_0)], [1][(a_1 * c_1) + (b_1 * d_1)], ... [m][(a_m * c_m) + (b_m * d_m)]]
        # (200,)

        """
        # another approach (basically same as original approach) 
        
        y_pred_2 = np.zeros(m)
        W_2 = np.zeros([m,m,m])
        theta_2 = np.zeros([m,n])

        for i in range(m):
            w = np.exp(-np.sum((self.x - x[i])**2, axis=1) / (2 * self.tau**2)) # (200,)
            W = np.diag(w)  # (200,200)
            theta = np.linalg.inv(self.x.T.dot(W).dot(self.x)).dot(self.x.T).dot(W).dot(self.y)  # (2,)
            # (2,200) @ (200,200) @ (200,2) = (2,200) @ (200,2) = (2,2)
            # (2,2) @ (2,200) @ (200,200) @ (200,) =
            # (2,200) @ (200,200) @ (200,) = (2,200) @ (200,) =
            # (2,)

            y_pred_2[i] = theta.T @ x[i]
            W_2[i] = W
            theta_2[i] = theta
        """
        return y_pred
        # *** END CODE HERE ***
