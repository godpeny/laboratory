import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    m,n = x.shape
    idx = np.random.permutation(m)
    samples = np.array(np.array_split(x[idx], K)) # (K, m/K, n)
    mu = np.mean(samples, axis=1) # (K,n)

    sigma = []
    for i in range(K):
        # rowvar=False: each row(m) is an observation (sample) and each column(n) is a feature (variable).
        # rowvar=True: each row(n) is a feature and each row(m) is an observation.
        sample_sigma = np.cov(samples[i], rowvar=False) # (n, n)
        sigma.append(sample_sigma)

    sigma = np.array(sigma) # (K, n, n)

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full(K, 1/K) # 0.25

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.ones((m,K)) / K

    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        print(it)
        it+=1

        m,n = x.shape

        # (1) E-step: Update your estimates in w
        numerator = []
        denominator = 0
        for i in range(K):
            mu_j = mu[i] # (n,)
            sigma_j = sigma[i] # (n,n)
            phi_j = phi[i] # (1,)

            nu_first_term = 1 / (np.power((2 * np.pi),(n/2)) * np.power(np.linalg.det(sigma_j),1/2)) # (1,)
            nu_exp_term = np.exp((-1/2) * np.einsum('ij,jk,ik->i', (x-mu_j), np.linalg.inv(sigma_j) , (x-mu_j))) # (m,1)
            nu = nu_first_term * nu_exp_term * phi_j # (m,1)

            numerator.append(nu)
            denominator += nu

        w = np.array(numerator/denominator).T # (m, K)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        _phi = np.average(w, axis=0)
        _mu = [] # (K,n)
        _sigma = [] # (K,n,n)

        for i in range(K):
            mu_j = mu[i]  # (n,)
            w_j = w[:, i]  # (m,)

            _m = np.einsum('m,mn->n', w_j, x) / np.sum(w_j)
            _mu.append(_m)

            _s = np.einsum('m,mn->mn', w_j, (x-mu_j)).T @ (x-mu_j) / np.sum(w_j)
            _sigma.append(_s)

        _mu = np.array(_mu)
        _sigma = np.array(_sigma)

        phi = _phi
        mu = _mu
        sigma = _sigma

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        _ll = 0
        for i in range(K):
            mu_j = mu[i]  # (n,)
            sigma_j = sigma[i]  # (n,n)
            phi_j = phi[i]  # (1,)
            w_j = w[:, i]  # (m,)

            first_term = 1 / (np.power((2 * np.pi), (n / 2)) * np.power(np.linalg.det(sigma_j), 1 / 2))  # (1,)
            exp_term = np.exp((-1 / 2) * np.einsum('ij,jk,ik->i', (x - mu_j), np.linalg.inv(sigma_j) , (x - mu_j)))  # (m,1)

            l = np.log(first_term * exp_term * phi_j / w_j) # (m,1)
            l = np.sum(w_j * l) #(1,)
            _ll += l

        prev_ll = ll
        ll = _ll

        print("***")
        print(prev_ll)
        print(ll)
        if prev_ll is not None and ll is not None:
            print(np.abs(prev_ll - ll))
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        print(it)
        it += 1

        m, n = x.shape
        m_tilde, n_tilde = x_tilde.shape

        # (1) E-step: Update your estimates in w
        numerator = []
        denominator = 0
        for i in range(K):
            mu_j = mu[i]  # (n,)
            sigma_j = sigma[i]  # (n,n)
            phi_j = phi[i]  # (1,)

            nu_first_term = 1 / (np.power((2 * np.pi), (n / 2)) * np.power(np.linalg.det(sigma_j), 1 / 2))  # (1,)
            nu_exp_term = np.exp(
                (-1 / 2) * np.einsum('ij,jk,ik->i', (x - mu_j), np.linalg.inv(sigma_j), (x - mu_j)))  # (m,1)
            nu = nu_first_term * nu_exp_term * phi_j  # (m,1)

            numerator.append(nu)
            denominator += nu

        w = np.array(numerator / denominator).T  # (m, K)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        _phi = np.zeros(phi.shape)
        _mu = np.zeros(mu.shape)
        _sigma = np.zeros(sigma.shape)

        for i in range(K):
            w_j = w[:,i] # (m,)
            mu_j = mu[i] # (n,)

            _phi_j = np.sum(w_j, axis=0) + (alpha * np.sum(z == i))
            _phi_j /= (m + (alpha * m_tilde))
            _phi[i] = _phi_j

            x_tilde_j = x_tilde[np.where(np.array([int(x) for x in np.squeeze(z)]) == i)]

            _m = (np.einsum('m,mn->n', w_j, x)
                  + (alpha * np.sum(x_tilde_j,axis=0)))
            _m /= (np.sum(w_j) + (alpha * np.sum(z == i)))
            _mu[i] = mu_j = _m

            _s = (np.einsum('m,mn->mn',w_j, (x-mu_j)).T @ (x-mu_j)) + (alpha * ((x_tilde_j-mu_j).T @ (x_tilde_j-mu_j)))
            _s /= (np.sum(w_j) + (alpha * np.sum(z == i)))
            _sigma[i] = _s

        phi = _phi
        mu = _mu
        sigma = _sigma

        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        _ll = 0
        for i in range(K):
            mu_j = mu[i]  # (n,)
            sigma_j = sigma[i]  # (n,n)
            phi_j = phi[i]  # (1,)
            w_j = w[:, i]  # (m,)

            unsup_first_term = 1 / (np.power((2 * np.pi), (n / 2)) * np.power(np.linalg.det(sigma_j), 1 / 2))  # (1,)
            unsup_exp_term = np.exp((-1 / 2) * np.einsum('ij,jk,ik->i', (x - mu_j), np.linalg.inv(sigma_j) , (x - mu_j)))  # (m,1)

            l_unsup = np.log(unsup_first_term * unsup_exp_term * phi_j / w_j) # (m,1)
            l_unsup = np.sum(w_j * l_unsup) #(1,)

            x_tilde_j = x_tilde[np.where(np.array([int(x) for x in np.squeeze(z)]) == i)]

            l_sup = np.sum(np.log(phi_j * (np.power((2 * np.pi), (-n_tilde / 2))) * np.power(np.linalg.det(sigma_j), -1/2) 
                           * np.exp(-1/2*(np.einsum('ij,jk,ik->i', (x_tilde_j - mu_j), np.linalg.inv(sigma_j) , (x_tilde_j - mu_j))))), axis=0)

            l = l_unsup + (alpha * l_sup)
            _ll += l

        prev_ll = ll
        ll = _ll

        print("***")
        print(prev_ll)
        print(ll)
        if prev_ll is not None and ll is not None:
            print(np.abs(prev_ll - ll))

        # *** END CODE HERE ***

    return w

# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        # main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
