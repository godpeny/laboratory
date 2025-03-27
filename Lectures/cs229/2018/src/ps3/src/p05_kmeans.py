from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import random

def main(peppers, K):
    path = "../data/" + peppers
    A = imread(path)
    c, r, _ = A.shape

    new_A = np.matrix.copy(A)
    C = np.ones((c,r)) # save index for the centroid
    J = 1000000 # the sum of squared distances between x and centroid
    eps = 1e-3   # Convergence threshold

    # plt.imshow(A);plt.show()

    ### Start ###
    # set each cluster centroid to the (r; g; b)-values of a randomly chosen pixel in the image.
    # then replace each pixel's (r; g; b) values with the value of the closest cluster centroid.
    num = random.sample(range(0, r * c), K)
    means = np.array([np.floor_divide(num, c), np.mod(num, c)]).T # init mean

    for it in range(30):
        print(it + 1)
        prev_A = np.matrix.copy(new_A)
        prev_means = np.matrix.copy(means)

        # (1) color the points
        print("  (1) color the points")
        for i in range(c):
            for j in range(r):
                mean_idx = -1; min_norm_value = 1000000

                for idx, mean in enumerate(means):
                    color = A[i][j]
                    mean_color = A[mean[0]][mean[1]]
                    norm = np.sqrt(np.sum((color - mean_color) ** 2))
                    if norm < min_norm_value:
                        mean_idx = idx
                        min_norm_value=norm

                new_A[i][j] = A[means[mean_idx][0]][means[mean_idx][1]]
                C[i][j] = mean_idx

        # (2) update mean
        print("  (2) update mean")
        new_means = np.matrix.copy(means)
        for idx, mean in enumerate(means):
            numerator = [0,0]
            denominator = 0

            for i in range(c):
                for j in range(r):
                    if sum((new_A[i][j]-new_A[mean[0]][mean[1]])**2) == 0:
                        denominator += 1
                        numerator[0] += i; numerator[1] += j

            new_means[idx][0] = np.floor(numerator[0] / denominator)
            new_means[idx][1] = np.floor(numerator[1] / denominator)

        means = new_means

        # (3) plot
        print("  (3) plot")
        plt.imshow(new_A / 255);plt.show()

        # (4) check convergence
        print("  (4) check convergence")
        cur_J = 0
        for i in range(c):
            for j in range(r):
                color = new_A[i][j]
                mean = means[int(C[i][j])]

                mean_color = new_A[mean[0]][mean[1]]
                cur_J += np.sqrt(np.sum((color - mean_color) ** 2))

        diff = np.abs(J - cur_J)
        print("from ", J, " to ", cur_J, " diff = ", diff)
        if J < cur_J:
            new_A = np.matrix.copy(prev_A)
            means = np.matrix.copy(prev_means)
            continue
        elif diff > eps:
            J = cur_J
        else:
            break

    plt.imshow(A);plt.show()


if __name__ == '__main__':
    pl_model = "peppers-large.tiff" # (512, 512, 3): r,g,b
    ps_model = "peppers-small.tiff" # (128, 128, 3): r,g,b
    num_clusters = 16

    main(peppers=ps_model, K=num_clusters)
