from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import random

def main(peppers, K):
    path = "../data/" + peppers
    A = imread(path)
    height, width, _ = A.shape

    new_A = np.matrix.copy(A)
    C = np.ones((height,width)) # save index for the centroid
    J = 1000000 # the sum of squared distances between x and centroid
    eps = 1e-3   # Convergence threshold
    max_iter = 30

    # plt.imshow(A);plt.show()

    ### Start ###
    # set each cluster centroid to the (r; g; b)-values of a randomly chosen pixel in the image.
    # then replace each pixel's (r; g; b) values with the value of the closest cluster centroid.

    # (0) initialize the centroids
    num = random.sample(range(0, height * width), K)
    centroids = []

    for idx, n in enumerate(num):
        centroids.append(A[np.floor_divide(num[idx], height)][np.mod(num[idx], height)])
    # centroid should be float for calculating precisely
    centroids = np.array(centroids).astype(float) # (K, 3) - [r,g,b]

    for it in range(max_iter):
        print(it + 1)

        # (1) color the points (assignment)
        # It is very important to note that in a proper k-means implementation for color quantization,
        # the assignment step and the update step should use the original pixel colors to compute the new centroids.
        # If you update the centroids from the already quantized image (where each pixel is exactly the centroid's color),
        # then nothing will change—the centroids would remain the same.
        # That’s why you typically maintain two copies: A and new_A
        # - A: The original image pixel colors remain unchanged.
        # - new_A: updated to display the centroid colors for visualization purposes.

        print("  (1) color the points")
        for i in range(height):
            for j in range(width):
                min_idx = -1; min_norm_value = 1000000
                min_rgb = [0,0,0]

                for idx, centroid in enumerate(centroids):
                    color = A[i][j]
                    mean_color = centroid
                    norm = np.sqrt(np.sum((color - mean_color) ** 2))
                    if norm < min_norm_value:
                        min_idx = idx
                        min_norm_value=norm
                        min_rgb = mean_color

                new_A[i][j] = min_rgb
                C[i][j] = min_idx

        # (2) update mean (update)
        # Compute the new centroid as the mean of the original pixel values in that cluster.
        # This allows the centroids to shift toward the true mean color of the pixels,
        # even though the visualization might show the current centroid values.
        print("  (2) update mean")
        for idx, centroid in enumerate(centroids):
            numerator = [0,0,0]
            denominator = 0

            for i in range(height):
                for j in range(width):
                    if sum((new_A[i][j]-centroid)**2) == 0:
                        denominator += 1
                        numerator[0] += A[i][j][0]
                        numerator[1] += A[i][j][1]
                        numerator[2] += A[i][j][2]

            if denominator > 0:
                centroids[idx][0] = numerator[0] / denominator
                centroids[idx][1] = numerator[1] / denominator
                centroids[idx][2] = numerator[2] / denominator

        # (3) plot
        print("  (3) plot")
        plt.imshow(new_A);plt.show()

        # (4) check convergence
        print("  (4) check convergence")
        cur_J = 0
        for i in range(height):
            for j in range(width):
                color = new_A[i][j]
                centroid = centroids[int(C[i][j])]
                cur_J += np.sqrt(np.sum((color - centroid) ** 2))

        diff = np.abs(J - cur_J)
        print("from ", J, " to ", cur_J, " diff = ", diff)
        if J < cur_J:
           break;
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
    main(peppers=pl_model, K=num_clusters)

    # p05b
    # originally, each pixel's (r,g,b) was 3*8 bits(0~255, 0~255, 0~255),
    # but now compressed into 4 bits (0~15 = 16 colors).
    # so the compression factor is 24/4 = 6.
