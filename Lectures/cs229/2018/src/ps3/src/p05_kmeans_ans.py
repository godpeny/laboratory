from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import random


def main(peppers, K):
    path = "../data/" + peppers
    A = imread(path)  # A has shape (height, width, channels)
    height, width, channels = A.shape

    # Make a copy of the original image for updating cluster colors
    new_A = A.copy()
    # This will hold the cluster assignments for each pixel (an integer per pixel)
    C = np.zeros((height, width), dtype=int)

    # Convergence tolerance
    eps = 1e-3
    max_iter = 30

    # ----- Initialization of centroids -----
    # Choose K random pixel indices from the image
    indices = random.sample(range(height * width), K)

    # Convert flat indices to 2D (row, col) coordinates.
    rows = np.floor_divide(indices, width)
    cols = np.mod(indices, width)
    # Get the color of these pixels and use as initial centroids (as float for averaging)
    centroids = np.array([A[r, c] for r, c in zip(rows, cols)]).astype(float)


    print("Initial centroids (colors):")
    print(centroids)

    # ----- K-means iteration -----
    for it in range(max_iter):
        print("Iteration:", it + 1)
        prev_centroids = centroids.copy()

        # (1) Assignment step: assign each pixel to the nearest centroid (in color space)
        for i in range(height):
            for j in range(width):
                pixel_color = A[i, j]  # original color at pixel (i,j)
                # Compute distance from pixel_color to each centroid
                distances = [np.linalg.norm(pixel_color - centroid) for centroid in centroids]
                # Get index of the closest centroid
                cluster_idx = np.argmin(distances)
                # Record assignment and update new_A to show the centroid color
                C[i, j] = cluster_idx
                new_A[i, j] = centroids[cluster_idx]

        # (2) Update step: recalc centroids as mean of colors of all pixels assigned to them
        for k in range(K):
            # Create a mask of all pixels assigned to cluster k
            mask = (C == k)
            if np.sum(mask) > 0:
                # Compute mean color over all pixels in cluster k (using the original image A)
                centroids[k] = A[mask].mean(axis=0)
            else:
                # If no pixel was assigned to this cluster, you might reinitialize it randomly.
                centroids[k] = A[random.randint(0, height - 1), random.randint(0, width - 1)]

        # (3) Convergence check: if centroids change very little, then break
        diff = np.linalg.norm(centroids - prev_centroids)
        print("Centroid change:", diff)
        if diff < eps:
            print("Converged")
            break

        # Optionally, display the intermediate clustered image
        plt.imshow(new_A / 255)
        plt.title(f"Iteration {it + 1}")
        plt.axis('off')
        plt.show()

    # Final result plot
    plt.imshow(new_A / 255)
    plt.title("Final Clustered Image")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # For testing purposes, use the small peppers image.
    ps_model = "peppers-small.tiff"  # Update the path as needed
    num_clusters = 16
    main(peppers=ps_model, K=num_clusters)