from matplotlib.image import imread
import matplotlib.pyplot as plt

def main(peppers):
    path = "../data/" + peppers
    A = imread(path)
    print(A.shape)

    plt.imshow(A);plt.show()

    ### Start ###
    # set each cluster centroid to the (r; g; b)-values of a randomly chosen pixel in the image.
    # and replace each pixel's (r; g; b) values with the value of the closest cluster centroid.


if __name__ == '__main__':
    pl_model = "peppers-large.tiff" # (512, 512, 3): r,g,b
    ps_model = "peppers-small.tiff" # (128, 128, 3): r,g,b
    main(peppers=ps_model)
