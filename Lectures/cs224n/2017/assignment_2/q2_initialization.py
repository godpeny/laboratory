import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

def xavier_weight_init():
    """Returns function that creates random tensor.

    The specified function will take in a shape (tuple or 1-d array) and
    returns a random tensor of the specified shape drawn from the
    Xavier initialization distribution.

    Hint: You might find tf.random_uniform useful.
    """
    def _xavier_initializer(shape, **kwargs):
        """Defines an initializer for the Xavier distribution.
        Specifically, the output should be sampled uniformly from [-epsilon, epsilon] where
            epsilon = sqrt(6) / <sum of the sizes of shape's dimensions>
        e.g., if shape = (2, 3), epsilon = sqrt(6 / (2 + 3))

        This function will be used as a variable initializer.

        Args:
            shape: Tuple or 1-d array that species the dimensions of the requested tensor.
        Returns:
            out: tf.Tensor of specified shape sampled from the Xavier distribution.
        """
        ### YOUR CODE HERE
        e = np.sqrt(6) / np.sqrt(sum(shape))
        out = tf.random.uniform(shape=shape, minval=-e, maxval=e)
        ### END YOUR CODE
        return out
    # Returns defined initializer function.
    return _xavier_initializer


def test_initialization_basic():
    """Some simple tests for the initialization.
    """
    print("Running basic tests...")
    xavier_initializer = xavier_weight_init()
    shape = (1,)
    xavier_mat = xavier_initializer(shape)
    assert xavier_mat.get_shape() == shape

    shape = (1, 2, 3)
    xavier_mat = xavier_initializer(shape)
    assert xavier_mat.get_shape() == shape
    print("Basic (non-exhaustive) Xavier initialization tests pass")


if __name__ == "__main__":
    test_initialization_basic()
