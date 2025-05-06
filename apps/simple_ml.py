"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, 'rb') as f:
        # 跳过前16个字节（魔数、数量、行数、列数）
        _ = np.frombuffer(f.read(16), dtype=np.uint8)
        # 读取图像数据
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        # 将数据转换为二维数组
        X = data.reshape(-1, 28 * 28).astype(np.float32) / 255.0

    with gzip.open(label_filename, 'rb') as f:
        # 跳过前8个字节（魔数、数量）
        _ = np.frombuffer(f.read(8), dtype=np.uint8)
        # 读取标签数据
        buf = f.read()
        y = np.frombuffer(buf, dtype=np.uint8)
    
    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    exp_Z = ndl.exp(Z)

    sum_exp_Z = ndl.summation(exp_Z, axes=(1,))

    log_sum_exp_Z = ndl.log(sum_exp_Z)

    correct_logit = ndl.summation(Z * y_one_hot, axes=(1,))

    loss = log_sum_exp_Z - correct_logit

    avg_loss = ndl.summation(loss) / Z.shape[0]

    return avg_loss
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples, input_dim = X.shape
    hidden_dim, num_classes = W2.shape

    for i in range(0, num_examples, batch):
        X_batch = X[i:i + batch]
        y_batch = y[i:i + batch]
        batch_size = X_batch.shape[0]

        Z1 = ndl.Tensor(X_batch) @ W1
        A1 = ndl.relu(Z1)
        Z2 = A1 @ W2

        exp_Z2 = ndl.exp(Z2)
        sum_exp_Z2 = ndl.summation(exp_Z2, axes=(1,))
        probs = exp_Z2 / sum_exp_Z2.reshape((sum_exp_Z2.shape[0], 1)) 

        I_y = ndl.one_hot(num_classes, ndl.Tensor(y_batch))

        G2 = probs - I_y
        G1 = ndl.Tensor(Z1.numpy() > 0) * (G2 @ ndl.transpose(W2))  # 第一层梯度

        W2 -= lr * (ndl.transpose(A1) @ G2) / batch_size
        W1 -= lr * (ndl.transpose(ndl.Tensor(X_batch)) @ G1) / batch_size

    return W1, W2

    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
