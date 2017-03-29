"""Loss functions."""
import numpy as np 
import tensorflow as tf
import semver
import pdb 


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """

    d = y_true - y_pred
    # tf.where: cond, true, false (element-wise selection) 
    # return tf.where(tf.abs(d)<max_grad, 0.5*tf.square(d), tf.abs(d)-0.5)  
    # 32x6
    return tf.reduce_sum(tf.where(tf.abs(d)<max_grad, 0.5*tf.square(d), tf.abs(d)-0.5), axis=1)


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    print 'Calling mean_huber_loss ...'
    # pdb.set_trace()
    # return tf.reduce_mean(huber_loss(y_true, y_pred, max_grad), axis = -1)
    return tf.reduce_mean(huber_loss(y_true, y_pred, max_grad), axis=0)
