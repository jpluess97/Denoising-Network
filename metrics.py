import tensorflow as tf
import numpy as np
import lib


def l2_loss(x, y):
    with tf.name_scope('l2'):
        return tf.reduce_sum(tf.reduce_mean(tf.squared_difference(x, y), axis=[0,1,2]))


def l1_loss(x, y):
    with tf.name_scope('l1'):
        return tf.reduce_sum(tf.reduce_mean(tf.math.abs(x - y),axis=[0,1,2]))
# tf.reduce_mean(tf.abs(x - y))


def tf_image_psnr(img1, img2, max_val, name=None):
    """Returns the Peak Signal-to-Noise Ratio between img1 and img2.
    This is intended to be used on signals (or images). Produces a PSNR value for
    each image in batch.
    The last three dimensions of input are expected to be [height, width, depth].
    Returns:
      The scalar PSNR between img1 and img2. The returned tensor has type `tf.float32`
      and shape [batch_size, 1].
    """
    # Need to convert the images to float32.  Scale max_val accordingly so that
    # PSNR is computed correctly.

    with tf.name_scope(name):
        max_val = tf.cast(max_val, img1.dtype)
        max_val = tf.image.convert_image_dtype(max_val, tf.float32)
        img1 = tf.image.convert_image_dtype(img1, tf.float32)
        img2 = tf.image.convert_image_dtype(img2, tf.float32)
        mse = tf.reduce_mean(tf.squared_difference(img1, img2), [-3, -2, -1])
        psnr_val = tf.subtract(
            20 * tf.log(max_val) / tf.log(10.0), 
            np.float32(10 / np.log(10)) * tf.log(mse))

    return psnr_val


def tf_image_ssim(img1, img2, max_val, name=None):
    """Computes SSIM index between img1 and img2.
    This function is based on the standard SSIM implementation.
    Note: The true SSIM is only defined on grayscale.  This function does not
    perform any colorspace transform.  (If input is already YUV, then it will
    compute YUV SSIM average.)
    Details:
      - 11x11 Gaussian filter of width 1.5 is used.
      - k1 = 0.01, k2 = 0.03 as in the original paper.
      The image sizes must be at least 11x11 because of the filter size.
    Returns:
      A tensor containing an SSIM value for each image in batch.  Returned SSIM
      values are in range (-1, 1], when pixel values are non-negative. Returns
      a tensor with shape: broadcast(img1.shape[:-3], img2.shape[:-3]).
    """

    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    with tf.name_scope(name):
        max_val = tf.cast(max_val, img1.dtype)
        max_val = tf.image.convert_image_dtype(max_val, tf.float32)
        img1 = tf.image.convert_image_dtype(img1, tf.float32)
        img2 = tf.image.convert_image_dtype(img2, tf.float32)
        ssim_per_channel, _ = _ssim_per_channel(img1, img2, max_val)
        # Compute average over color channels.
        ssim_val = tf.reduce_mean(ssim_per_channel, [-1])
    return ssim_val

def _ssim_per_channel(img1, img2, max_val=1.0):
    """Computes SSIM index between img1 and img2 per color channel.
    This function matches the standard SSIM implementation from:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
    quality assessment: from error visibility to structural similarity. IEEE
    transactions on image processing.
    Details:
      - 11x11 Gaussian filter of width 1.5 is used.
      - k1 = 0.01, k2 = 0.03 as in the original paper.
    Args:
      img1: First image batch.
      img2: Second image batch.
      max_val: The dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
    Returns:
      A pair of tensors containing and channel-wise SSIM and contrast-structure
      values. The shape is [..., channels].
    """
    filter_size = tf.constant(11, dtype=tf.int32)
    filter_sigma = tf.constant(1.5, dtype=img1.dtype)

    shape1, shape2 = tf.shape_n([img1, img2])

    # TODO(sjhwang): Try to cache kernels and compensation factor.
    kernel = _fspecial_gauss(filter_size, filter_sigma)
    kernel = tf.tile(kernel, multiples=[1, 1, shape1[-1], 1])

    # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
    # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
    compensation = 1.0

    # TODO(sjhwang): Try FFT.
    # TODO(sjhwang): Gaussian kernel is separable in space. Consider applying
    #   1-by-n and n-by-1 Gaussain filters instead of an n-by-n filter.
    def reducer(x):
        shape = tf.shape(x)
        x = tf.reshape(x, shape=tf.concat([[-1], shape[-3:]], 0))
        y = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
        return tf.reshape(y, tf.concat([shape[:-3], tf.shape(y)[1:]], 0))

    luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation)

    # Average over the second and the third from the last: height, width.
    axes = tf.constant([-3, -2], dtype=tf.int32)
    ssim_val = tf.reduce_mean(luminance * cs, axes)
    cs = tf.reduce_mean(cs, axes)
    return ssim_val, cs

def _fspecial_gauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  size = tf.convert_to_tensor(size, tf.int32)
  sigma = tf.convert_to_tensor(sigma)

  coords = tf.cast(tf.range(size), sigma.dtype)
  coords -= tf.cast(size - 1, sigma.dtype) / 2.0

  g = tf.square(coords)
  g *= -0.5 / tf.square(sigma)

  g = tf.reshape(g, shape=[1, -1]) + tf.reshape(g, shape=[-1, 1])
  g = tf.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
  g = tf.nn.softmax(g)
  return tf.reshape(g, shape=[size, size, 1, 1])

def _ssim_helper(img1, img2, reducer, max_val, compensation=1.0):
    """Args:
      img1: First set of images.
      img2: Second set of images.
      reducer: Function that computes 'local' averages from set of images.
        For non-covolutional version, this is usually tf.reduce_mean(img1, [1, 2]),
        and for convolutional version, this is usually tf.nn.avg_pool or
        tf.nn.conv2d with weighted-sum kernel.
      max_val: The dynamic range (i.e., the difference between the maximum
        possible allowed value and the minimum allowed value).
      compensation: Compensation factor. See above.
    Returns:
      A pair containing the luminance measure, and the contrast-structure measure.
    """
    _SSIM_K1 = 0.01
    _SSIM_K2 = 0.03
    c1 = (_SSIM_K1 * max_val) ** 2
    c2 = (_SSIM_K2 * max_val) ** 2

    # SSIM luminance measure is
    # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
    mean0 = reducer(img1)
    mean1 = reducer(img2)
    num0 = mean0 * mean1 * 2.0
    den0 = tf.square(mean0) + tf.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    # SSIM contrast-structure measure is
    #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
    # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
    #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
    num1 = reducer(img1 * img2) * 2.0
    den1 = reducer(tf.square(img1) + tf.square(img2))
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    # SSIM score is the product of the luminance and contrast-structure measures.
    return luminance, cs

def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

#Charbonnier Loss ( also called pseudo huber loss)
def compute_charbonnier_loss(tensor1, tensor2, is_mean=True):
    epsilon = 1e-6
    if is_mean:
        loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))
    else:
        loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))

    return loss

#Mixture of SSIM and L1 loss with parameter alpha
def SSIM_l1_loss(img1, img2, alpha = 0.25):
    SSIM = tf.reduce_sum(tf.reduce_mean(tf.image.ssim(img1, img2, 1.0)))
    l1 = tf.reduce_sum(tf.reduce_mean(tf.abs(img1 - img2),axis=[0,1,2]))

    mix = alpha * (1-SSIM) + (1 - alpha) * l1
    return mix

#Mixture of SSIM and L2 loss with parameter alpha
def SSIM_l2_loss(img1, img2, alpha = 0.25):
    SSIM = tf.reduce_sum(tf.reduce_mean(tf.image.ssim(img1, img2, 1.0)))
    l2 = tf.reduce_sum(tf.reduce_mean(tf.squared_difference(img1, img2), axis=[0,1,2]))
    mix = alpha * (1-SSIM) + (1 - alpha) * l2
    return mix

def group_norm(x, G=4, eps=1e-5, scope='group_norm') :
    with tf.variable_scope(scope) :
        N, H, W, C = x.get_shape().as_list()
        G = min(G, C)

        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))


        x = tf.reshape(x, [N, H, W, C]) * gamma + beta

    return x