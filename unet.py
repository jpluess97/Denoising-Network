import numpy as np
import tensorflow as tf

import lib
import metrics


class UNET(object):

    def __init__(self, model_name='model',
                 # train settings and initialization
                 learning_rate=1e-5, loss_schedule_idx=0,
                 # input and data pre-process
                 data_type='rgb',
                 # model architecture
                 residual=True, act_fn="relu", conv_type='sep_conv', with_batch_norm=True,
                 fore_block_type='conv', num_fore_blocks=1, num_fore_filter=64,
                 block_type='RB', down_means='shuffle', up_means='transpose',
                 num_enc_blocks=[2, 2, 2], num_enc_filters=[16, 32, 64], enc_kernel_size=3,
                 num_dec_blocks=[2, 2, 2], num_dec_filters=[16, 32, 64], dec_kernel_size=3,
                 last_block_type='conv', num_last_blocks=1, num_last_filter=256,
                 # loss functions
                 loss_type='SSIM_l1_loss'
                 ):

        assert data_type in ['rgb', 'raw']
        assert conv_type in ['conv', 'sep_conv']
        assert down_means in ['pool', 'shuffle', 'compose']
        assert up_means in ['bilinear', 'shuffle', 'pool', 'transpose']
        # train settings and initialization
        self.learning_rate = learning_rate
        # input and data pre-process
        self.data_type = data_type
        # architecture
        self.conv_type = conv_type
        self.residual = residual
        if act_fn == "relu":
            self.act_fn = lib._relu

        # Sigmoid activation to prevent clipping effect
        elif act_fn == "sigmoid":
            self.act_fn = lib._sigmoid
       # elif avt_fn == "satlin":
            #self.act_fn = lib._satlin
        else:
            raise ValueError()
        self.act_fn_out = lib._htangent
        self.with_batch_norm = with_batch_norm
        self.fore_block_type = fore_block_type
        self.num_fore_blocks = num_fore_blocks
        self.num_fore_filter = num_fore_filter
        self.block_type = block_type
        self.down_means = down_means
        self.up_means = up_means
        self.num_enc_blocks = num_enc_blocks
        self.num_enc_filters = num_enc_filters
        self.enc_kernel_size = enc_kernel_size
        self.num_dec_blocks = num_dec_blocks
        self.num_dec_filters = num_dec_filters
        self.dec_kernel_size = dec_kernel_size
        self.last_block_type = last_block_type
        self.num_last_blocks = num_last_blocks
        self.num_last_filter = num_last_filter

        # loss functions
        self.loss_schedule_idx = loss_schedule_idx
        self.loss_type = loss_type

        self.train_size = [256, 256]
        self.eval_size = [256, 256]
        self.test_size = [1024, 1024]

        # other initialization
        self.model_name = model_name
        self.kernel_initializer = tf.random_normal_initializer(stddev=0.01)
        self.bias_initializer = tf.constant_initializer(value=0.0)
        self.grayscale = self.data_type == 'grayscale' or self.data_type == 'raw'
        self.save_to_summary = True
        self.training = True
        self.summary = dict()


    def _convolution(self, input, num_filters=64, kernel_size=3, strides=1,
                     use_bias=True, use_bn=False):
        out = tf.layers.conv2d(input, num_filters, kernel_size, strides,
                               kernel_initializer=self.kernel_initializer,
                               bias_initializer=self.bias_initializer,
                               padding='same',
                               use_bias=use_bias)
        #out = tf.contrib.layers.layer_norm(out, center=True, scale=True, activation_fn=None, reuse=None,
                                           variables_collections=None, outputs_collections=None, trainable=True,
                                           begin_norm_axis=1, begin_params_axis=-1, scope=None)
        out = tf.layers.batch_normalization(out, training=self.training) if use_bn else out
        out = self.act_fn(out, training=self.training)


        return out
#trial with different activation function
    def _convolution_out(self, input, num_filters=64, kernel_size=3, strides=1,
                     use_bias=True, use_bn=False):
        out = tf.layers.conv2d(input, num_filters, kernel_size, strides,
                               kernel_initializer=self.kernel_initializer,
                               bias_initializer=self.bias_initializer,
                               padding='same',
                               use_bias=use_bias)
        dense_layer = tf.layers.Dense(16)
        out = dense_layer(out)
        print("out")
        print(out)
#no batch normalization
        #out = tf.layers.batch_normalization(out, training=self.training) if use_bn else out
        out = tf.contrib.layers.layer_norm(out, center=True, scale=True, activation_fn=None, reuse=None, variables_collections=None, outputs_collections=None, trainable=True, begin_norm_axis=1, begin_params_axis=-1, scope=None)
        out = self.act_fn_out(out, training=self.training)


        return out

#trial with diff activation function out
    def _separable_convolution_out(self, input, num_filters=64, kernel_size=3,
                               use_bias=True, use_bn=False):
        depthwise_filter = tf.get_variable('depthwise_filter', [kernel_size, kernel_size, input.shape[-1], 1],
                                           initializer=tf.random_normal_initializer(stddev=0.01))

        out = tf.nn.depthwise_conv2d(input, depthwise_filter, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
        out = tf.layers.conv2d(out, num_filters, 1,
                               kernel_initializer=self.kernel_initializer,
                               bias_initializer=self.bias_initializer,
                               padding='same',
                               use_bias=use_bias)
        out = tf.layers.batch_normalization(out, training=self.training) if use_bn else out
        out = self.act_fn_out(out, training=self.training)

        return out

    def _separable_convolution(self, input, num_filters=64, kernel_size=3,
                     use_bias=True, use_bn=False):
        depthwise_filter = tf.get_variable('depthwise_filter', [kernel_size, kernel_size, input.shape[-1], 1],
                                                   initializer=tf.random_normal_initializer(stddev=0.01))


        out = tf.nn.depthwise_conv2d(input, depthwise_filter, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
        out = tf.layers.conv2d(out, num_filters, 1,
                               kernel_initializer=self.kernel_initializer,
                               bias_initializer=self.bias_initializer,
                               padding='same',
                               use_bias=use_bias)
        out = tf.layers.batch_normalization(out, training=self.training) if use_bn else out
        out = self.act_fn(out, training=self.training)

        return out

    def _res_block(self, x, cond=None, num_filters=64, kernel_size=3, with_shortcut=True,
                   use_bias=True, use_bn=False):
        out = x

        out = tf.concat([cond, out], axis=-1) if cond is not None else out
        with tf.variable_scope('conv_0'):
            if self.conv_type == 'conv':
                out = self._convolution(out, num_filters, kernel_size, use_bn=use_bn)
            elif self.conv_type == 'sep_conv':
                out = self._separable_convolution(out, num_filters, kernel_size, use_bn=use_bn)
        with tf.variable_scope('conv_1'):
            if self.conv_type == 'conv':
                out = self._convolution(out, x.shape[-1], kernel_size, use_bn=use_bn)
            elif self.conv_type == 'sep_conv':
                out = self._separable_convolution(out, x.shape[-1], kernel_size, use_bn=use_bn)

        out = x + out if with_shortcut else out

        return out


    def _scalar_to_image(self, scalars, shape):
        scalar = tf.reshape(scalars, [shape[0], 1, 1, 1])
        image = tf.ones([shape[0], shape[1], shape[2], 1]) * scalar
        return image

    def _downsample_layer(self, input, factor, num_filters=64, kernel_size=3, means='pool'):
        if input is None or factor == 1:
            return input
        with tf.variable_scope('down_sample'):
            if means == 'shuffle':
                return tf.space_to_depth(input, block_size=factor)
            if means == 'pool':
                return tf.layers.average_pooling2d(input, pool_size=factor, strides=factor, padding='same')
            if means == 'compose':
                return tf.layers.conv2d(input, num_filters, kernel_size,
                                        strides=factor,
                                        padding='same',
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer=self.bias_initializer)

    def _upsample_layer(self, input, factor, shape, num_filters=64, kernel_size=2, means='transpose'):
        if input is None or factor == 1:
            return input
        with tf.variable_scope('up_sample'):
            if means == 'shuffle':
                return tf.depth_to_space(input, block_size=factor)
            if means == 'pool':
                return tf.image.resize_nearest_neighbor(input, [shape[1], shape[2]], align_corners=True)
            if means == 'bilinear':
                return tf.image.resize_bilinear(input, [shape[1], shape[2]], align_corners=True)
            if means == 'transpose':
                out = tf.layers.conv2d_transpose(input, num_filters, kernel_size,
                                                 strides=factor,
                                                 padding='same',
                                                 kernel_initializer=self.kernel_initializer,
                                                 bias_initializer=self.bias_initializer)
                # return tf.image.resize_image_with_crop_or_pad(out, shape[1], shape[2])
                return out

    def _unet(self, input, cond=None, num_enc_blocks=6, num_enc_filters=16, enc_kernel_size=3,
              num_dec_blocks=6, num_dec_filters=16, dec_kernel_size=3,
              down_means='pool', up_means='pool', global_skip=True):

        input = tf.concat([cond, input], axis=-1) if cond is not None else input
        enc_features = []
        enc_features.append(input)
        num_levels = len(num_enc_blocks)
        with tf.variable_scope('encoder'):
            for i in range(num_levels):
                for j in range(num_enc_blocks[i]):
                    with tf.variable_scope('block_{}_{}'.format(i, j)):
                        if self.block_type == 'RB':
                            enc_features[i] = self._res_block(enc_features[i],
                                                              num_filters=num_enc_filters[i],
                                                              kernel_size=enc_kernel_size,
                                                              use_bn=self.with_batch_norm)
                if i < num_levels - 1:
                    with tf.variable_scope('down_{}'.format(i)):
                        enc_features.append(
                            self._downsample_layer(enc_features[i], factor=2, num_filters=num_enc_filters[i],
                                                   means=down_means)
                        )
        with tf.variable_scope('decoder'):
            for i in reversed(range(1, num_levels)):
                with tf.variable_scope('up_{}'.format(i)):
                    enc_features[i] = self._upsample_layer(enc_features[i], factor=2, shape=tf.shape(enc_features[i - 1]),
                                                            num_filters=num_dec_filters[i - 1], kernel_size=dec_kernel_size,
                                                            means=up_means)
                enc_features[i - 1] = tf.concat([enc_features[i], enc_features[i - 1]], axis=-1) if global_skip else enc_features[i - 1]
                with tf.variable_scope('concat_conv_{}'.format(i)):
                    if self.conv_type == 'conv':
                        enc_features[i - 1] = self._convolution(enc_features[i - 1], num_filters=num_dec_filters[i-1],
                                                                kernel_size=dec_kernel_size, use_bn=self.with_batch_norm)
                    elif self.conv_type == 'sep_conv':
                        enc_features[i - 1] = self._separable_convolution(enc_features[i - 1],
                                                                          num_filters=num_dec_filters[i-1],
                                                                          kernel_size=dec_kernel_size,
                                                                          use_bn=self.with_batch_norm)
                for j in range(num_dec_blocks[i]):
                    with tf.variable_scope('block_{}_{}'.format(i, j)):
                        if self.block_type == 'RB':
                            enc_features[i - 1] = self._res_block(enc_features[i - 1],
                                                              num_filters=num_enc_filters[i - 1],
                                                              kernel_size=enc_kernel_size,
                                                              use_bn=self.with_batch_norm)

        return enc_features[0]



    def _dn_net(self, x, cond=None, training=False):
        assert len(self.num_enc_blocks) == len(self.num_enc_filters)
        assert len(self.num_enc_blocks) > 0
        assert len(self.num_dec_blocks) == len(self.num_dec_filters)
        assert len(self.num_dec_blocks) > 0
        if self.training:
            print('> TRAINING STATUS')
        else:
            print('> NO TRAINING STATUS')

        with tf.name_scope('input'):
            input = x

        # initial feature extraction with num_filters corresponding to initial level
        init_feature = input
        with tf.variable_scope('fore_conv'):
            for i in range(self.num_fore_blocks):
                if self.fore_block_type == 'conv':
                    init_feature = self._convolution(init_feature, self.num_fore_filter)
                elif self.fore_block_type == 'sep_conv':
                    init_feature = self._separable_convolution(init_feature, self.num_fore_filter)

        with tf.variable_scope('backbone'):
            output = self._unet(init_feature, cond=cond,
                                num_enc_blocks=self.num_enc_blocks,
                                num_enc_filters=self.num_enc_filters,
                                enc_kernel_size=self.enc_kernel_size,
                                num_dec_blocks=self.num_dec_blocks,
                                num_dec_filters=self.num_dec_filters,
                                dec_kernel_size=self.dec_kernel_size,
                                down_means=self.down_means, up_means=self.up_means, global_skip=True)

        # final feature reconstruction
        with tf.variable_scope('last_rec'):
            for _ in range(self.num_last_blocks):
                if self.last_block_type == 'conv':
                    output = self._convolution_out(output, self.num_last_filter)
                elif self.last_block_type == 'sep_conv':
                    output = self._separable_convolution_out(output, self.num_last_filter)

        with tf.variable_scope('output'):
            out = tf.layers.conv2d(output, x.shape[-1], 3,
                                   kernel_initializer=self.kernel_initializer,
                                   bias_initializer=self.bias_initializer,
                                   padding='same')
            out = out + x if self.residual else out
        #add linear layer
        return out

    def _loss(self, prediction, reference, loss_type, alpha=0):
        """ Compute loss based on loss_type
            x: gt [B, H, W, F], y: predicted [B, H, W, F]
        """
        if loss_type == 'l1':
            loss = metrics.l1_loss(prediction, reference)
        elif loss_type == 'l2':
            loss = metrics.l2_loss(prediction, reference)
        elif loss_type == "SSIM":
            #loss = metrics.tf_image_ssim(prediction, reference, max_val=1)
            loss = metrics.ssim_loss(prediction, reference)
        elif loss_type == "Charbonnier":
            loss = metrics.compute_charbonnier_loss(prediction, reference)
        elif loss_type == "SSIM_l1_loss":
            loss = metrics.SSIM_l1_loss(prediction, reference,alpha)
        elif loss_type == "SSIM_l2_loss":
            loss = metrics.SSIM_l2_loss(prediction, reference, alpha)
        elif loss_type == "PC_SSIM_l1_loss":
            loss = metrics.PC_SSIM_l1_loss(prediction, reference)
        else:
            raise ValueError()
        return loss


    def loss(self, prediction, reference, name='loss', type ='l1',alpha=0):

        type="l1"
        loss = self._loss(prediction, reference, loss_type=type, alpha=alpha)
        #print("loss")
        #print(loss)
        return loss

    def learning_rate_scheduler(self, step, num_steps, decay_strategy='cosine'):
        if decay_strategy == 'cosine':
            if (step + 1) / (num_steps // 4) < 1:
                decayed_lr = self.learning_rate
            else:
                decayed_lr = self.learning_rate * (
                        0.55 + 0.45 * np.cos(np.pi * (step + 1 - 0.25 * num_steps) / (0.75 * num_steps)))

        return decayed_lr

    def loss_scheduler(self, step, num_steps):
        if (step + 1) / (num_steps // 4) < 1:
            loss_idx = 1
            loss_name = 'pre-train loss'
        else:
            loss_idx = 1
            loss_name = 'refine loss'
        return loss_idx, loss_name

    def train(self, x, name='train', noise_level=None, training=True):
        self.training = training
        self.noise_level = noise_level
        with tf.name_scope(name):
            with tf.variable_scope(self.model_name, reuse=tf.AUTO_REUSE):
                out = self._dn_net(x)
        return out

    def evaluate(self, x, name='evaluate', noise_level=None, training=False):
        self.training = training
        self.noise_level = noise_level
        with tf.name_scope(name):
            with tf.variable_scope(self.model_name, reuse=tf.AUTO_REUSE):
                out = self._dn_net(x)
        return out

    def get_image_summary(self):
        return self.summary
