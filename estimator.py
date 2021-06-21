import os
import glob
import time
import json
import re
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.python.client import timeline
from tensorflow.python import pywrap_tensorflow
from PIL import Image
import PIL.Image
import cv2



import pdb

# To use matplotlib without attached display
import matplotlib

#from logs.test import Unet_trial

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data
import metrics
import utils
import lib
import cv2
# import moxing as mox

from tensorflow.python.framework import graph_util

from models import *
from test import *

class Estimator(object):

    def __init__(self, params):

        # Names of the directories created in save_dir containing the 
        # weights of the models obtained during the last 5 epochs, as
        # well as the best overall model obtained during training
        self.save_checkpoint_every_n_epochs = None
        self.periodic_save_dir = None
        self.last_save_dir = None
        self.best_save_dir = None

        # Unique placeholders in both train and eval graphs
        self.loss_schedule = tf.placeholder(tf.int32, name='loss_schedule')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # RAW data, so grayscale is true and number of input channel is 1
        self.grayscale = False
        self.in_channels = 3

        # Total number of training steps
        self.num_train_steps = 0
        self.iterator = None
        self.optimizer = None

        self.global_step = None
        # Network used for train/eval/test
        self.net = None
        self.model_name = 'model'

        self.bit_inputs = 8
        self.bit_model_outputs = 8

    def _get_network(self, params, training=False):
        if self.net is None:
            # Initialise network if self.net is None
            if params.model == 'unet':
                self.net = unet.UNET(
                    # train settings and initialization
                    learning_rate=params.learning_rate,
                    # input and data pre-process
                    data_type=params.data_type,
                    # model architecture
                    residual=params.residual,
                    act_fn=params.activation,
                    conv_type=params.conv_type,
                    with_batch_norm=params.with_batch_norm,
                    fore_block_type=params.fore_block_type,
                    num_fore_blocks=params.num_fore_blocks,
                    num_fore_filter=params.num_fore_filter,
                    block_type=params.block_type,
                    down_means=params.down_means,
                    up_means=params.up_means,
                    num_enc_blocks=params.num_enc_blocks,
                    num_enc_filters=params.num_enc_filters,
                    enc_kernel_size=params.enc_kernel_size,
                    num_dec_blocks=params.num_dec_blocks,
                    num_dec_filters=params.num_dec_filters,
                    dec_kernel_size=params.dec_kernel_size,
                    last_block_type=params.last_block_type,
                    num_last_blocks=params.num_last_blocks,
                    num_last_filter=params.num_last_filter,
                    # loss functions
                    loss_type=params.model_loss,
                    )
            else:
                raise ValueError('Unsupported network model: {}'.format(params.model))

        return self.net

    #Unet trial
# def get_network_unet(self, params, training=False):
        #self.net = Unet_trial(self, params)
        #return self.net


    def _build_train_eval_model(self, dataset, params, training=False, test=False):
        scope_name = 'train' if training else 'eval'
        scope_name_fn = lambda scope: '{}_{}'.format(scope_name, scope)
        self.global_step = tf.train.get_or_create_global_step()
        # Data pipeline
        with tf.name_scope(scope_name_fn('data')):

            if params.trainset_type == 'png':
                if training:
                    # In case of multi-gpu training we need a global (shared) iterator
                    if self.iterator is None:
                        self.iterator = dataset.make_initializable_iterator()
                    # iterator = self.iterator
                    batch_noisy, batch_clean = self.iterator.get_next()
                else:
                    iterator = dataset.make_initializable_iterator()
                    batch_noisy, batch_clean = iterator.get_next()

        # Build graph for training / evaluation
        net = self._get_network(params, training=training)

        # Build train/eval graph
        net_train_eval_fn = net.train if training else net.evaluate
        if params.model == 'unet':
            batch_predicted_net = net_train_eval_fn(batch_noisy)

            batch_predicted = batch_predicted_net

        if params.use_l2 == 1:
            loss_type='SSIM_l2_loss'
        else:
            loss_type='SSIM_l1_loss'

        # Evaluate metrics
        with tf.name_scope(scope_name_fn('metrics')):
            loss = net.loss(batch_predicted_net, batch_clean, name='loss', type=loss_type, alpha=params.alpha)
            psnr = tf.reduce_mean(metrics.tf_image_psnr(batch_clean, batch_predicted,
                                                        max_val=1.0, name='psnr'))
            ssim = tf.reduce_mean(metrics.tf_image_ssim(batch_clean, batch_predicted,
                                                        max_val=1.0, name='ssim'))


            loss_metric, loss_metric_update = tf.metrics.mean(loss, name='loss_mean')
            psnr_metric, psnr_metric_update = tf.metrics.mean(psnr, name='psnr_mean')
            ssim_metric, ssim_metric_update = tf.metrics.mean(ssim, name='ssim_mean')

        # Define training step that minimizes the loss with the Adam optimizer
        if training:
            with tf.name_scope('optimizer'):
                if self.optimizer is None:
                    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    #self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, clipvalue=1.0)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                model_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_name)
                grads_and_vars = self.optimizer.compute_gradients(loss, model_var)
                grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
                with tf.control_dependencies(update_ops):
                    train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

            # Define summary ops only for training
            tf.summary.scalar('metrics/loss', tf.reshape(loss,[]), collections=[scope_name])
            tf.summary.scalar('metrics/psnr', psnr, collections=[scope_name])
            tf.summary.scalar('metrics/ssim', ssim, collections=[scope_name])

        with tf.name_scope(scope_name_fn('summary')):

            max_outputs = 3
            model_summary = net.get_image_summary()

            for k, v in model_summary.items():
                tf.summary.image(k, v, max_outputs=max_outputs, collections=[scope_name])

            batch_sequence = tf.concat(
                [batch_noisy, batch_clean, batch_predicted, tf.abs(batch_predicted - batch_clean)], axis=2)
            tf.summary.image('noise_clean_pred_diff', batch_sequence,
                             max_outputs=max_outputs, collections=[scope_name])
            
            summary = tf.summary.merge_all(key=scope_name)
            print()

        # Model specification as a dictionary containing all the graph
        # operations that we might wish to run in a session
        model = dict()
        if training:
            model['init'] = [
                self.iterator.initializer,
                tf.variables_initializer(
                    tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope_name_fn('metrics')))
            ]
        else:
            model['init'] = [
                iterator.initializer,
                tf.variables_initializer(
                    tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope_name_fn('metrics')))
            ]

        model['metrics'] = {
            'loss': loss_metric,
            'psnr': psnr_metric,
            'ssim': ssim_metric,
        }

        model['metrics_update'] = tf.group(*[
            loss_metric_update,
            psnr_metric_update,
            ssim_metric_update
        ])

        model['batch_clean'] = batch_clean
        model['batch_noisy'] = batch_noisy
        model['batch_predicted'] = batch_predicted
        model['summary'] = summary
        model['loss_scheduler'] = net.loss_scheduler

        if training:
            model['learning_rate_scheduler'] = net.learning_rate_scheduler
            model['train_op'] = train_op
            model['update_ops'] = update_ops
            model['grads_and_vars'] = grads_and_vars

        return model

    def _build_train_model(self, dataset, params):
        print('> BUILDING TRAIN MODEL GRAPH')
        # TODO - multi-GPU also here?
        model = self._build_train_eval_model(dataset, params, training=True, test=False)
        return model

    def _build_eval_model(self, dataset, params):
        print('> BUILDING EVAL MODEL GRAPH')
        # TODO - multi-GPU also here?
        model = self._build_train_eval_model(dataset, params, training=False, test=False)
        return model

    def _build_test_model(self, dataset, params):
        print('> BUILDING TEST MODEL GRAPH')
        # TODO - multi-GPU also here?
        model = self._build_train_eval_model(dataset, params, training=False, test=True)
        return model

    def _evaluate_session(self, sess, model, num_steps, params, writer=None):
        write_summary_fn = lambda step: writer is not None and step == num_steps-1
        loss_scheduler_fn = model['loss_scheduler']

        # Initialize iterator and metrics
        temp = sess.run(model['init'])

        # Define which graph operations to run
        ops = dict()

        ops['batch_clean_shape'] = tf.shape(model['batch_clean'])
        ops['metrics_update'] = model['metrics_update']




        global_step_val = sess.run(tf.train.get_global_step())
        loss_schedule, loss_name = loss_scheduler_fn(global_step_val, self.num_train_steps)

        fps = 0.0
        ops_val = None
        options = None
        metadata = None

        for step in range(num_steps):
            try:
                start_time = time.time()

                # Feed the batch_clean data back to the graph together with the
                # current learning rate and loss schedule
                feed_dict = {
                    self.loss_schedule: loss_schedule
                }

                # Only evaluate summaries once at end of the session
                if write_summary_fn(step):
                    ops['summary'] = model['summary']
                    #eval['summary'] = model['summary']
                ops_val = sess.run(ops, feed_dict=feed_dict, options=options, run_metadata=metadata)
                #val_eval = sess.run(eval, feed_dict=feed_dict, options=options, run_metadata=metadata)
                # Compute average running time including I/O and memory transfer
                fps += (ops_val['batch_clean_shape'][-1] / self.in_channels) / (time.time() - start_time) / num_steps

            except tf.errors.OutOfRangeError:
                break
            finally:
                height = ops_val['batch_clean_shape'][1] if ops_val else 0
                width = ops_val['batch_clean_shape'][2] if ops_val else 0
                avg_fps = fps * (num_steps / (step + 1.0))
                message = '{}x{}x{} at {:.1f} fps'.format(height, width, self.in_channels, avg_fps)
                utils.progress(step+1, num_steps, 'eval {} ({})'.format(loss_name, message))
                #print(ops_val['batch_clean_shape'])
        print()

        # Compute final metric values (averaged over all evaluation dataset)

        metrics_val = sess.run(model['metrics'], feed_dict=feed_dict)

        if writer is not None:
            # Manually add evaluation metric values to summary
            writer.add_summary(ops_val['summary'], global_step_val)
            for k, v in metrics_val.items():
                summary = tf.Summary(value=[tf.Summary.Value(tag='metrics/{}'.format(k), simple_value=v)])
                writer.add_summary(summary, global_step_val)

        return metrics_val, avg_fps

    def _test_session(self, sess, model, num_steps, params, writer=None):
        write_summary_fn = lambda step: writer is not None and step == num_steps-1
        loss_scheduler_fn = model['loss_scheduler']

        # Initialize iterator and metrics
        sess.run(model['init'])

        # Define which graph operations to run
        ops = dict()
        test = dict()
        ops['batch_clean_shape'] = tf.shape(model['batch_clean'])
        ops['metrics_update'] = model['metrics_update']
        ops["batch_predicted"] = model["batch_predicted"]
        test["batch_predicted"] = model["batch_predicted"]



        global_step_val = sess.run(tf.train.get_global_step())
        loss_schedule, loss_name = loss_scheduler_fn(global_step_val, self.num_train_steps)

        fps = 0.0
        ops_val = None
        options = None
        metadata = None

        for step in range(num_steps):
            try:
                start_time = time.time()

                # Feed the batch_clean data back to the graph together with the
                # current learning rate and loss schedule
                feed_dict = {
                    self.loss_schedule: loss_schedule
                }

                # Only evaluate summaries once at end of the session
                if write_summary_fn(step):
                    ops['summary'] = model['summary']
                ops_val = sess.run(ops, feed_dict=feed_dict, options=options, run_metadata=metadata)

                #Save predicted batch of images for display
                batch_predicted = ops_val.get("batch_predicted")

                result = batch_predicted[0, :, :, :]
                result = result * 256
                result = result.astype(int)
                data = result
                array_im = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
                # clipping effect removal
                array_im = np.where(array_im > 255, 255, array_im)
                array_im = np.where(array_im < 0, 0, array_im)
                im = Image.fromarray(array_im)
                im.save("file_" + str(step) + "_" + str(params.use_l2) + "_" + str(params.alpha) + ".png")


                fps += (ops_val['batch_clean_shape'][-1] / self.in_channels) / (time.time() - start_time) / num_steps

            except tf.errors.OutOfRangeError:
                break
            finally:
                height = ops_val['batch_clean_shape'][1] if ops_val else 0
                width = ops_val['batch_clean_shape'][2] if ops_val else 0
                avg_fps = fps * (num_steps / (step + 1.0))
                message = '{}x{}x{} at {:.1f} fps'.format(height, width, self.in_channels, avg_fps)
                utils.progress(step+1, num_steps, 'eval {} ({})'.format(loss_name, message))
        print()

        # Compute final metric values (averaged over all evaluation dataset)

        metrics_val = sess.run(model['metrics'], feed_dict=feed_dict)
        #batch_predicted = sess.run(model["batch_predicted"])
        #print("batch_predicted")
        #print(model["batch_predicted"])

        if writer is not None:
            # Manually add evaluation metric values to summary
            writer.add_summary(ops_val['summary'], global_step_val)
            for k, v in metrics_val.items():
                summary = tf.Summary(value=[tf.Summary.Value(tag='metrics/{}'.format(k), simple_value=v)])
                writer.add_summary(summary, global_step_val)

        return metrics_val, avg_fps, batch_predicted

    def _train_session(self, sess, model, num_steps, params, writer=None):
        # Evaluate summaries for tensorboard only once every n steps
        write_summary_fn = lambda step: writer is not None and (step) % (params.write_summary_every_n_steps) == 0
        learning_rate_scheduler_fn = model['learning_rate_scheduler']
        loss_scheduler_fn = model['loss_scheduler']

        # Initialize iterator and metrics
        sess.run(model['init'])
        global_step = tf.train.get_global_step()

        # Define the graph operations to run
        ops = dict()
        ops['train_op'] = model['train_op']
        ops['metrics'] = model['metrics']
        ops['metrics_update'] = model['metrics_update']

        for step in range(num_steps):
            global_step_val = sess.run(global_step) + self.restored_global_step
            learning_rate = learning_rate_scheduler_fn(global_step_val - self.restored_global_step, self.num_train_steps - self.restored_global_step, params.lr_decay_strategy)
            loss_schedule, loss_name = loss_scheduler_fn(global_step_val, self.num_train_steps)
            try:
                # Feed the current learning rate and loss schedule
                feed_dict = {
                    self.learning_rate: learning_rate,
                    self.loss_schedule: loss_schedule
                }
                if write_summary_fn(step):
                    restemp, summary_val = sess.run([ops, model['summary']], feed_dict=feed_dict)
                    writer.add_summary(summary_val, global_step_val)
                else:
                    _ = sess.run(ops, feed_dict=feed_dict)
            except tf.errors.OutOfRangeError:
                # OutOfRangeError means that the data in the dataset have been exahusted
                # We should never be here if num_steps is correctly initialized
                break
            finally:
                utils.progress(step+1, num_steps, 'train {} (lr {})'.format(loss_name, learning_rate))
        print()

        # Compute final metric values (averaged over all training dataset)
        metrics_val = sess.run(model['metrics'], feed_dict=feed_dict)

        return metrics_val

    def _get_train_variables(self, scope=None, exclude_scope=None):
        train_vars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        if scope is not None:
            train_vars = [v for v in train_vars if v.op.name.find(scope) >= 0]
        if exclude_scope is not None:
            train_vars = [v for v in train_vars if not v.op.name.find(exclude_scope) >= 0]
        return train_vars

    def _get_global_variables(self, scope=None, exclude_scope=None):
        train_vars = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
        if scope is not None:
            train_vars = [v for v in train_vars if v.op.name.find(scope) >= 0]
        if exclude_scope is not None:
            train_vars = [v for v in train_vars if not v.op.name.find(exclude_scope) >= 0]
        return train_vars

    def _parameters_count(self, scope=None, exclude_scope=None):
        # Return number of trainable parameters, this must be called after
        # defining a training model
        train_vars = self._get_train_variables(scope=scope, exclude_scope=exclude_scope)
        if train_vars:
            return np.sum([np.prod(v.get_shape().as_list()) for v in train_vars])
        return 0

    def _update_best_metrics(self, eval_metrics, best_eval_metrics):
        # If metric is loss, then current is better than best if
        # its value is smaller, otherwise if metric is psnr or ssim
        # current is better if its value is larger than the best one
        is_better_fn = lambda metric, current, best: current < best if metric == 'loss' else current > best

        for k, v in eval_metrics.items():
            # For each metric, check if current value is better 
            # than the corresponding best value
            if k in best_eval_metrics:
                is_better = is_better_fn(k, v, best_eval_metrics[k])
            else:
                is_better = True
            # Update best value if current value is better
            if is_better:
                best_eval_metrics[k] = v

        # Return dict containing the values of the best metrics found so 
        # far, to be used in next iteration
        return best_eval_metrics

    def train_and_evaluate(self, train_data, eval_data, params, log_dir, test_data=None, save_dir=None, output_dir=None, restore_dir=None):
        # Define training and evaluation models in separate graphs
        train_model = self._build_train_model(train_data['dataset'], params)
        eval_model = self._build_eval_model(eval_data['dataset'], params)
        if test_data is not None:
            test_model = self._build_test_model(test_data['dataset'], params)

        # Set number of training steps
        self.num_train_steps = train_data['num_steps'] * params.num_epochs

        model_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_name)

        # Initialize tf.Saver instances to save weights during training
        # This must be defined after the definition of our training model
        if save_dir is not None:
            self.last_save_dir = os.path.join(save_dir, 'last')
            self.best_save_dir = os.path.join(save_dir, 'best')
            last_saver = tf.train.Saver(max_to_keep=3, name='last_saver', var_list=model_var) # keep the last two epochs
            best_saver = tf.train.Saver(max_to_keep=3, name='best_saver', var_list=model_var) # keep the best checkpoint based on eval psnr

        #      # saver = tf.train.import_meta_graph('restore_dir/epoch-39.meta')This will contain the values of the best evaluation metrics
        best_eval_metrics = dict()
        best_cpk_metrics = dict()
        best_eval_psnr = 0.0

        # Get timestamp to compute total elapsed time
        start_time = time.time()

        # Initialise session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            print()
            print('Training')
            print(' - Training samples {}'.format(train_data['num_samples']))
            print(' - Evaluation samples {}'.format(eval_data['num_samples']))
            print(' - Epochs {}'.format(params.num_epochs))
            print(' - Batch size {}'.format(params.batch_size))
            print(' - Training steps {}'.format(self.num_train_steps))
            print()
            print('Network')
            print(' - Model {}'.format(params.model.upper()))
            print(' - Parameters {}'.format(self._parameters_count()))
            print()
            print('Data')
            print(' - Data type {}'.format(params.data_type.upper()))
            print(' - Patch size {}x{}x{}'.format(
                params.patch_size, params.patch_size, self.in_channels))
            # print(' - Noise {}'.format('conditional information' if params.conditional else 'blind'))
            # print(' - Levels {}'.format(pretty_noise_levels()) if self.random_iso else ' - Depend on data')
            print()
            print('Directories')
            print(' - Logs {}'.format(log_dir))
            print(' - Checkpoint {}'.format(save_dir))
            print(' - Output {}'.format(output_dir))
            print(' - Restore {}'.format(restore_dir))
            print()

            # Init all graph variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Instantiate summary writers for both training and evaluation
            train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
            eval_writer = tf.summary.FileWriter(os.path.join(log_dir, 'eval'), sess.graph)

            # Reload weights from directory if specified
            restored_epoch = 0
            self.restored_global_step = 0
            if restore_dir is not None:
                if os.path.isdir(restore_dir):
                    checkpoint = tf.train.latest_checkpoint(restore_dir)
                    if checkpoint:
                        restored_epoch = int(checkpoint.split('-')[-1])
                        self.restored_global_step = train_data['num_steps'] * restored_epoch
                        print('Restored model {} at epoch {} from {}'.format(params.data_type.upper(), restored_epoch, restore_dir))

                        last_saver.restore(sess, checkpoint)
                    else:
                        print('Could not restore model {} from {}'.format(params.data_type.upper(), load_dir))

            fps = 0.0
            # fine_tuned_steps = 0
            for _epoch in range(restored_epoch, params.num_epochs):
                # We want 1-based count of the epoch number
                epoch = _epoch + 1

                print('Epoch {}/{}'.format(epoch, params.num_epochs), end='\n')
                # To reduce memory footprint, we evaluate summaries and
                # write evaluation images to disk only every n epochs
                eval_writer_epoch = eval_writer if epoch % params.write_eval_summary_every_n_epochs == 0 else None
                train_start = time.time()
                train_metrics = self._train_session(
                    sess, train_model, train_data['num_steps'], params,
                    writer=train_writer)
                train_time = time.time() - train_start
                eval_metrics, eval_fps= self._evaluate_session(
                    sess, eval_model, eval_data['num_steps'], params,
                    writer=eval_writer_epoch)


                if (_epoch == (params.num_epochs-1)):
                    if test_data is not None:
                        number_steps = 3
                        test_metrics, test_fps, test_val= self._test_session(
                            sess, test_model, test_data['num_steps'], params)
                        print("I AM IN")
                        print(test_metrics)
                        image = test_model["batch_predicted"]

                        result = test_val[0,:, :, :]
                        #tf.reduce_mean(result,axis=1)
                        print(result)
                        result = result*256
                        result = result.astype(int)

                        data = result
                        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
                        print(rescaled)

                        im = Image.fromarray(rescaled)
                        im.save('03.png')














                        #x = np.zeros((1, 1024, 1024, 3))
                        #result = batch[0, :, :, :]
                        #print(result.shape)
                        #print(result)
                        #img = Image.fromarray(result, 'RGB')
                        #img.save('test.png')
                        #img.show()


                        #image_print = tf.io.encode_png(image, compression=-1, name=None)

                eval_time = time.time() - train_time - train_start
                elapsed_time = time.time() - start_time
                # Keep track of the best metrics we have seen so far
                best_eval_metrics = self._update_best_metrics(eval_metrics, best_eval_metrics)
                fps += (eval_fps / params.num_epochs)

                # Save model, we have to Savers, one containing the last 5
                # checkpoints, and another containing the best model so far
                # according to eval PSNR
                if save_dir is not None:
                    # Save last model
                    last_saver.save(sess, os.path.join(self.last_save_dir, 'epoch'), global_step=epoch)
                    # Save model if current evaluation psnr is best so far
                    if params.model == 'unet':
                        if eval_metrics['psnr'] > best_eval_psnr:
                            best_eval_psnr = eval_metrics['psnr']
                            best_cpk_metrics = eval_metrics
                            best_saver.save(sess, os.path.join(self.best_save_dir, 'epoch'), global_step=epoch)
                            best_epoch = epoch

                # Print some output
                #print("testvalues")
                #for i in test_model["metrics"]:
                    #print(i)

                print('\ttrain\t\teval\t\tbest eval\tbest ckp ({})'.format(best_epoch))
                for k, v in train_metrics.items():
                    print('{}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}'.format(k, v, eval_metrics[k], best_eval_metrics[k], best_cpk_metrics[k]))
                print()
                print('elapsed {} / avg epoch {} / train time {} / eval time {} / avg fps {:.1f} / to go {}'.format(
                    utils.time_to_string(elapsed_time),
                    utils.time_to_string(elapsed_time / (epoch - restored_epoch)),
                    utils.time_to_string(train_time),
                    utils.time_to_string(eval_time),
                    fps * ((params.num_epochs - restored_epoch) / (_epoch - restored_epoch + 1.0)),
                    utils.time_to_string(elapsed_time / (epoch - restored_epoch) * (params.num_epochs - epoch))
                ))
                print()