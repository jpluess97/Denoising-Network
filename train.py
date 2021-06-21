
import os
import glob
import sys
import time
import traceback
import argparse
import json
import random

import data
import utils
import estimator
import tensorflow as tf

#data insertion



# Use only GPU 0
#os.environ["CUDA_VISIBLE_DEVICES"]= '0'

# env = 'local'
base_save_dir = './checkpoint'
base_log_dir = './logs'
base_output_dir = './output'


###############################################################################
###############################################################################

# I/O folders
# Default name of JSON file containing all model parameters
params_file = 'params.json'

write_summary_every_n_steps = 2048
write_eval_summary_every_n_epochs = 1

num_epochs = 200
batch_size = 64

patch_size = 256

###############################################################################
###############################################################################

# tf.enable_eager_execution()

# Now we can parse the cmd line arguments
parser = argparse.ArgumentParser()

# Data
parser.add_argument('--name', default=None, help='name of the experiment')
parser.add_argument('--train_flist', default=None, help='file containing the list of training sequences')
parser.add_argument('--eval_flist', default=None, help='file containing the list of evaluation sequences')
parser.add_argument('--test_flist', default=None, help='file containing the list of evaluation sequences')
parser.add_argument('--cpu_thr', type=int, default=40, help='number of thread on CPU to prepare the dataset')
parser.add_argument('--trainset_type', default='png', choices=['png'], help='')
parser.add_argument('--data_type', default='rgb', choices=['rgb'], help='data domain for denoising')

# Checkpoint / summary
parser.add_argument('--save_dir', default=base_save_dir, help='directory to save model weights')
parser.add_argument('--restore_dir', default=None, help='directory to load model weights for training')
parser.add_argument('--log_dir', default=base_log_dir, help='directory to save tensorboard logs')
parser.add_argument('--params', default=None, help='JSON file containing serialized arguments of a saved model (useful when restore_dir or test_load_dir is given)')
parser.add_argument('--write_summary_every_n_steps', type=int, default=write_summary_every_n_steps, help='write tensorboard training summary every n training steps')
parser.add_argument('--write_eval_summary_every_n_epochs', type=int, default=write_eval_summary_every_n_epochs, help='write tensorboard evaluation summary every n training epochs')

# Output images
parser.add_argument('--output_dir', default=base_output_dir, help='directory to write validation sequences (if write_output is enabled)')

# Model
parser.add_argument('--model', default='unet', choices=['unet'], help='select model to train')
parser.add_argument('--residual', action='store_true', help='enable global residual connection')
parser.add_argument('--activation', default='relu', choices=['relu'], help='activation function in network')
parser.add_argument('--conv_type', default='conv', choices=['conv','sep_conv'], help='select model to train')
parser.add_argument('--block_type', default='RB', choices=['RB'], help='select model to train')
parser.add_argument('--with_batch_norm', action='store_true', help='whether to use Batch Normalization in backbone')
parser.add_argument('--num_fore_blocks', type=int, default=1, help='number of conv blocks filters in the model')
parser.add_argument('--num_fore_filter', type=int, default=32, help='number of channel in the model')
parser.add_argument('--fore_block_type', default='conv', choices=['conv', 'sep_conv'], help='')
parser.add_argument('--num_enc_blocks', nargs='+', type=int, default=[1, 2, 2, 1], help='number of conv blocks filters in the model')
parser.add_argument('--num_enc_filters', nargs='+', type=int, default=[32, 64, 128, 256, 256], help='number of conv blocks filters in the model')
parser.add_argument('--enc_kernel_size', type=int, default=3, help='number of conv blocks filters in the model')
parser.add_argument('--num_dec_blocks', nargs='+', type=int, default=[1, 2, 2, 1], help='number of conv blocks filters in the model')
parser.add_argument('--num_dec_filters', nargs='+', type=int, default=[32, 64, 128, 256, 256], help='number of conv blocks filters in the model')
parser.add_argument('--dec_kernel_size', type=int, default=3, help='number of conv blocks filters in the model')
parser.add_argument('--num_last_blocks', type=int, default=1, help='number of conv blocks filters in the model')
parser.add_argument('--num_last_filter', type=int, default=64, help='number of conv blocks filters in the model')
parser.add_argument('--last_block_type', default='conv', choices=['conv', 'sep_conv'], help='')
parser.add_argument('--down_means', default='shuffle', choices=['shuffle', 'pool', 'compose'], help='')
parser.add_argument('--up_means', default='shuffle', choices=['shuffle', 'pool', 'transpose', 'bilinear'], help='')

# Specific loss function
parser.add_argument('--model_loss', default='SSIM', choices=['l1', 'l2',"SSIM"], help='model loss')

parser.add_argument('--num_epochs', type=int, default=num_epochs, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=batch_size, help='size of the mini-batch')
parser.add_argument('--patch_size', type=int, default=patch_size, help='size of the patches used for training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate for training')
parser.add_argument('--lr_decay_strategy', default='cosine', choices=['cosine'], help='')
parser.add_argument('--optimizer', default='Adam', choices=['Adam'], help='')
# TFlite
parser.add_argument('--convert_to_tflite', action='store_true', help='')
parser.add_argument('--tflite_input_as_none', action='store_true', help='')

parser.add_argument('--alpha', type=float, default=0.25, help='alpha rate for loss function')
parser.add_argument('--use_l2', type=int, default=0, help='whether to use l2 with alpha')



###############################################################################
###############################################################################


def get_unique_dir_names(args):
    """ Generate a unique name from the given arguments. This
    name will be used to create a directory that will contain the
    tensorboard logs as well as the model checkpoints
    """

    # Append timestamp to name in order to avoid clashes
    name = '{}'.format(time.time())

    log_dir = os.path.join(args.log_dir, name)
    save_dir = os.path.join(args.save_dir, name)
    output_dir = os.path.join(args.output_dir, name)

    return log_dir, save_dir, output_dir

def save_args(args, save_dir, params_file=params_file):
    """ Serialize given arguments to a JSON file
    """
    filename = os.path.join(save_dir, params_file)
    # Write params as JSON data to file
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True, ensure_ascii=False)

def load_args():
    """ Parse arguments from command line, If args.params is a valid JSON
    file, load the serialized argument in the JSON file.
    """
    # Parse current args
    args = parser.parse_args()
    # args, _ = parser.parse_known_args()
    # Return current argument if args.params file is not given
    if args.params is None:
        return args
    # Convert JSON data to Namespace arguments
    json_data = json.load(open(args.params))
    params = argparse.Namespace(**json_data)

    # Replace input directories with the ones in current args
    params.train_flist = args.train_flist
    params.eval_flist = args.eval_flist
    if args.test_flist is not None:
        params.test_flist = args.test_flist
    #params.random_perm = args.random_perm
    params.restore_dir = args.restore_dir
    params.batch_size = args.batch_size
    # Replace name
    params.name = args.name
    return params

def build_train_eval_datasets(args):
    """ Build training and evaluation datasets from the given arguments.
    This function will return a dictionary of TensorFlow operation that
    can be used directly in the computation graph. If an evaluation 
    dataset is not given, then the training dataset will be split 
    according to args.split_ratio (by default 80/20)
    """
    if args.trainset_type == 'png':
        assert args.train_flist is not None
        train_list = data.get_directories(args.train_flist)

    # train_list1 = []
    if args.eval_flist is None:
        train_list, eval_list = data.get_split_list(train_list, 0.9, random_perm=True)
    else:
        eval_list = data.get_directories(args.eval_flist)
     # test data
    if args.test_flist is not None:
        test_list = data.get_directories(args.test_flist)

    # Create dataset from training and evaluation files
    train_data = data.build_dataset(train_list, args, training=True)

    eval_data = data.build_dataset(eval_list, args, dir_flist=True, training=False)

    #test data
    if args.test_flist is not None:
        test_data = data.build_dataset(test_list, args, dir_flist=True, training=False)


    if args.test_flist is not None:
        return train_data, eval_data, test_data

    return train_data, eval_data, #test_data


###############################################################################
###############################################################################


if __name__ == '__main__':
    args = load_args()
    log_dir, save_dir, output_dir = get_unique_dir_names(args)
    print([log_dir, save_dir, output_dir])
    # Create directories if necessary
    utils.make_dirs([log_dir, save_dir, output_dir])
    # Write args as JSON data to file
    save_args(args, save_dir)
    # add return test_data
    if args.test_flist is not None:
        train_data, eval_data, test_data = build_train_eval_datasets(args)
    else:
        train_data, eval_data = build_train_eval_datasets(args)

    est = estimator.Estimator(args)
    if args.test_flist is not None:
        est.train_and_evaluate(train_data, eval_data, args, log_dir, test_data,
                                save_dir=save_dir,
                                output_dir=output_dir,
                                restore_dir=args.restore_dir
                                )
    else:
        est.train_and_evaluate(train_data, eval_data, args, log_dir,
                                save_dir=save_dir,
                                output_dir=output_dir,
                                restore_dir=args.restore_dir
                                )

