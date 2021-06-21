import os
import multiprocessing
import numpy as np
import warnings
from PIL import Image
import tensorflow as tf
import matplotlib
import random
import cv2
#from tensorflow_core.python.data.ops.dataset_ops import DatasetV1Adapter


def get_filenames(flist, full_path=True, folder_list=True):
    """Args:
      flist: file containing the names of all images to use
      full_path: return list of filenames with trailing absolute path
    Returns:
      A list of strings containing the full path to all images
    """
    with open(flist, 'r') as File:
        filenames = [f.strip() for f in File.readlines()]
        if full_path:
            filenames = [os.path.join(os.path.dirname(flist), f) for f in filenames]
    return filenames


def get_split_list(flist, split_ratio, random_perm=False):
    """ Useful to create training and evaluation datasets from
    images contained in a single folder: split_ratio% images will
    be returned in the first returned list, and the rest in the 
    # second returned list
    Args:
      flist: list of strings
      split_ratio: the first output list will contain a subset of
        the available image controlled by this ratio
    Returns:
      A pair of lists containing the full path to the files
    """
    assert split_ratio < 1.0
    split_len = min(len(flist) - 1, int(len(flist) * split_ratio))

    random_flist = np.random.permutation(flist)

    eval_flist = random_flist[split_len:]
    train_list = random_flist[:split_len]
    if not random_perm:
        train_list = np.sort(train_list)

    # print(train_list[0:20])
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # print(eval_flist[0:20])
    # exit(0)
    return train_list, eval_flist


def get_directories(flist):
    """Args:
      flist: file defining a list of directories, each containing
        a set of video frames - this is tailored to the raw data
    Returns:
      A list of strings containing the full path to the directories
    """
    with open(flist, 'r') as File:
        directories_in = File.readlines()
        directories_out = []
        path = os.path.dirname(flist)
        directories_out.extend([os.path.join(path, d.strip()) for d in directories_in])
    return directories_out


def build_dataset(flist, params, dir_flist=True, training=False):
    num_parallel_calls = params.cpu_thr
    num_samples = len(flist)
    grayscale = True if params.data_type == 'raw' else False
    num_samples_db = num_samples

    assert num_samples > 0

    # Create dataset

    with tf.device('/cpu:0'):
        batch_size = params.batch_size
        if training:
            if params.trainset_type == 'png':
                read_data_fn = lambda idx: read_MAI_image(tf.gather(flist, idx), grayscale=grayscale)
                # read_augmented = lambda idx: aug(tf.gather(flist, idx), grayscale=grayscale)
                read_data_fn_aug = lambda idx: read_MAI_image_aug(tf.gather(flist, idx), grayscale=grayscale)

                dataset = (
                    tf.data.Dataset.from_tensor_slices(np.arange(0, num_samples, dtype=np.int32))  # slice over indices
                        .map(read_data_fn, num_parallel_calls=num_parallel_calls)  # read data
                        # .map(read_augmented,num_parallel_calls=num_parallel_calls)
                        .batch(batch_size)  # extract batch of desired size from the shuffled dataset
                        .prefetch(batch_size)  # always have one batch ready

                )
                print(dataset)

                dataset_aug= (
                    tf.data.Dataset.from_tensor_slices(np.arange(0, num_samples, dtype=np.int32))  # slice over indices
                        .map(read_data_fn_aug, num_parallel_calls=num_parallel_calls)  # read data
                        # .map(read_augmented,num_parallel_calls=num_parallel_calls)
                        .batch(batch_size)  # extract batch of desired size from the shuffled dataset
                        .prefetch(batch_size)  # always have one batch ready

                )
                print(dataset_aug)


                dataset = dataset.concatenate(dataset_aug)
                print(dataset)

            num_samples_db = num_samples*2
            num_steps = ((num_samples + batch_size - 1) // batch_size) * 2
            num_steps_db = num_steps * 2

        else:
            if dir_flist:
                # By not specifying num_frames we read all available frames
                if params.trainset_type == 'png':
                    read_data_fn = lambda idx: read_MAI_image(tf.gather(flist, idx), grayscale=grayscale)
                    # read_data_fn_aug = lambda idx: read_MAI_image_aug(tf.gather(flist, idx), grayscale=grayscale)

                    # read_data_fn = tf.gather(flist, idx).map(read_MAI_image, num_parallel_calls=num_parallel_calls)
                    dataset = (tf.data.Dataset.from_tensor_slices(np.arange(0, num_samples, dtype=np.int32))
                               .map(read_data_fn, num_parallel_calls=num_parallel_calls)
                               # .batch(num_eval_sequences)
                               # .map(concat_sequences_fn, num_parallel_calls=num_parallel_calls)
                               .batch(1)
                               .prefetch(1)
                               )

            num_steps = num_samples
            num_steps_db = num_samples
            num_samples_db = num_samples

    data = dict()
    data['dataset'] = dataset
    data['num_samples'] = num_samples_db
    data['num_steps'] = num_steps_db

    return data


def read_png_image(filename, channels=1, dtype=tf.uint8):
    assert channels in [1, 3]
    assert dtype in [tf.uint8, tf.uint16]
    # filename = tf.Print(filename, [filename], 'filename')
    file = tf.read_file(filename)
    # We could use tf.image.decode_image (which would handle
    # for free png, jpeg, bmp, and gif) but it won't return
    # the output tensor shape
    # https://github.com/tensorflow/tensorflow/issues/8551
    image = tf.image.decode_png(file, channels=channels, dtype=dtype)
    # if grayscale:
    # assert channels == 3
    # image = tf.image.rgb_to_grayscale(image)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def read_MAI_image(directory, grayscale=True, bitdepth=8):
    channels = 1 if grayscale else 3
    directory = tf.string_split([directory], delimiter=' ').values[0]

    # Read gt and noisy image
    filename_gt = tf.string_join([directory, 'im.gt.png'], separator=os.path.sep)
    frame_gt = read_png_image(filename_gt, channels=channels, dtype=tf.uint8)

    filename_n = tf.string_join([directory, 'im.noisy.png'], separator=os.path.sep)
    frame_n = read_png_image(filename_n, channels=channels, dtype=tf.uint8)
    img = tf.concat([frame_n, frame_gt], axis=-1)
    img = tf.image.random_flip_up_down(img)

    # return (img[...,:3], img[...,3:])
    return frame_n, frame_gt


def read_MAI_image_aug(directory, grayscale=True, bitdepth=8):
    channels = 1 if grayscale else 3
    directory = tf.string_split([directory], delimiter=' ').values[0]

    # Read gt and noisy image
    filename_gt = tf.string_join([directory, 'im.gt.png'], separator=os.path.sep)
    frame_gt = read_png_image(filename_gt, channels=channels, dtype=tf.uint8)

    filename_n = tf.string_join([directory, 'im.noisy.png'], separator=os.path.sep)
    frame_n = read_png_image(filename_n, channels=channels, dtype=tf.uint8)

    # Join noisy and groudtruth for data augmentation
    img = tf.concat([frame_n, frame_gt], axis=-1)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.central_crop(img, central_fraction=0.5)

    print("AUGMENTATION")

    return img[..., :3], img[..., 3:]


def rotate(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_up_down(x)

    # Rotate 0, 90, 180, 270 degrees
    return (x[..., :3], x[..., 3:])
