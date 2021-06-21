import os, cv2
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from shutil import rmtree
from skimage import img_as_float
import skimage.io


    
def main(args):
    if os.path.exists(args.output_path):
        rmtree(args.output_path)
    os.makedirs(args.output_path)
    
    interpreter = tf.lite.Interpreter(model_path=args.tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    print(str(input_details))
    output_details = interpreter.get_output_details()
    print(str(output_details))
    files = [f for f in os.listdir(args.input_path) if f.endswith('.png')]
    print('{} images to be test'.format(len(files)))
    for f in tqdm(files):
        # img = cv2.cvtColor(cv2.imread(os.path.join(args.input_path, f)), cv2.COLOR_RGB2BGR) / 255.
        img = skimage.io.imread(os.path.join(args.input_path, f)) / 255.
        img = img.astype(np.float32)
        # img = extract_bayer_channels(img)
        with tf.device('/gpu:0'):
            interpreter.set_tensor(input_details[0]['index'], img[np.newaxis, ...])
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            prediction = np.clip(np.squeeze(prediction), 0, 1)
        
        cv2.imwrite(os.path.join(args.output_path, f), (prediction[:, :, ::-1] * 255.))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tflite_path', default='triplenet.tflite')
    parser.add_argument('-i', '--input_path',  default='input_tflite')
    parser.add_argument('-o', '--output_path', default='prediction')
    args = parser.parse_args()
    main(args)