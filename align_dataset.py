import argparse
import os
import sys
from imageio import imread, imwrite
import numpy as np
from skimage import transform, img_as_ubyte
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  
from tqdm import tqdm  
from align.detect_face import create_mtcnn, detect_face 
from align.mtcnn_detector import extract_image_chips  

# main function to process images
def main(args):
    input_dir = 'input_folder/archive/facesmin3/lfw-deepfunneled'  # input directory for images
    if not os.path.exists(input_dir):  #check if input directory exists
        print(f"Directory does not exist: {input_dir}")
        exit(1)
    else:
        print(f"Directory exists: {input_dir}")
    
    output_dir = 'output_folder'  # directory for output images
    image_size = 160  # size of output aligned images
    padding = 0.3  # padding around faces for alignment

    # get all image file paths from the input directory
    image_list = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            print(f"Found file: {file}")
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                image_list.append(os.path.join(root, file))

    print(f"Total images found: {len(image_list)}")

    if image_list:
        print(f"Sample images: {image_list[:5]}")
    else:
        print("No images found. Check the directory structure or file extensions.")

    # mtcnn face detection parameters
    minsize = 20  # minimum size of face to detect
    threshold = [0.6, 0.7, 0.7]  # confidence thresholds for face detection
    factor = 0.709  # scaling factor for image pyramid

    print('Creating networks and loading parameters')
    with tf.compat.v1.Graph().as_default():
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU': 0}))  # disable gpu
        with sess.as_default():
            pnet, rnet, onet = create_mtcnn(sess, model_path='align/model')  # load mtcnn models

    file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(input_dir) for file in files if file.endswith(('.jpg', '.png'))]
    nrof_successfully_aligned = 0  # counter for successfully aligned faces

    # Process each image file
    for file_path in tqdm(file_paths, desc='Aligning... ', unit='imgs'):
        output_file_path = file_path.replace(input_dir, output_dir).replace('.jpg', '.png')  # define output path
        if not os.path.exists(output_file_path):
            output_sub_dir = os.path.dirname(output_file_path)
            if not os.path.exists(output_sub_dir):  # create subdirectory if needed
                os.makedirs(output_sub_dir, exist_ok=True)
            try:
                img = imread(file_path)  #read image
            except (IOError, ValueError, IndexError) as e:
                print(f'{file_path}: {e}')
            else:
                if len(img.shape) == 2:  # if grayscale, convert to RGB
                    img = np.tile(np.expand_dims(img, -1), [1, 1, 3])
                img = img[:, :, 0:3]  # retain only RGB channels

                # detect faces and landmarks
                bbxs, lms = detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                if bbxs.shape[0] > 0:  # if faces are detected
                    img_size = np.asarray(img.shape)[0:2]
                    if bbxs.shape[0] > 1:  # handle multiple faces in one image
                        bounding_box_size = (bbxs[:, 2] - bbxs[:, 0]) * (bbxs[:, 3] - bbxs[:, 1])
                        img_center = img_size / 2
                        offsets = np.vstack([(bbxs[:, 0] + bbxs[:, 2]) / 2 - img_center[1],
                                             (bbxs[:, 1] + bbxs[:, 3]) / 2 - img_center[0]])
                        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                        index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # choose face closest to center
                    else:
                        index = 0  # single face detected

                    # align face
                    aligned = extract_image_chips(img=img, points=np.transpose(np.expand_dims(lms[:, index], 1)),
                                                  desired_size=image_size, padding=padding)[0]
                    nrof_successfully_aligned += 1
                else:
                    aligned = transform.resize(img, (image_size, image_size), order=1)  # if no face is detected, resize image
                imwrite(output_file_path, img_as_ubyte(aligned))  # save the aligned face

    print(f'\nTotal number of images: {len(file_paths)}')
    print(f'Number of successfully aligned images: {nrof_successfully_aligned}')

# acript entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Path to the dataset which will be aligned.', default='/mnt/ssd/datasets/LFW/LFW')
    parser.add_argument('--output_dir', type=str, help='Path to the directy for the aligned faces.', default='/mnt/ssd/datasets/LFW/LFW_aligned')
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=160)
    main(parser.parse_args(sys.argv[1:]))  # Call main function with command-line arguments
