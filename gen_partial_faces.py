import argparse
import os
import sys

import cv2  # open source computer vision library
import numpy as np  # numerical operations
from tqdm import tqdm  # progress bar utility

def main(args):
    input_dir = 'output_folder'  # input directory containing images
    output_dir = 'partial_faces'  # output directory for saving partial faces
    image_size = args.image_size  # target image size (square dimensions)
    PADDING = 0.3  # padding around the face for cropping

    FACTORS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]  # different cropping factors
    PARTS = ['lEye', 'rEye', 'Nose', 'Mouth']  # facial parts to extract

    mean_face_shape_x = [0.2194, 0.7747, 0.4971, 0.3207, 0.6735]  # normalized mean x-coordinates of landmarks
    mean_face_shape_y = [0.1871, 0.1871, 0.5337, 0.7633, 0.7633]  # normalized mean y-coordinates of landmarks
    mean_face_shape_x[3] = 0.5 * (mean_face_shape_x[3] + mean_face_shape_x[4])  # average x-coordinate for mouth

    # calculate coordinates of facial landmarks with padding
    point_dict = {PARTS[idx]: [(PADDING + mean_face_shape_x[idx]) / (2 * PADDING + 1) * image_size,
                               (PADDING + mean_face_shape_y[idx]) / (2 * PADDING + 1) * image_size + 15]
                  for idx in range(4)}

    # get the list of image files (jpg, png) in the input directory
    file_paths = [os.path.join(root, file) for root, directory, files in os.walk(input_dir) 
                 for file in files if file.endswith('.jpg') or file.endswith('.png')]

    # iterate over all image files and generate partial faces
    for file_path in tqdm(file_paths, desc='generating partial faces... ', unit='imgs'):
        try:
            img = cv2.imread(file_path)  # read the image
        except (IOError, ValueError, IndexError) as e:  # handle file reading errors
            print('{}: {}'.format(file_path, e))
        else:
            img = img[:, :, 0:3] - 128  # normalize the image
            for part in PARTS:  # iterate through each facial part
                for factor in FACTORS:  # iterate through cropping factors
                    # create a mask for the specific facial part
                    roi = np.zeros(img.shape[0:2], dtype=np.uint8)
                    x1 = int(point_dict[part][0] - (15 + 120 * factor))
                    y1 = int(point_dict[part][1] - (10 + 105 * factor))
                    x2 = int(point_dict[part][0] + (15 + 120 * factor))
                    y2 = int(point_dict[part][1] + (10 + 105 * factor))
                    cv2.rectangle(roi, (x1, y1), (x2, y2), 1, -1)  # draw rectangle mask over the part

                    # apply the mask to the image to create a partial face
                    partial = cv2.bitwise_and(img, img, mask=roi) + 128

                    # save the partial face image
                    output_file_path = file_path.replace(input_dir, os.path.join(output_dir, part, 'factor_' + str(factor))).replace('.jpg', '.png')
                    if not os.path.exists(output_file_path):
                        if not os.path.exists(os.path.dirname(output_file_path)):
                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        cv2.imwrite(output_file_path, partial)

                    # shift the partial face to center it in the output image
                    dx = 80 - point_dict[part][0]
                    dy = 80 - point_dict[part][1]
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    partial_centered = cv2.warpAffine(partial, M, (image_size, image_size), borderValue=(128, 128, 128))

                    # save the centered partial face image
                    output_file_path = file_path.replace(input_dir, os.path.join(output_dir, part + '_centered', 'factor_' + str(factor))).replace('.jpg', '.png')
                    if not os.path.exists(output_file_path):
                        if not os.path.exists(os.path.dirname(output_file_path)):
                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        cv2.imwrite(output_file_path, partial_centered)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        help='path to the dataset which will be aligned.',
                        default='/mnt/ssd/datasets/LFW/LFW_aligned')
    parser.add_argument('--output_dir', type=str,
                        help='path to the directory for the partial faces.',
                        default='/mnt/ssd/datasets/LFW/PartialLFW')
    parser.add_argument('--image_size', type=int,
                        help='image size (height, width) in pixels.', default=160)
    main(parser.parse_args(sys.argv[1:]))
