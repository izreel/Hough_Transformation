import cv2
import numpy as np
import opencv_hough
import custom_hough

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("-i", "--image", dest="image",
                        help="specify the name of the image", metavar="IMAGE")
    args = parser.parse_args()
    input_image = cv2.imread(args.image, 0)

    if '.jpg' in args.image:
        part_output_name = args.image[len(args.image) - 5]
        
        #built in hough transform from OpenCV
        #opencv_hough.hough_transform(input_image, part_output_name)
        
        #self implemented hough trasformation
        #custom_hough.hough_transformation(input_image, part_output_name)

main()