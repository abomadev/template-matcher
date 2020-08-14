# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
from pathlib import Path

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the image file")
args = vars(ap.parse_args())


template_path_file = args["image"]

template_filename = Path(template_path_file).stem


# load the image from disk
image = cv2.imread(template_path_file)


# loop over the rotation angles again ensuring
# no part of the image is cut off
for angle in np.arange(-60, 61, 1):
    rotated = imutils.rotate_bound(image, angle)
    if angle < 0:
        deg = f"n{abs(angle)}"
    else:
        deg = f"{angle}"
        
    filename = f"templates/{template_filename}-{deg}.png"
    cv2.imwrite(filename,rotated)