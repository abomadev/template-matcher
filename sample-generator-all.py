import cv2
import numpy as np
from pathlib import Path
import sys
import argparse
import imutils
import os

def preview_images(*imgs):
    for i, img in enumerate(imgs, start=1):
        # print(f"{i}")
        cv2.imshow(f'Image {i}', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def top_right_roi_vertices(base, template):
    base_h, base_w, base_channels =  base.shape
    template_h, template_w, template_channels =  template.shape
    # 1 - Define region of interest
    ## Top right
    # 1.1 - Define the top left overlay point
    top_left_x = base_w - template_w
    top_left_y = 0


    # 1.2 - Define the bottom right overlay point base on the size of the template
    bottom_right_x = top_left_x + template.shape[1]
    bottom_right_y = top_left_y + template.shape[0]
    
    return top_left_x, top_left_y, bottom_right_x, bottom_right_y  

def top_left_roi_vertices(base,template):
    base_h, base_w, base_channels =  base.shape
    template_h, template_w, template_channels =  template.shape
    # 1 - Define region of interest
    ## Top left
    # 1.1 - Define the top left overlay point
    top_left_x = 0
    top_left_y = 0

    # 1.2 - Define the bottom right overlay point base on the size of the template
    bottom_right_x = top_left_x + template_w
    bottom_right_y = top_left_y + template_h
        
    return top_left_x, top_left_y, bottom_right_x,bottom_right_y 

def bottom_left_roi_vertices(base, template):
    base_h, base_w, base_channels =  base.shape
    template_h, template_w, template_channels =  template.shape
    # 1 - Define region of interest
    ## Top right
    # 1.1 - Define the top left overlay point
    top_left_x = 0
    top_left_y = base_h - template_h


    # 1.2 - Define the bottom right overlay point base on the size of the template
    bottom_right_x = top_left_x + template_w
    bottom_right_y = top_left_y + template_h
    
    return top_left_x, top_left_y, bottom_right_x, bottom_right_y  

def bottom_right_roi_vertices(base, template):
    base_h, base_w, base_channels =  base.shape
    template_h, template_w, template_channels =  template.shape
    # 1 - Define region of interest
    ## Top right
    # 1.1 - Define the top left overlay point
    top_left_x = base_w - template_w
    top_left_y = base_h - template_h


    # 1.2 - Define the bottom right overlay point base on the size of the template
    bottom_right_x = top_left_x + template_w
    bottom_right_y = top_left_y + template_h
    
    return top_left_x, top_left_y, bottom_right_x, bottom_right_y  

def center_roi_vertices(base, template):
    base_h, base_w, base_channels =  base.shape
    template_h, template_w, template_channels =  template.shape
    # 1 - Define region of interest
    ## Top right
    # 1.1 - Define the top left overlay point
    top_left_x = round(base_w/2 - template_w/2)
    top_left_y = round(base_h/2 - template_h/2)


    # 1.2 - Define the bottom right overlay point base on the size of the template
    bottom_right_x = top_left_x + template_w
    bottom_right_y = top_left_y + template_h
    
    return top_left_x, top_left_y, bottom_right_x, bottom_right_y  

def create_sample(base, template, template_opacity, position='tl', base_opacity=1):
    ''' 
    In order to overlay two images of different sizes, need to cut out a region of interest where we want to place the smaller image of the exact same size
    of the smaller image, then overlay them. Then override the base image's numpy
    array with the result. 
    '''
    # 0 - Perform any rotations if needed
    base = base.copy()
    
    # 1 - Define region of interest
    if position == 'tl':
        x_offset, y_offset, x_end, y_end = top_left_roi_vertices(base, template)
    elif position == 'tr':
        x_offset, y_offset, x_end, y_end = top_right_roi_vertices(base, template)
    elif position == 'bl':
        x_offset, y_offset, x_end, y_end = bottom_left_roi_vertices(base, template)
    elif position == 'br':
        x_offset, y_offset, x_end, y_end = bottom_right_roi_vertices(base, template)
    elif position == 'cr':
        x_offset, y_offset, x_end, y_end = center_roi_vertices(base, template)
    else:
        print("ERROR - not a valid position")
    roi = base[y_offset:y_end, x_offset:x_end]

    # 2 - Overlay the template over the roi and specify opacity levels alpha and beta
    blended_roi = cv2.addWeighted(src1=roi,alpha=base_opacity,src2=template,beta=template_opacity, gamma=0)

    # # 3 - Override the base image's numpy array to the template image
    base[y_offset:y_end, x_offset:x_end] = blended_roi
    
    return base


def generate(base_file_path,template_path_file):

    base_filename = Path(base_file_path).stem
    template_filename_arr = Path(template_path_file).stem.split("-")
    suit = template_filename_arr[0]
    angle = template_filename_arr[2]

    positions = ['tl', 'tr', 'bl', 'br','cr']
    opacities = [25, 50, 75, 100]

    # for c in cats:
    #     for s in suits:
            # base_filename = f"cat-sample-{c}"
            # base_file_path = f"samples/{base_filename}.png"
            # template_path_file = f"templates/{s}-template.png"

    base_img = cv2.imread(base_file_path)
    # base_img = cv2.cvtColor(base_img,cv2.COLOR_BGR2RGB)
    template_img = cv2.imread(template_path_file)

    for p in positions:
        for o in opacities:
            sample_img = create_sample(base_img, template_img, o/100, p)
            filename = f"samples/{base_filename}-{suit}-{angle}-{p}-{o}.png"
            cv2.imwrite(filename,sample_img)
#### Start ####



templates = []
samples = []

directory = "originals"
for filename in os.listdir(directory):
# for filename in ["spade-template-0.png","heart-template-0.png","club-template-0.png","diamond-template-0.png"]:
    if "sample" in filename:
        samples.append(os.path.join(directory, filename))
        # print(os.path.join(directory, filename))
    else:
        continue
    
directory = "templates"
for filename in os.listdir(directory):
# for filename in ["spade-template-0.png","heart-template-0.png","club-template-0.png","diamond-template-0.png"]:
    if filename.endswith(".png"):
        templates.append(os.path.join(directory, filename))
        # print(os.path.join(directory, filename))
    else:
        continue
# args = sys.argv

print(len(samples))
print(len(templates))

for s in samples:
    for t in templates:
        generate(s,t)

# sample_img_tl = create_sample(base_img, template_img, 0.5, 'tl')
# sample_img_tr = create_sample(base_img, template_img, 0.5, 'tr')
# sample_img_bl = create_sample(base_img, template_img, 0.5, 'bl')
# sample_img_br = create_sample(base_img, template_img, 0.5, 'br')


# preview_images(sample_img_tl, sample_img_tr, sample_img_bl, sample_img_br)
