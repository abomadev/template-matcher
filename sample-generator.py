import cv2
import numpy as np
from pathlib import Path
import sys
import argparse
import imutils

def preview_images(*imgs):
    for i, img in enumerate(imgs, start=1):
        # print(f"{i}")
        cv2.imshow(f'Image {i}', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# None wrapped
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

def top_center_roi_vertices(base, template):
    base_h, base_w, base_channels =  base.shape
    template_h, template_w, template_channels =  template.shape
    # 1 - Define region of interest
    ## Top right
    # 1.1 - Define the top left overlay point
    top_left_x = round(base_w/2 - template_w/2)
    top_left_y = 0

    # 1.2 - Define the bottom right overlay point base on the size of the template
    bottom_right_x = top_left_x + template_w
    bottom_right_y = template_h
    
    return top_left_x, top_left_y, bottom_right_x, bottom_right_y  


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

def bottom_center_roi_vertices(base, template):
    base_h, base_w, base_channels =  base.shape
    template_h, template_w, template_channels =  template.shape
    # 1 - Define region of interest
    ## Top right
    # 1.1 - Define the top left overlay point
    top_left_x = round(base_w/2 - template_w/2)
    top_left_y = base_h - template_h

        
    # 1.2 - Define the bottom right overlay point base on the size of the template
    bottom_right_x = top_left_x + template_w
    bottom_right_y = base_h
    
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



# Wrapped
def right_wrapped_roi_vertices(base, template):
    base_h, base_w, base_channels =  base.shape
    template_h, template_w, template_channels =  template.shape
    # 1 - Define region of interest
    ## Top right
    # 1.1 - Define the top left overlay point
    top_left_x = base_w - template_w
    top_left_y = 0


    # 1.2 - Define the bottom right overlay point base on the size of the template
    top_roi = {}
    bottom_roi = {}
    
    top_roi["top_left_x"] = top_left_x
    top_roi["top_left_y"] = top_left_y
    top_roi["bottom_right_x"] = top_left_x + template_w
    top_roi["bottom_right_y"] = top_left_y + int(template_h/2)
    
    bottom_roi["top_left_x"] = top_left_x
    bottom_roi["top_left_y"] = base_h - (template_h - int(template_h/2))
    bottom_roi["bottom_right_x"] = top_left_x + template_w
    bottom_roi["bottom_right_y"] = base_h
        
    return top_roi, bottom_roi

def left_wrapped_roi_vertices(base,template):
    base_h, base_w, base_channels =  base.shape
    template_h, template_w, template_channels =  template.shape
    # 1 - Define region of interest
    ## Top left
    # 1.1 - Define the top left overlay point
    top_left_x = 0
    top_left_y = 0

    # 1.2 - Define the bottom right overlay point base on the size of the template
    
    top_roi = {}
    bottom_roi = {}
    
    top_roi["top_left_x"] = top_left_x
    top_roi["top_left_y"] = top_left_y
    top_roi["bottom_right_x"] = top_left_x + template_w
    top_roi["bottom_right_y"] = top_left_y + int(template_h/2)
    
    bottom_roi["top_left_x"] = top_left_x
    bottom_roi["top_left_y"] = base_h - (template_h - int(template_h/2))
    bottom_roi["bottom_right_x"] = top_left_x + template_w
    bottom_roi["bottom_right_y"] = base_h
        
    return top_roi, bottom_roi 

def center_wrapped_roi_vertices(base, template):
    base_h, base_w, base_channels =  base.shape
    template_h, template_w, template_channels =  template.shape
    # 1 - Define region of interest
    ## Top right
    # 1.1 - Define the top left overlay point
    top_left_x = round(base_w/2 - template_w/2)
    top_left_y = 0


    # 1.2 - Define the bottom right overlay point base on the size of the template
    top_roi = {}
    bottom_roi = {}
    
    top_roi["top_left_x"] = top_left_x
    top_roi["top_left_y"] = top_left_y
    top_roi["bottom_right_x"] = top_left_x + template_w
    top_roi["bottom_right_y"] = top_left_y + int(template_h/2)
    
    bottom_roi["top_left_x"] = top_left_x
    bottom_roi["top_left_y"] = base_h - (template_h - int(template_h/2))
    bottom_roi["bottom_right_x"] = top_left_x + template_w
    bottom_roi["bottom_right_y"] = base_h
    
        
    return top_roi, bottom_roi

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
    elif position == 'tc':
        x_offset, y_offset, x_end, y_end = top_center_roi_vertices(base, template)
    elif position == 'bc':
        x_offset, y_offset, x_end, y_end = bottom_center_roi_vertices(base, template)    
    else:
        print("ERROR - not a valid position")
    roi = base[y_offset:y_end, x_offset:x_end]

    # 2 - Overlay the template over the roi and specify opacity levels alpha and beta
    blended_roi = cv2.addWeighted(src1=roi,alpha=base_opacity,src2=template,beta=template_opacity, gamma=0)

    # # 3 - Override the base image's numpy array to the template image
    base[y_offset:y_end, x_offset:x_end] = blended_roi
    
    return base

def create_wraped_sample(base, template, template_opacity, position='l', base_opacity=1):
    ''' 
    In order to overlay two images of different sizes, need to cut out a region of interest where we want to place the smaller image of the exact same size
    of the smaller image, then overlay them. Then override the base image's numpy
    array with the result. 
    '''
    # 0 - Perform any rotations if needed
    base = base.copy()
    tw, th,c = template.shape

    


    # 1 - Define region of interest
    if position == 'l':
        bottom_roi, top_roi = left_wrapped_roi_vertices(base, template)
    elif position == 'r':
        bottom_roi, top_roi = right_wrapped_roi_vertices(base, template)
    elif position == 'c':
        bottom_roi, top_roi = center_wrapped_roi_vertices(base, template)
    else:
        print("ERROR - not a valid position")
        
    print(f"{top_roi['top_left_y']}:{top_roi['bottom_right_y']}, {top_roi['top_left_x']}:{top_roi['bottom_right_x']}")
    print(f"{bottom_roi['top_left_y']}:{bottom_roi['bottom_right_y']}, {bottom_roi['top_left_x']}:{bottom_roi['bottom_right_x']}")
    # Get top roi
    roi = base[top_roi["top_left_y"]:top_roi["bottom_right_y"], top_roi["top_left_x"]:top_roi["bottom_right_x"]]
    h1 = top_roi['bottom_right_y'] - top_roi['top_left_y']
    h2 = bottom_roi['bottom_right_y'] - bottom_roi['top_left_y']
    template_top = template[:h1,:]
    template_bottom = template[h1:,:]
    
    
    # cv2.imshow(f"Template {template.shape}", template)
    # cv2.imshow(f"Top {template_top.shape}", template_top)
    # cv2.imshow(f"Bottom {template_bottom.shape}", template_bottom)
    # cv2.imshow(f"ROI {roi.shape}", roi)
    # cv2.waitKey(0) 
    
    
    
    
    # 2 - Overlay the template over the roi and specify opacity levels alpha and beta
    top_blended_roi = cv2.addWeighted(src1=roi,alpha=base_opacity,src2=template_top,beta=template_opacity, gamma=0)
    # # 3 - Override the base image's numpy array to the template image
    base[top_roi["top_left_y"]:top_roi["bottom_right_y"], top_roi["top_left_x"]:top_roi["bottom_right_x"]] = top_blended_roi
 
     # Get bottom roi
    roi = base[bottom_roi["top_left_y"]:bottom_roi["bottom_right_y"], bottom_roi["top_left_x"]:bottom_roi["bottom_right_x"]]
    # 2 - Overlay the template over the roi and specify opacity levels alpha and beta
    bottom_blended_roi = cv2.addWeighted(src1=roi,alpha=base_opacity,src2=template_bottom,beta=template_opacity, gamma=0)
    # # 3 - Override the base image's numpy array to the template image
    base[bottom_roi["top_left_y"]:bottom_roi["bottom_right_y"], bottom_roi["top_left_x"]:bottom_roi["bottom_right_x"]] = bottom_blended_roi
            
    return base

#### Start ####



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the image file")
ap.add_argument("-t", "--template", required=True,
	help="path to the template file")
ap.add_argument("--wrap", action="store_true")
args = vars(ap.parse_args())

# args = sys.argv


base_file_path = args["image"]
template_path_file = args["template"]

base_filename = Path(base_file_path).stem
template_filename_arr = Path(template_path_file).stem.split("-")
suit = template_filename_arr[0]
angle = template_filename_arr[2]


# base_img = cv2.imread(base_file_path)
# # base_img = cv2.cvtColor(base_img,cv2.COLOR_BGR2RGB)
# template_img = cv2.imread(template_path_file)

# suits = ["heart","club","spade","diamond"]
# cats = ["01","02","03","04","05","06","07","08","09","10","11"]

# positions = ['tl', 'tr', 'bl', 'br','cr']
positions = ['tc', 'bc']
wrapped_positions = ['l', 'r','c']
opacities = [25, 50, 75, 100]

# for c in cats:
#     for s in suits:
        # base_filename = f"cat-sample-{c}"
        # base_file_path = f"samples/{base_filename}.png"
        # template_path_file = f"templates/{s}-template.png"

base_img = cv2.imread(base_file_path)
# base_img = cv2.cvtColor(base_img,cv2.COLOR_BGR2RGB)
template_img = cv2.imread(template_path_file)

if not args["wrap"]:
    for p in positions:
        for o in opacities:
            sample_img = create_sample(base_img, template_img, o/100, p)
            filename = f"samples/{base_filename}-{suit}-{angle}-{p}-{o}.png"
            cv2.imwrite(filename,sample_img)

if args["wrap"]:
    for p in wrapped_positions:
        o = 100
        sample_img = create_wraped_sample(base_img, template_img, o/100, p)
        filename = f"samples/{base_filename}-{suit}-{angle}-{p}-{o}-wrapped.png"
        cv2.imwrite(filename,sample_img)
# sample_img_tl = create_sample(base_img, template_img, 0.5, 'tl')
# sample_img_tr = create_sample(base_img, template_img, 0.5, 'tr')
# sample_img_bl = create_sample(base_img, template_img, 0.5, 'bl')
# sample_img_br = create_sample(base_img, template_img, 0.5, 'br')


# preview_images(sample_img_tl, sample_img_tr, sample_img_bl, sample_img_br)
