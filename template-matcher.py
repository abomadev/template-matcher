import cv2
import numpy as np 
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
import os
from matplotlib.pyplot import figure, draw, pause




class MatchTemplate:
    def __init__(self, basepath, showtype):
        # All the 6 methods for comparison in a list
        # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        # self.methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED']
        self.methods = ['cv2.TM_CCOEFF_NORMED']

        self.best_method = ''
        self.best_template = ''
        self.best_max_val = 0
        self.best_match_img = None
        self.best_match_method = self.methods[0]
                
        base_img = cv2.imread(basepath)

        directory = "templates"
        
        
        fg = figure()
        
        self.ax = fg.gca()

        self.h = self.ax.imshow(base_img)  # set initial display dimensions

        tmpfg = figure()
        
        self.tmpax = tmpfg.gca()

        self.tmph = self.tmpax.imshow(base_img)  # set initial display dimensions
        # for img in imgs:
        #     h.set_data(img)
        #     draw(), pause(1e-3)

        
        # plt.imshow(base_img)
        for filename in os.listdir(directory):
        # for filename in ["spade-template-0.png","heart-template-0.png","club-template-0.png","diamond-template-0.png"]:
            if filename.endswith(".png"):
                self.find_match(base_img,os.path.join(directory, filename), showtype)
                # print(os.path.join(directory, filename))
            else:
                continue

        # Best template should be set at this point
        # template_path = self.best_template
        # template_filename = Path(template_path).stem
        # template = cv2.imread(template_path,0)
        # w, h = template.shape[::-1]


        cv2.rectangle(self.best_match_img,self.top_left, self.bottom_right, (0,255,0), 2)

        self.best_match_img = cv2.cvtColor(self.best_match_img,cv2.COLOR_BGR2RGB)

        self.h.set_data(self.best_match_img)
        fg.suptitle(f'Matching Result: BT:{self.best_template} BM:{self.best_method}')
        draw(), pause(10000)

                    
    def shift(self, img, rowlength):
        return np.roll(img, int(rowlength), axis=0)
        
    def grayscale(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def show_match(self,base,template,max_loc,w,h):
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # roi = base[y_offset:y_end, x_offset:x_end]
        roi = base[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        

        # 2 - Overlay the template over the roi and specify opacity levels alpha and beta
        blended_roi = cv2.addWeighted(src1=roi,alpha=1,src2=template,beta=1, gamma=0)

        # # 3 - Override the base image's numpy array to the template image
        base[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blended_roi

        # base = cv2.cvtColor(base,cv2.COLOR_GRAY2BGR)
        self.h.set_data(base)
        draw(), pause(1e-4)
        

    def find_match(self, base,templatepath, showtype):
        
        template_path = templatepath
        template_filename = Path(template_path).stem

        img2 = self.grayscale(base.copy())

        template = cv2.imread(template_path,0)
        w, h = template.shape[::-1]


        # Shape of base image
        a, b = img2.shape
        # shift = w * (h/3)
        # print(w,h,c,shift,type(base_img))
        # base_img = np.roll(base_img, int(shift), axis=0)
    
        # self.h.set_data(base_img)
        # draw(), pause(10000)
        
        # cv2.imshow("Original", img2)
        # cv2.imshow("Shifted", self.shift(img2, shift))
        # cv2.waitKey(0) 
        # print(f"RANGE: {len(range(h))}")
        # For the height of the template, shift
        for i in range(h):
            shift_amount = i*w
            img3 = self.shift(img2, shift_amount)
        
            # self.tmph.set_data(img2)
            # draw(), pause(1e-3)
            # print(f"{i}: {img3}")
        
            for meth in self.methods:
                img = img3.copy()
                method = eval(meth)

                # Apply template Matching
                res = cv2.matchTemplate(img,template,method)
                
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    
                #     if min_val > self.best_max_val:
                #         self.top_left = min_loc
                #         self.bottom_right = (self.top_left[0] + w, self.top_left[1] + h)
                    
                #         self.best_max_val = min_val
                #         self.best_method = meth
                #         self.best_template = template_filename
                # else:
                if max_val > self.best_max_val:
                    self.top_left = max_loc
                    self.bottom_right = (self.top_left[0] + w, self.top_left[1] + h)
                    
                    self.best_max_val = max_val
                    self.best_method = meth
                    self.best_template = template_filename
                    
                    self.best_match_img = img
                    
                    if showtype == 'best':
                        # Show improvements
                        self.show_match(img,template,max_loc,w,h)
                if showtype == 'all':
                    # Show improvements
                    self.show_match(img,template,max_loc,w,h)
                
                    
                # print(meth, min_val, max_val, self.best_max_val, self.top_left, self.bottom_right)
                
                



    
# technologies:
# Want something that can be written in python for ease of maintainence
# OpenCL or CUDA both have python libraries and can be on a GPU



    
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the image file")
ap.add_argument('--show-best', action='store_true')
ap.add_argument('--show-all', action='store_true')

args = vars(ap.parse_args())

show_type = None
if args["show_all"]:
    show_type = "all"
elif args["show_best"] and not args["show_all"]:
    show_type = "best"
else:
    pass
print(args, show_type)
MatchTemplate(args["image"], show_type)

