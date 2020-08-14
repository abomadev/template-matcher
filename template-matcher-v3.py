import cv2
import numpy as np 
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
import os
from matplotlib.pyplot import figure, draw, pause




class MatchTemplate:
    def __init__(self, basepath, showtype):

        self.best_template_name = ''
        base_img = cv2.imread(basepath)

        h,w,channels = base_img.shape

        
        self.best_sum_under_mask = 0
        self.best_shift = 0
        self.best_match_img = None
        self.best_template = None
        self.best_difference = None
        self.threshold_base = None
        self.best_template_base = None

        directory = "templates"
        
        
        self.fg = figure()
        
        self.ax = self.fg.gca()

        self.h = self.ax.imshow(base_img)  # set initial display dimensions

        # for img in imgs:
        #     h.set_data(img)
        #     draw(), pause(1e-3)


        
        # plt.imshow(base_img)
        for filename in os.listdir(directory):
        # for filename in ["spade-template-0.png","heart-template-0.png","club-template-0.png","diamond-template-0.png"]:
            if filename.endswith(".png"):
                self.find_match(self.grayscale(base_img),os.path.join(directory, filename), showtype)
            else:
                continue
        # # 45200
        original = cv2.cvtColor(base_img,cv2.COLOR_BGR2RGB)
        # shifted_result = np.roll(original, self.best_shift)
        
        # # difference = cv2.subtract(self.best_match_img,self.best_template_base)
        # # cv2.imshow("best_template_base", self.best_template_base)
        # # cv2.imshow("BASE", original)
        # cv2.imshow(f"shifted_result: {self.best_shift}", shifted_result)
        # # # cv2.imshow("Shifted", self.shift(img2, shift))
        # cv2.waitKey(0) 
            
        # plt.subplot(321),plt.imshow(original)
        # plt.title(f'Original Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(322),plt.imshow(self.threshold_base,cmap = 'gray')
        # plt.title(f'Threshold of {self.threshold}'), plt.xticks([]), plt.yticks([])
        # plt.subplot(323),plt.imshow(self.best_template,cmap = 'gray')
        # plt.title('Template'), plt.xticks([]), plt.yticks([])
        # plt.subplot(324),plt.imshow(self.best_template_base,cmap = 'gray')
        # plt.title('Resized Template'), plt.xticks([]), plt.yticks([])
        # plt.subplot(325),plt.imshow(self.best_difference,cmap = 'gray')
        # plt.title(f'Diff - {self.best_sum_under_mask}'), plt.xticks([]), plt.yticks([])
        # plt.subplot(326),plt.imshow(self.best_match_img,cmap = 'gray')
        # plt.title('Best Match'), plt.xticks([]), plt.yticks([])
        # plt.suptitle(f"Image: {os.path.basename(basepath)}\nBest Template: {self.best_template_name}\n\n")

        # plt.show()
        
        print(f"Image: {os.path.basename(basepath)}\nBest Template: {self.best_template_name}\n\n")

        # cv2.rectangle(self.best_match_img,self.top_left, self.bottom_right, (0,255,0), 2)

        # self.best_match_img = cv2.cvtColor(self.best_match_img,cv2.COLOR_BGR2RGB)

  

                    
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
        # Create copy of base image
        base_img = base.copy()
        
        # Read in template image
        template_path = templatepath
        template_filename = Path(template_path).stem
        template_img = cv2.imread(template_path,0)

        # Create an all black image of the same size as the base image. This will be used to create the resulting template
        template_base = cv2.convertScaleAbs(np.zeros(base_img.shape))
        
        # Create an all black image of the same size as the template image
        template_all_black = cv2.convertScaleAbs(np.zeros(template_img.shape))
        
        
        # 2 - Overlay the template over the all black template to remove the transparent background
        template_roi = cv2.addWeighted(src1=template_all_black,alpha=1,src2=template_img,beta=1, gamma=0)
        
        # # 3 - Override the base image's numpy array to the template image
        
        base_h, base_w = template_base.shape
        template_h, template_w = template_roi.shape
        
        top_left_x = round(base_w/2 - template_w/2)
        top_left_y = base_h - template_h
        bottom_right_x = top_left_x + template_w
        bottom_right_y = base_h
        
        # Create the final template which consists of a black background of the same size as the base image and with the template
        # embedded centerally at the bottom.
        template_base[top_left_y:bottom_right_y,top_left_x:bottom_right_x] = template_roi

        template_mask = template_base

        
        for i in range(base_h):
            shift_amount = i*base_w
            shifted_base = np.roll(base_img, int(shift_amount))
            
            
            mask_applied = cv2.bitwise_or(shifted_base, shifted_base, mask=template_mask)
            
            # cv2.imshow(f"Mask Applied: {mask_applied.shape} {}", mask_applied)
            # cv2.waitKey(0) 
            
            sum_under_mask = np.sum(mask_applied)
        

            # cv2.imshow("Original", base_img)
            # # cv2.imshow("Threshold", thresh1)
            # cv2.imshow(f"shifted_base: {i}", shifted_base)
            # cv2.imshow("template_base", template_base)
            # cv2.imshow("Difference", difference)
            # # cv2.imshow("Shifted", self.shift(img2, shift))
            # cv2.waitKey(0) 
        
            # self.tmph.set_data(img2)
            # draw(), pause(1e-3)
            # print(f"{i}: {img3}")

            if sum_under_mask > self.best_sum_under_mask:
                self.best_template_name = template_filename
                self.best_sum_under_mask = sum_under_mask
                self.best_match_img = mask_applied
                self.threshold_base = base_img
                self.best_template = template_img
                self.best_template_base = template_base
                
                # cv2.imshow("Original", base_img)
                # cv2.imshow("Threshold", thresh1)
                # cv2.imshow(f"mask_applied: {i}", mask_applied)
                # cv2.imshow("template_base", template_base)
                # cv2.imshow("Difference", difference)
                # cv2.imshow("Shifted", self.shift(img2, shift))
                # cv2.waitKey(0) 
            
                
                # self.best_match_img = img
                self.best_shift = shift_amount
                
                if showtype == 'best':
                    # Show improvements
                    print(f'Best:{self.best_template_name} - {self.best_shift} - {self.best_sum_under_mask} | Curr: {template_filename} - {sum_under_mask}')
                    self.fg.suptitle(f'Matching:{template_filename} BM:{sum_under_mask}')
                    self.h.set_data(mask_applied)
                    draw(), pause(1e-4)
                    # self.show_match(img,template,max_loc,w,h)
            if showtype == 'all':
                # Show improvements
                self.h.set_data(mask_applied)
                # self.fg.suptitle(f'Matching:{template_filename} BM:{sum_under_mask}')
                
                draw(), pause(1e-4)
            
                    
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
# print(args, show_type)
MatchTemplate(args["image"], show_type)

