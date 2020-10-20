# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-09-25 09:35:30
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:06:39

import cv2
import os
import argparse
from tqdm import tqdm as progressBar

"""
Generate squared patches of a determined size from each image in an speficied folder
It uses a sliding method with overlapping specified in code as step_size
"""

class patchGenerator():
    def __init__(self,image_folder, size, mode,step):
        self.image_folder = image_folder
        self.window_size = size
        self.mode = mode
        self.step = step     
        self.counter_images = 0 #needed to specifiy a unique name for each patch
        
    def slidingWindow(self,image,image_name):
        # slide a window across the image
        counter_windows = 0
        step_size = self.window_size // self.step
        patch_dir = ''
        new_width, new_height = image.shape[1],image.shape[0]
        flag_resize =  False
        #Resize image if it does not fit the proportions
        if image.shape[0] % step_size != 0:
            new_height = (image.shape[0]//step_size)*step_size
            flag_resize = True
        if image.shape[1] % step_size != 0:
            new_width = (image.shape[1]//step_size)*step_size
            flag_resize = True
        if flag_resize == True:
            image = cv2.resize(image, (new_width,new_height), interpolation = cv2.INTER_CUBIC)
        
        #Reco mode generate a unique folder for ALL images
        if self.mode == 'reco':
            patch_dir = os.path.abspath(os.path.join(self.image_folder, os.pardir))+'/patches_'+str(self.window_size)+'_'+self.image_folder.split('/')[-2]+'/'
        #CD mode generates a folder for EACH image
        elif self.mode == 'cd':
            patch_dir = os.path.abspath(os.path.join(self.image_folder, os.pardir))+'/patches_'+str(self.window_size)+'_'+image_name+'/'
        if not os.path.isdir(patch_dir):
            os.makedirs(patch_dir)
    
        #Slides the image
        for y in range(0, image.shape[0], step_size):
            for x in range(0, image.shape[1], step_size):
                # yield the current window
                # obtains a UNIQUE_KEY filename for future joinFolder applyment
                file_name = patch_dir+self.image_folder.split('/')[-4]+'_'+str(counter_windows).rjust(5,'0')+'_'+str(image_name.split('.')[0])+'.jpg'
                img_window = image[y:y + self.window_size,x:x + self.window_size]
                    #print('Folder path: ',file_name)
                if(img_window.shape == (self.window_size,self.window_size,3)):
                    cv2.imwrite(file_name,img_window)
                counter_windows += 1
        
        
    def main(self):
        for image in progressBar(sorted(os.listdir(self.image_folder))):
            image_name = image.split('.')[0]
            self.slidingWindow(cv2.imread(self.image_folder+image),image_name)
            self.counter_images += 1
            
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('image_folder',type=str,help='set root of images')
    parser.add_argument('--size',type=int,help='set size of patches',required=True)
    parser.add_argument('-m','--mode',metavar='mode',type=str,help='set mode: reco - cd', default='reco')
    parser.add_argument('--step',type=int,help='set step of patches',default=1)
    args = parser.parse_args()
    
    patchGen = patchGenerator(args.image_folder, args.size, args.mode, args.step)
    patchGen.main()



