# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-09-25 09:35:30
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:23:07
import argparse
import os
from shutil import copyfile
from tqdm import tqdm as ProgressBar
"""
Joins a list of folders into a new one,with the same filenames
"""
def joinFolders(new_folder_name,folder_list):
    #obtains absolute path of output folder
    new_folder_path = os.path.abspath(new_folder_name)
    if os.path.isdir(new_folder_path) != True:
        os.makedirs(new_folder_path)
    for folder in folder_list:
        folder_path = os.path.abspath(folder)
        print('folder_path:',folder_path)
        for filename in ProgressBar(os.listdir(folder_path)):
            #copies file from original folder to the new one
            copyfile(os.path.join(folder_path,filename),os.path.join(new_folder_path,filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='join files of many folders into a new one')
    parser.add_argument('new_folder_name',help='new_folder_name')
    parser.add_argument('-f','--folder_list',type=str,nargs='+',required=True,help='write path to folders to merge')
    args = parser.parse_args()
    joinFolders(args.new_folder_name,args.folder_list)