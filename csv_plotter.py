# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:40:29 2019

@author: gatv
"""

import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file',help='Path of csv file')
    parser.add_argument('--monitor',help='value to plot')
    args = vars(parser.parse_args())
    
    df = pd.read_csv(args['file'])
    lines = df[args['monitor']].plot.line()
    