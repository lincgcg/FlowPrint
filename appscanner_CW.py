#!/usr/bin/python3
#-*- coding:utf-8 -*-
import psutil
import mmap
import numpy as np
import json
import os
import time
import xlrd
import pickle
import binascii
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import scapy.all as scapy
from scipy.stats import skew,kurtosis
import sys
import csv
csv.field_size_limit(sys.maxsize)
import copy
import tqdm
import random
import shutil
import argparse
import binascii
import psutil
from PIL import Image
from sklearn.model_selection import train_test_split

from appscanner.preprocessor import Preprocessor
from appscanner.appscanner_CW   import AppScanner
from sklearn.preprocessing   import MinMaxScaler
from sklearn.metrics import classification_report


def dataset_extract(random_seed, pcap_path, samples, class_num, target_dir):
    
    
    preprocessor = Preprocessor()
    # Load flows from file 'flows.p'
    X, y = preprocessor.load(os.path.join(target_dir, "flows.p"))
    # Split data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    # Scale features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    
    # Pass through appscanner
    scanner = AppScanner(threshold=0.9)

    # Fit appscanner with flows and labels
    scanner.fit(X_train, y_train)

    # Predict labels of test data
    y_pred = scanner.predict(X_test)

    # Print report with 4 digit precision
    print(classification_report(y_test, y_pred, digits=4))

def main():
    
    parser = argparse.ArgumentParser(description='Test for argparse')
    

    # 保存最终png文件的地方
    parser.add_argument("--target_dir", type=str,
                        help='''Path of the middle files(e.g., "/Volumes/LCG_2/Datasets/ISCX-VPN/application/finetune/woDCS/result/")''')
    # 类别数量
    parser.add_argument("--class_num", type=int,
                        help='''class number(e.g., 20)''')
    # random seed
    parser.add_argument("--random_seed", type=int,
                        help='''random seed''')
    # pacp文件的目录
    parser.add_argument("--pcap_path", type=str,
                        help='''Path of the pcap dataset path(e.g., "/Users/cglin/Desktop/DCS/application/sliced/")''')
    
    args = parser.parse_args()
    
    samples = [5000]
    samples = samples * args.class_num
    
    
    dataset_extract(args.random_seed, args.pcap_path, samples, args.class_num, args.target_dir)


if __name__ == '__main__':
    main()
