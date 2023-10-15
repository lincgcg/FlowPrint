from flowprint.preprocessor import Preprocessor
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
from flowprint.flowprint import FlowPrint
# Imports
from sklearn.metrics import classification_report


def dataset_extract(random_seed, pcap_path, samples, class_num, target_dir):
    # 遍历pcap文件
    dataset = {}
    label_name_list = []
    session_pcap_path  = {}

    ## 获取每个label的name和dir
    for parent, dirs, files in os.walk(pcap_path):
        if label_name_list == []:
            label_name_list.extend(dirs)

        for dir in label_name_list:
            for p,dd,ff in os.walk(parent + "/" + dir):
                session_pcap_path[dir] = pcap_path + dir
        break

    # 建立label name 到label ID的map
    label_id = {}
    for index in range(len(label_name_list)):
        label_id[label_name_list[index]] = index

    r_file_record = []
    print("\nBegin to generate features.")
    label_count = 0
    print(session_pcap_path.keys())
    
    # Used for Flowprint
    files_data = []
    labels_data = []
    
    for key in tqdm.tqdm(session_pcap_path.keys()):
        
        target_all_files = [x[0] + "/" + y for x in [(p, f) for p, d, f in os.walk(session_pcap_path[key])] for y in x[1]]
        random.seed(random_seed)
        target_all_files.sort()
        r_files = random.sample(target_all_files, samples[label_count] if len(target_all_files) > samples[label_count] else len(target_all_files))
        
        for file in r_files:
            files_data.append(file)
            labels_data.append(key)
    
    # Create Preprocessor object
    preprocessor = Preprocessor(verbose=True)
    # Create Flows and labels
    X, y = preprocessor.process(files = files_data,
                                labels= labels_data)

    # Save flows and labels to file 'flows.p'
    preprocessor.save(os.path.join(target_dir, "flows.p"), X, y)
    
    # Load flows from file 'flows.p'
    X, y = preprocessor.load(os.path.join(target_dir, "flows.p"))
    # Split data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    # Create FlowPrint object
    flowprint = FlowPrint(
        batch       = 300,
        window      = 30,
        correlation = 0.1,
        similarity  = 0.9
    )

    # Fit FlowPrint with flows and labels
    flowprint.fit(X_train, y_train)

    # Create fingerprints for test data
    fp_test = flowprint.fingerprint(X_test)
    # Predict best matching fingerprints for each test fingerprint
    y_pred = flowprint.recognize(fp_test)

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
