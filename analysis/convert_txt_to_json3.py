#!/usr/bin/env python

#usage: python convert_txt_to_json.py -y [enter the year you want to process] -p [enter the pack size]
#  python convert_txt_to_json.py -y 2016_postVFP -p 3
#  python convert_txt_to_json.py -y 2016_preVFP -p 3
#  python convert_txt_to_json.py -y 2017 -p 3
#  python convert_txt_to_json.py -y 2018 -p 3
import uproot
import json
import pyxrootd.client
import fnmatch
import numpy as np
import numexpr
import subprocess
import concurrent.futures
import warnings
import os
import difflib
from optparse import OptionParser
import glob
import yaml
import sys
from datasets import *

parser = OptionParser()
parser.add_option('-y', '--year', help='year', dest='year')
parser.add_option('-p', '--pack', help='pack', dest='pack')
(options, args) = parser.parse_args()

# a dict for dataset and cross section reference
if '2016' in options.year:
    year = options.year.split('_')[0]

else:
    year = options.year
    
yml_file = {}
#yml_file['2016'] = 'UL_data_processing/dataset_yml_files/MonoTop_UL_2016.yml'
#yml_file['2017'] = 'UL_data_processing/dataset_yml_files/MonoTop_UL_2017.yml'
#yml_file['2018'] = 'monotop_customul_2018.yml'
#with open('datasets.py', 'r') as file:
#    data = yaml.safe_load(file)
#    print('data: ', data)

lst = list()
with open('metadata/2017list.txt', 'r') as ff:
    filelist = ff.readlines()
    for fff in filelist:
        file = fff.split('\n')[0]
        lst.append(file)

xs_reference = {}
for dataset in lst:
#     print(dataset,':',data['2018'][dataset]['xs'])
    if 'BTagCSV' in dataset:
        xs_reference[dataset] = -1
    else:    
        xs_reference[dataset] = datasets[year][dataset]


    
#flist = glob.glob('metadata/UL_'+options.year+'/*.txt')
flist = glob.glob('2017ListFiles/*.list')
#metadata/customNano/UL_2018_txtfiles/
datadict_obj = {}
for txt_file in flist:
    print('txt_file: ', txt_file)
    if 'SingleMuon' in txt_file:
        continue
#    if 'MET' not in txt_file:
#        continue

#     print(txt_file)
    
    lstobj = []
    n = int(options.pack)
    with open(txt_file) as f:
        data = f.readlines()
        for filename in data:
            file = filename.split('\n')[0]
            lstobj.append(file)
            
    if not len(lstobj) == len(set(lstobj)):
        print("your dataset has duplicate root files")

    output=[lstobj[i:i + n] for i in range(0, len(lstobj), n)] 
 
    for i, lst in enumerate(output):
        print(txt_file)
        if 'BTagCSV' in txt_file:
            key_name = txt_file.split('/')[1].split('.')[0] + '____'+str(i+1)+'_'
            dataset_name = txt_file.split('/')[1].split('.')[0]
        else:
            key_name = txt_file.split('/')[1].split('.')[0] + '____'+str(i+1)+'_'
            if 'preVFP' in key_name:
                dataset_name = txt_file.split('/')[3].split('.')[0][:-6]
            elif 'postVFP' in key_name:
                dataset_name = txt_file.split('/')[3].split('.')[0][:-7]
            else:
                dataset_name = txt_file.split('/')[1].split('.')[0]
        datadict_obj[key_name] = {'files': lst, 'xs': xs_reference[dataset_name]}
       
import json
#folder = "metadata/UL_"+options.year+'.json'
folder = "metadata/BFF_T2_KISTI_"+options.year+'_v1.json'
with open(folder, "w") as fout:
    json.dump(datadict_obj, fout, indent=4)
    
