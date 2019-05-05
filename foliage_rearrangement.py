# -*- coding: utf-8 -*-
"""
Created on Sun May  5 12:06:48 2019

@author: kkrao
"""

###SCRIPT DOES NOT WORK FOR e, k, AND x SPECIES
import os
import pandas as pd

os.chdir(r'D:\Krishna\DL\leafnet\dataset_all\foliage')
lib = pd.read_excel(r'D:\Krishna\DL\leafnet\foliage_datatable.xls')

lib["Prefix"] = lib["Prefix"].str.replace(" ", "")
## sort the table whith descending order of species length
lib = lib.loc[lib["Prefix"].str.len().sort_values(ascending = False).index]

count = 0
for dataset in ['train_original', 'test_original']:
    for filename in os.listdir(dataset):
        count+=1
#        check which species is the image of
        for species in lib["Prefix"]:
            if species in filename.replace(".tif",""):
                new_path = dataset.replace('original','rearranged')+os.sep+species
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
                #move the file
                os.rename(dataset+os.sep+filename,new_path+os.sep+filename)
                # once the file is moved dont match the file with any other species
                break
        if count%100==0:    
            print('[INFO] Processed {:5d} images'.format(count))