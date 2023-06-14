import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sys import platform
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
import re

if platform == 'win32':
    os.environ['CDF_LIB'] = "C:\\Users\janeg\Desktop\CDF\\bin"
    curr_path = os.getcwd()
    temp_dir = "\\tempdir"
    full_path = curr_path+temp_dir
else:
    os.environ["CDF_LIB"] = "/usr/bin/"#"/uio/hume/student-u58/janeod/.conda/envs/SVM/lib/python3.10/site-packages/spacepy"
    cdfpath = ...

from spacepy import pycdf

def retrive_urls():
    locs = ["gill", "atha", "chbg", "ekat", "fsim", "fsmi", 
            "fvkn", "gako", "gbay", "gill", "inuv", "kapu",
            "kian", "kuuj", "mcgr", "nrsq", "pgeo", "rank",
            "snap", "talo", "tpas", "whit", "yknf"]
    months = ["10", "11", "12", "01", "02"]
    
    os.chdir('urls')
    for loc in tqdm(locs):
        urls = []
        for month in tqdm(months):
            if month == "01" or month == "02":
                year = "2013"
            else:
                year = "2012"

            cdf_url = f"http://themis.ssl.berkeley.edu/data/themis/thg/l1/asi/{loc}/{year}/{month}/"
            r = requests.get(cdf_url)
            soup = BeautifulSoup(r.text, 'html.parser')

            for link in soup.find_all('a'):
                if re.findall(".*_asf_.*", link.get('href')):
                    urls.append(f"{cdf_url}{link.get('href')} ")
        file = open(f'urls_{loc}.txt', 'w')
        file.writelines(urls)
        file.close()
    os.chdir('..')

#retrive_urls()

file = open('urls.txt', 'r')
urls = file.readline().split()
file.close()
url = urls[0]

if not os.path.exists(full_path):
    os.mkdir(full_path)

os.chdir(full_path)

"""
Do stuff
"""

# Download cdf-file

r = requests.get(url)
a = open("temp.cdf", 'wb')
a.write(r.content)
a.close()

# Process cdf-file

# Delete cdf-file

os.remove("temp.cdf")

os.chdir('..')
os.rmdir(full_path)

# Save pandas as feather