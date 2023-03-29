#! /usr/bin/env python3

import os
import sys

if len(sys.argv) != 3:
    print("Usage: python3 mv-json.py [dir with json labels] [dir with raw data]")
    exit(1)

json_dir = sys.argv[1]
data_dir = sys.argv[2]

for file in os.listdir(json_dir):
    index = file.split('_')[0]
    print(f"moving {index} annotations")
    os.system(f'mv {json_dir}/{file} {data_dir}/{index}')

