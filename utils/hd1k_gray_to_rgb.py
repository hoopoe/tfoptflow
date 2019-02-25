import numpy as np
import argparse
import random
import os 
import json
import cv2
from pathlib import Path
from tqdm import tqdm

def convert(input_folder, output_folder):
  data_path = Path(input_folder)
  for file_name in tqdm(list((data_path).glob('000000_*'))):
    out_path = Path(output_folder) / file_name.name
    gray = cv2.imread(str(file_name), 0)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(str(out_path), rgb)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", dest="input", help="input folder with images", default="D:/datasets/hd1k_full_package/hd1k_input/image_2_orig")
  parser.add_argument("-o", "--output", dest="output", help="output folder with images", default="D:/datasets/hd1k_full_package/hd1k_input/image_2")
  args = parser.parse_args()

  Path(args.output).mkdir(parents=True, exist_ok=True)

  convert(args.input, args.output)