import numpy as np
import argparse
import random
import os 
import json
# import cv2
from pathlib import Path
from tqdm import tqdm
from flowlib import evaluate_flow_file

def convert(ground_truth_folder, predicted_folder):
  data_path = Path(ground_truth_folder)
  h = set()
  
  for file_name in tqdm(list((data_path).glob('*'))):
    series, num = file_name.stem.split("_")
    h.add(series)

  _IDs = []
  for i in h:
    frames = sorted(list((data_path).glob('{}*'.format(i))))
    idx = 0
    while idx < len(frames) - 1:
      _IDs.append(frames[idx])
      idx += 1

  err_list = []
  for i in tqdm(_IDs):
    predicted_path = Path(predicted_folder) / i.name
    average_pe = evaluate_flow_file(i, predicted_path)
    err_list.append(average_pe)

  reduce(lambda x, y: x + y, err_list) / len(err_list)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-g", "--ground_truth", dest="ground_truth", help="input folder with ground_truth flow", default="D:/datasets/hd1k_full_package/hd1k_flow_gt/flow_occ")
  parser.add_argument("-p", "--predicted", dest="predicted", help="predicted folder with predicted flow", default="D:/datasets/flow_occ_pred_by_pwc_net/flow_occ_pred")
  args = parser.parse_args()

  convert(args.ground_truth, args.predicted)