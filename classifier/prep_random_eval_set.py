# combine all sets (train, val, test) and randomly sample eval set from them
# train set size input from parameter
# val set size fixed to 1
# test set is the rest
# original files for the sets are in groundtruth_swap folder
# save new sets in groundtruth_random_eval folder. 
#     note: every time this is run, new sets are re-generated, 
#     the folder needs to be backed up into numbered subfolders
#     e.g., groundtruth_random_eval_1, groundtruth_random_eval_2, etc.
#     the groundtruth_random_eval folder is always only have the latest sets

import os
import pandas as pd
import numpy as np
import sys
import shutil
import random

def prep_random_eval_set(base_path, gt_input_folder, gt_output_folder, train_size, random_seed=7):
    # read original csv files
    # no header row assumed
    train_df = pd.read_csv(os.path.join(base_path, gt_input_folder, 'manual_train.csv'), header=None)
    val_df = pd.read_csv(os.path.join(base_path, gt_input_folder, 'manual_validation.csv'), header=None)
    test_df = pd.read_csv(os.path.join(base_path, gt_input_folder, 'test.csv'), header=None)
    
    # combine all data
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # shuffle the combined data
    combined_df = combined_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # sample new train, val, test sets
    new_train_df = combined_df.iloc[:train_size]
    new_val_df = combined_df.iloc[train_size:train_size+1]
    new_test_df = combined_df.iloc[train_size+1:]
    
    # backup existing output folder if it exists
    if os.path.exists(os.path.join(base_path, gt_output_folder)):
        existing_folders = [d for d in os.listdir(os.path.join(base_path, gt_output_folder)) if d.startswith(gt_output_folder+'_')]
        next_index = len(existing_folders) + 1
        backup_folder = f"{gt_output_folder}_{next_index}"
        # move each exsting csv file to backup folder
        os.makedirs(os.path.join(base_path, gt_output_folder, backup_folder))
        for file_name in ['manual_train.csv', 'manual_validation.csv', 'test.csv']:
            src_path = os.path.join(base_path, gt_output_folder, file_name)
            if os.path.exists(src_path):
                dst_path = os.path.join(base_path, gt_output_folder, backup_folder, file_name)
                shutil.move(src_path, dst_path)
        print(f"Backed up existing folder to {backup_folder}")
    else:
        # create output folder if it doesn't exist
        os.makedirs(os.path.join(base_path, gt_output_folder))
        print(f"Created new folder {gt_output_folder}")

    # save new csv files
    new_train_df.to_csv(os.path.join(base_path, gt_output_folder, 'manual_train.csv'), index=False)
    new_val_df.to_csv(os.path.join(base_path, gt_output_folder, 'manual_validation.csv'), index=False)
    new_test_df.to_csv(os.path.join(base_path, gt_output_folder, 'test.csv'), index=False)
    
    print(f"New random eval sets created in {gt_output_folder} with train size {train_size}")

def main():
    if len(sys.argv) != 6:
        print("Usage: python prep_random_eval_set.py <base_path> <gt_input_folder> <gt_output_folder> <train_size> <random_seed>")
        print("Example: python prep_random_eval_set.py \"\" groundtruth_swap groundtruth_random_eval 1000 7")
        sys.exit(1)
    
    base_path = sys.argv[1]
    gt_input_folder = sys.argv[2]
    gt_output_folder = sys.argv[3]
    train_size = int(sys.argv[4])
    random_seed = int(sys.argv[5])
    
    print(f"Base path: {base_path}")
    print(f"Ground truth input folder: {gt_input_folder}")
    print(f"Ground truth output folder: {gt_output_folder}")
    print(f"Train set size: {train_size}")
    print("==============================================================")
    
    prep_random_eval_set(base_path, gt_input_folder, gt_output_folder, train_size, random_seed)
    
    print("==============================================================")

if __name__ == "__main__":
    print(len(sys.argv))
    for i, arg in enumerate(sys.argv):
        print(f"Arg {i}: {arg}")
    main()