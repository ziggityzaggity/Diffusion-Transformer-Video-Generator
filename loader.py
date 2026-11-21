import os
import sys
import csv
class VideoFeatureDataset(Dataset):

    def __init__(self, split_csv, features_dir, split_name):
        # read split_csv
        with open(split_csv, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            self.rows = [row for row in reader if row['split'] == split_name]
        self.features_dir = features_dir

    def __len__():
        # return number of rows

    def __getitem__(i):
        # row = self.rows[i]

        # 1. load the saved tokens
        # file = features_dir / (row.video with "/" replaced by "_") + ".pt"
        # data = load(file)                    # dict with "cls", "patch", "label"
        # cls   = data["cls"]   as float32     # [T, 384]
        # patch = data["patch"] as float32     # [T, 256, 384]
        # label = row.label as float           # 0.0 or 1.0
        # T = number of frames in cls

        # 2. frame labels (only val and test need them)
        # if row.frame_labels == "boxes":
        #     frame_labels = load_frame_labels(row.video)   # [T] of 0/1
        # else if row.frame_labels == "all-negative":
        #     frame_labels = zeros of length T
        # else:                                 # "none": presence-only training videos
        #     frame_labels = None

        # return cls, patch, label, frame_labels


def load_frame_labels(video):                    # video is like "Test/101"
    set_name, number = split video on "/"
    folder = LDPolypVideo/ann/{set_name}/Annotations/{number}
    labels = []
    # for each .txt file in folder, sorted by name:
    #     first line = number of boxes
    #     labels.append(1 if number of boxes > 0 else 0)
    return labels
