# this file turns every video in the split into saved DINOv2 features, repeated for all 268 videos.
# We run dinov2 on every video, and save the patch tokens + cls tokens for each video in a separate file. The output is a folder of 268 .pt files, one for each video.
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from dinov2_encoder import (
    load_dinov2,
    preprocess
)

from extract_video import (
    extract_frames,
    encode_frames,
    crop_box_from_grays
)

# full paths: Python does not expand "~" to the home folder
SPLIT_PATH = "/home/medcvr/dataset/split/split.csv"
DATASET_ROOT = "/home/medcvr/dataset"
OUTPUT_ROOT = "/home/medcvr/dinov2_features"

# set to a small number (e.g. 3) to test on a few videos first; None runs everything
MAX_VIDEOS = None


def build_path(row):
    """turns a split row into the file or folder on disk, """
    if row["source"] == "LDPolypVideo-labelled":
        split_parts = row["video"].split("/")
        return os.path.join(DATASET_ROOT, "LDPolypVideo", "labelled", split_parts[0], "Images", split_parts[1])
    elif row["source"] == "LDPolypVideo-presence":
        if row["label"] == "1":
            return os.path.join(DATASET_ROOT, "LDPolypVideo", "pos_videos", "videos with polyps", row["video"])
        return os.path.join(DATASET_ROOT, "LDPolypVideo", "neg_videos", "videos without polyps", row["video"])
    elif row["source"] == "HyperKvasir":
        return os.path.join(DATASET_ROOT, "HyperKvasir", "videos", row["video"])

    raise ValueError(f"Unknown source: {row['source']}")


def extract_folder_frames(folder_path, num_samples=10):
    """ the reader for labelled videos: every .jpg in the folder, in order, cropped."""
    paths = sorted(Path(folder_path).glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"No frames found in the folder {folder_path}")

    raw_frames = []
    for path in paths:
        frame = cv2.imread(str(path))  # BGR, like video frames
        if frame is None:
            raise RuntimeError(f"Failed to read image {path}")
        raw_frames.append(frame)

    # find the crop box from about 10 evenly spaced frames
    picks = np.linspace(0, len(raw_frames) - 1, num_samples).astype(int)
    gray_frames = [cv2.cvtColor(raw_frames[i], cv2.COLOR_BGR2GRAY) for i in picks]
    x, y, w, h = crop_box_from_grays(gray_frames)

    # use the crop box from those 10 evenly spaced frames to crop all frames, and preprocess them for DINOv2
    frames = []
    for frame in raw_frames:
        frame = frame[y:y+h, x:x+w]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(preprocess(Image.fromarray(frame)))

    return frames


def main():
    # load DINOv2 once, before the loop
    model, device = load_dinov2()
    print("Device:", device)

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    rows = [row for row in csv.DictReader(open(SPLIT_PATH))
            if row["split"] not in ("unused", "excluded")]
    if MAX_VIDEOS is not None:
        rows = rows[:MAX_VIDEOS]

    failed = []
    for i, row in enumerate(rows, start=1):
        # "Test/101" would otherwise be read as a folder, so replace the slash
        output_path = os.path.join(OUTPUT_ROOT, row["video"].replace("/", "_") + ".pt")

        # skip finished videos, so the run can be stopped and resumed
        if os.path.exists(output_path):
            print(f"[{i}/{len(rows)}] skip {row['video']} (already done)")
            continue

        try:
            path = build_path(row)

            # labelled LDPolypVideo videos are frame folders; everything else is a video file
            if row["source"] == "LDPolypVideo-labelled":
                frames = extract_folder_frames(path)
            else:
                frames = extract_frames(path)

            cls, patch = encode_frames(model, frames, device)

            torch.save({"cls": cls, "patch": patch, "label": int(row["label"])}, output_path)
            print(f"[{i}/{len(rows)}] {row['video']}: cls {tuple(cls.shape)}, patch {tuple(patch.shape)}")

        except Exception as e:
            # one bad video should not stop the whole run
            failed.append(row["video"])
            print(f"[{i}/{len(rows)}] FAILED {row['video']}: {e}")

    print(f"Done. {len(rows) - len(failed)} of {len(rows)} videos OK.")
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    main()
