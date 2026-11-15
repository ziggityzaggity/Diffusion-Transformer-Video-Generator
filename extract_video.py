# Generate with help from GPT-6
import cv2
import torch
from pathlib import Path
from PIL import Image
import numpy as np

from dinov2_encoder import (
    load_dinov2,
    extract_tokens,
    preprocess
)

# Video to process
VIDEO_PATH = "/home/medcvr/dataset/LDPolypVideo/pos_videos/videos with polyps/1_9.avi"

# Where extracted features will be saved
OUTPUT_PATH = "patch_tokens/test_video_tokens.pt"

# Number of frames processed together
BATCH_SIZE = 8

# we need to find the crop box for the endoscope image area, so we can crop out the black border
def find_crop_box(cap, num_samples=10):
    """Find the endoscope image area (x, y, w, h) from frames spread through the video."""

    # count total frames
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # pick 10 evenly spaced frames to find the crop box
    positions = np.linspace(0, total - 1, num_samples).astype(int)

    # convert each 
    gray_frames = []
    for i in positions:
        # jump to the frame position and read the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        success, frame = cap.read()

        if not success:
            continue

        # convert to grayscale
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    return crop_box_from_grays(gray_frames)


# shared by videos (find_crop_box) and frame folders, which have no video capture
def crop_box_from_grays(gray_frames):
    """Find the endoscope image area (x, y, w, h) from a list of grayscale frames."""

    if not gray_frames:
        raise RuntimeError("Could not read any frames to find the crop box")

    # brightest value at each pixel across the sampled frames
    combined = np.max(np.stack(gray_frames), axis=0)

    # turns the image into black and white.
    corners = [combined[:20, :20], combined[:20, -20:], combined[-20:, :20], combined[-20:, -20:]]
    border_level = np.median(np.concatenate([c.ravel() for c in corners]))
    _, thresh = cv2.threshold(combined, border_level + 15, 255, cv2.THRESH_BINARY)

    # find the outline of each white region
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # no white regions found
    if not contours:
        height, width = combined.shape
        return 0, 0, width, height

    # keep biggest region
    largest_contour = max(contours, key=cv2.contourArea)
    
    # find the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    return x, y, w, h


def extract_frames(video_path):
    """Extract approximately one frame per second."""

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine video FPS")

    x, y, w, h = find_crop_box(cap)  # Find the crop box for the endoscope image area
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
    
    frames = []
    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        # Sample approximately one frame per second
        if frame_index % max(1, round(fps)) == 0:
            
            frame = frame[y:y+h, x:x+w]
            if frame_index == 0:
                # save cropped frame for debugging
                # it crops the date out partly
                cv2.imwrite("cropped_frame.jpg", frame)
                # continue
                
            # OpenCV uses BGR; PIL uses RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            image = Image.fromarray(frame)

            # Apply the same preprocessing as our image test
            frames.append(preprocess(image))

        frame_index += 1

    cap.release()

    if not frames:
        raise RuntimeError("No frames extracted from the video")

    return frames


def encode_frames(model, frames, device):
    """Run DINOv2 over preprocessed frames in batches.

    Returns CLS tokens [T, 384] and patch tokens [T, 256, 384], both float16.
    """

    # Process frames in batches to reduce GPU memory use
    all_tokens = []
    all_cls = []
    for start in range(0, len(frames), BATCH_SIZE):

        # slicing stops at the end of the list, so the smaller last batch needs no special case
        batch = torch.stack(frames[start:start + BATCH_SIZE])

        patch_tokens, batch_cls = extract_tokens(model, batch, device)

        # Save patch tokens in float16 to reduce disk usage
        all_tokens.append(patch_tokens.half())
        # save cls tokens in float16 to reduce disk usage
        all_cls.append(batch_cls.half())

    # Combine tokens from all batches
    return torch.cat(all_cls, dim=0), torch.cat(all_tokens, dim=0)


def main():
    # Load pretrained, frozen DINOv2
    model, device = load_dinov2()

    print("Device:", device)

    # Extract video frames
    frames = extract_frames(VIDEO_PATH)

    print("Frames extracted:", len(frames))

    video_cls_tokens, video_patch_tokens = encode_frames(model, frames, device)

    # 256 patches per frame, each patch token is 384-dimensional
    print("Total tokens:", video_patch_tokens.shape[0] * video_patch_tokens.shape[1])
    
    # Create output folder
    Path(OUTPUT_PATH).parent.mkdir(
        parents=True, exist_ok=True
    )

    # Save extracted patch tokens
    torch.save({"cls": video_cls_tokens, "patch": video_patch_tokens}, OUTPUT_PATH)    
    some_loaded_tokens = torch.load(OUTPUT_PATH)  # sanity check
    print("Loaded patch shape:", some_loaded_tokens["patch"].shape)
    print("Loaded CLS shape:", some_loaded_tokens["cls"].shape)
    print("Patch token shape:", video_patch_tokens.shape)
    print("CLS token shape:", video_cls_tokens.shape)
    print("Saved to:", OUTPUT_PATH)
    print("SUCCESS: Video patch tokens extraction complete!")


if __name__ == "__main__":
    main()
