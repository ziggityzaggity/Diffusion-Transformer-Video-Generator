# Generate with help from GPT-6
import cv2
import torch
from pathlib import Path
from PIL import Image

from dinov2_encoder import (
    load_dinov2,
    extract_patch_tokens,
    preprocess
)

# Video to process
VIDEO_PATH = "samples/test_video.mp4"

# Where extracted features will be saved
OUTPUT_PATH = "patch_tokens/test_video_tokens.pt"

# Number of frames processed together
BATCH_SIZE = 8


def extract_frames(video_path):
    """Extract approximately one frame per second."""

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine video FPS")

    frames = []
    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        # Sample approximately one frame per second
        if frame_index % max(1, round(fps)) == 0:
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


def main():
    # Load pretrained, frozen DINOv2
    model, device = load_dinov2()

    print("Device:", device)

    # Extract video frames
    frames = extract_frames(VIDEO_PATH)

    print("Frames extracted:", len(frames))

    # Process frames in batches to reduce GPU memory use
    all_tokens = []

    for start in range(0, len(frames), BATCH_SIZE):
        
        # a small bug here: the last batch may be smaller than BATCH_SIZE, so we need to handle that case
        if start + BATCH_SIZE > len(frames):
            batch_frames = frames[start:len(frames)]
        else:
            batch_frames = frames[start:start + BATCH_SIZE]
        
        batch = torch.stack(batch_frames)

        patch_tokens = extract_patch_tokens(model, batch, device)

        # Save patch tokens in float16 to reduce disk usage
        all_tokens.append(patch_tokens.half())

        print(
            f"Processed {min(start + BATCH_SIZE, len(frames))}"
            f"/{len(frames)} frames"
        )

    # Combine patch tokens from all batches
    video_patch_tokens = torch.cat(all_tokens, dim=0)

    # Create output folder
    Path(OUTPUT_PATH).parent.mkdir(
        parents=True, exist_ok=True
    )

    # Save extracted patch tokens
    torch.save(video_patch_tokens, OUTPUT_PATH)

    print("Patch token shape:", video_patch_tokens.shape)
    print("Saved to:", OUTPUT_PATH)
    print("SUCCESS: Video patch tokens extraction complete!")


if __name__ == "__main__":
    main()
