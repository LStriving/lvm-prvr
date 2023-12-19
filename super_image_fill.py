import numpy as np
import decord
import os
from PIL import Image
import einops
import json
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import multiprocessing

def create_superimage_sequences(images, grid_size, super_img_size, target_video_image_size):
    # Calculate the number of superimages needed
    num_superimages = math.ceil(len(images) / (grid_size ** 2))

    superimages = []

    for i in range(num_superimages):
        # Calculate the range of images for the current superimage
        start_idx = i * (grid_size ** 2)
        end_idx = (i + 1) * (grid_size ** 2)

        # Select the images for the current superimage
        current_images = images[start_idx:end_idx]

        # Calculate the number of rows and columns in the grid
        rows = min(grid_size, len(current_images))
        cols = min(grid_size, len(current_images))

        # Create an empty superimage
        superimage = np.zeros((super_img_size,super_img_size, 3), dtype=np.uint8)

        # Populate the superimage with the sub-images
        for j, img in enumerate(current_images):
            row = j // cols
            col = j % cols
            # resize the images to target_video_image_size
            img = Image.fromarray(img)
            img = img.resize((target_video_image_size, target_video_image_size))
            img = np.array(img)
            superimage[row * target_video_image_size: (row + 1) * target_video_image_size, col * target_video_image_size: (col + 1) * target_video_image_size] = img

        superimages.append(superimage)

    return superimages

def process_video(video_file_name, video_dir, save_dir, anno_file, fps, grid_size, super_img_size, num_workers):
    video_path = os.path.join(video_dir, video_file_name)
    if not os.path.exists(video_path):
        video_path = os.path.join(video_dir, video_file_name.split(".")[0] + ".mkv")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_file_name} does not exist.")
    vid = video_file_name.split(".")[0]
    image_dir = os.path.join(save_dir, vid)
    # create the save directory if it does not exist
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    video_reader = decord.VideoReader(video_path,
                                    num_threads=1,
                                    width=446, height=336,ctx=decord.cpu(0))
    original_fps = video_reader.get_avg_fps()
    frame_num = len(video_reader)
    target_frame_num = int(frame_num * fps / original_fps)
    frame_idx = np.linspace(0, frame_num, target_frame_num, endpoint=False, dtype=int)
    imgs = video_reader.get_batch(frame_idx).asnumpy()

    result_superimage_sequences = create_superimage_sequences(imgs, grid_size, super_img_size, super_img_size // grid_size)

    # Convert and save the result superimages as separate files
    for i, superimage in enumerate(result_superimage_sequences):
        result_image = Image.fromarray(superimage)
        save_path = os.path.join(image_dir, f"img_{(i + 1):04d}.jpg")
        result_image.save(save_path)

def main(video_dir, fps, grid_size, super_img_size, save_dir, anno_file, debug=False, resume=False, num_workers=8):
    assert os.path.exists(video_dir), f"Video directory {video_dir} does not exist."
    assert os.path.exists(anno_file), f"Annotation file {anno_file} does not exist."

    with open(anno_file, "r") as f:
        anno = json.load(f)
    video_names = list(anno.keys())
    video_file_names = [video_name + ".mp4" for video_name in video_names]
    if debug:
        video_file_names = video_file_names[:10]

    video_file_names = sorted(video_file_names,reverse=True)
    pool = multiprocessing.Pool(processes=num_workers)
    for video_file_name in tqdm(video_file_names):
        pool.apply_async(process_video, (video_file_name, video_dir, save_dir, anno_file, fps, grid_size, super_img_size, num_workers))
    pool.close()
    pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create superimage sequences from video frames.')
    parser.add_argument('--video_dir', type=str, help='Path to the directory containing video files.',default='/liyirui/dataset/anet_videos')
    parser.add_argument('--fps', type=float, default=0.5, help='Frames per second for the superimage sequences.')
    parser.add_argument('--grid_size', type=int, default=2, help='Size of the grid for arranging sub-images in each superimage.')
    parser.add_argument('--super_img_size', type=int, default=336, choices=[224,336], help='Size of the superimage.')
    parser.add_argument("--save_dir", type=str, default="./output", help="Path to save the superimage sequences.")
    parser.add_argument("--anno_file", type=str, default="/liyirui/project/mssl/activitynet/TextData/val_1.json", help="Path to the annotation file.")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode.")
    parser.add_argument("--resume", action="store_true", help="Whether to resume", default=True)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for multiprocessing.")

    args = parser.parse_args()

    main(args.video_dir, args.fps, args.grid_size, args.super_img_size, args.save_dir, args.anno_file, args.debug, args.resume, args.num_workers)
