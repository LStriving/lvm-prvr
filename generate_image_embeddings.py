import clip
import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import jpeg4py as jpeg
import argparse

class ImageDataset(Dataset):
    def __init__(self,root,transforms) -> None:
        super().__init__()
        assert os.path.exists(root), f"{root} does not exist"
        self.root = root
        self.video_names = os.listdir(root)
        self.transforms = transforms
        self.frame_names = []
        for video_name in os.listdir(root):
            video_imags = os.listdir(os.path.join(root,video_name))
            if len(video_imags) == 0:
                print(f'{video_name} has no frames extracted!')
                # remove the video name from the list
                self.video_names.remove(video_name)
                continue
        self.frame_names = [
            [
                os.path.join(video_name,frame_name)
                for frame_name in sorted(os.listdir(os.path.join(root,video_name)), key=lambda x: int(x[4:-4]))
            ]
            for video_name in self.video_names
        ]
        self.frame_ids = [video_name[2:] for video_name in self.video_names]
        assert len(self.frame_names) == len(self.video_names), "frame names and video names should have the same length"

    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, index):
        frame_names = self.frame_names[index]
        frame_vid = self.video_names[index]
        frames = []
        for frame_name in frame_names:
            frame_path = os.path.join(self.root,frame_name)
            frame = jpeg.JPEG(frame_path).decode()
            frame = Image.fromarray(frame)
            if self.transforms is not None:
                frame = self.transforms(frame)
            frames.append(frame)
        frames = torch.stack(frames)
        frame_id = [frame_vid[2:] for _ in range(len(frames))]
        # frame_id = torch.tensor(frame_id)
        return frames, frame_id

def infer_worker(args):
    image_dataloader,model,output_root = args
    with torch.no_grad():
        for _, (images, vid) in tqdm(enumerate(image_dataloader)):
            images = images.cuda()
            image_features = model.module.encode_image(images).float()  # N_queries x 512
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().numpy()
            vid = np.array(vid)

            # save the features by video according to the vid
            unique_vids = np.unique(vid)
            for cur_id in unique_vids:
                # keep the frames within the same video by selecting with the mask
                cur_vid_features = image_features[vid == cur_id]
                # save the features
                np.save(f"{output_root}/{cur_id}.npy", cur_vid_features)

def collate_fn(batch):
    images, vid = zip(*batch)
    images = torch.cat(images)
    vid = [item for sublist in vid for item in sublist]
    return images, vid

def infer(
        batch_size=32,
        num_gpus=1,
        num_workers=4,
        image_size=336,
        image_path="./extracted336_grid2",
        output="./features/image_features",
        resume=False
    ):
    # create output folder
    output += f"@{image_size}px"
    if not os.path.exists(output):
        os.makedirs(output)
    # load model first before creating dataset
    if image_size == 336:
        model,image_preprocess = clip.load("ViT-L/14@336px")
    elif image_size == 224:
        model,image_preprocess = clip.load("ViT-L/14")
    print("Model loaded")
    print("Image preprocess: ",image_preprocess)
    # model,image_preprocess = clip.load("ViT-L/14px")
    image_dataloader = DataLoader(ImageDataset(image_path,transforms=image_preprocess),
                                  batch_size=batch_size,shuffle=False,num_workers=num_workers,collate_fn=collate_fn)
    
    # Use DataParallel for distributed processing
    model = torch.nn.DataParallel(model,device_ids=[i for i in range(num_gpus)])
    
    args = (image_dataloader,model,output)
    infer_worker(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=336, choices=[224, 336])
    parser.add_argument("--image_path", type=str, default="./extracted336_grid2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output", type=str, default="./features/image_features")
    parser.add_argument("--resume",action="store_true")

    args = parser.parse_args()

    infer(
        args.batch_size,
        args.num_gpus,
        args.num_workers,
        args.image_size,
        args.image_path,
        args.output,
        args.resume
    )

