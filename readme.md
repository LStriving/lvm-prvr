## Reproduce settings
### text:
- truncate with max text length: 77

### image:
- keep last image: True (padding with black)
- lib: decord (no ffmpeg or denseflow), read with (446, 336) first
- resize to: (224, 224) or (336, 336) (no center crop)
- grid: 2x2
- saveing format: jpg
- decode: jpeg4py

### video:
- all videos in val_1 are extracted except 16 videos: (smil gpu)
    ```
    v_xAMZGWqRmqE
    v_5-vAXCUN8X0
    v_hHMqyl_Dugs
    v_a8dUtKcAunw
    v_7xpkFhlxo2Q
    v_MXDeLfF5rok
    v_Iiwz1JtC7rk
    v_A7oh6l1AIvs
    v_-M-Dr6HqDhU
    v_lgB0Ynn38-k
    v_PT4x_Y5lu_g
    v_K3sJnHGHQHM
    v_0dkIbKXXFzI
    v_-SCRtjT7dto
    v_RTwa2d6Oqvo
    v_VhzPqd0Su5I
    ```
- average frame num: 15.054

### checkpoints:
_VITL14 = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"

_VITL14_336 = "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"


## Procedure
### Description
#### use CLIP model to do video-text retrieval
##### stage1: extract the video embeddings and text embeddings and save them
    video: 
        extracted with 0.5 fps;
        resize the images and fill them into fixed size grid (N*N) layerout as superimages sequentially;
        for example, if the grid size is 2, then the superimage (shape: (224, 224,3) or (336, 336, 3)) is
        
        [
            [img1, img2],
            [img3, img4]
        ]
        
        then feed superimages into CLIP model to get the video embeddings;
    text:
        fill the text into fixed size(77, ) or (128, ) grid layerout sequentially;
        then feed into CLIP model to get the text embedding;

##### stage2: aggregate visual embeddins; calculate the similarity between video embeddings and text embeddings and rank the videos

    retrieval:
        using the text embedding as query to softmax the video embeddings
        aggregate the softmaxed video embeddings to get the final video embedding
        using the final video embedding to calculate the similarity with the text embedding
        using the similarity to rank the videos and calculate the recall@K


### 1. Extract and create super image sequence 
```bash
python super_image_fill.py --save_dir ../dataset/out336 --anno_file ./ms-sl/activitynet/TextData/val_1.json --video_dir /mnt/cephfs/dataset/activitynet_video/all_videos/ --num_workers 32 --grid_size 2 --super_img_size 336 --resume --fps 0.5
```

### 2. Extract text feature
```bash
python process_text_embedding.py --model_path ViT-L/14 --batch_size 128 --anno_path ./annos/val_1.json
```

### 3. Extract image feature 
```bash
# (on A800)
 python image_embeddings.py --batch_size 100  --num_gpus 1 --num_worker 16 --ckpt /liyirui/project/LVM-prvr/CLIP/pretrained_model/ViT-L-14.pt --image_size 224
# local
python generate_imaga_embeddings.py --batch_size 20
```

### 4. Calculate similarity
```bash

```