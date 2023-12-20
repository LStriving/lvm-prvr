"""
calculate the similarity between text and video through video-text similarity batch
"""
import os
import pickle as pkl
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

'''video-text similarity batch (Video-num, Text-num)'''

def main(args):
    assert os.path.exists(args.pickle_root), f"{args.pickle_root} does not exist"
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load all the pickle files
    pickle_files = os.listdir(args.pickle_root)
    pickle_files = [f for f in pickle_files if f.endswith('.pkl')]

    all_sim = []
    text_id = []
    video_id = []

    for video in tqdm(pickle_files):
        with open(os.path.join(args.pickle_root, video), 'rb') as f:
            data = pkl.load(f)
        all_sim.append(data['sim'].numpy())
        video_id.append(video.split('.')[0]) # str
    text_id = data['vid_id'] # (17505,) (str)
    all_sim = np.array(all_sim) # (Video-num, Text-num) (4901,17505)
    # transpose the similarity matrix
    scores = all_sim.T # (Text-num, Video-num) (17505,4901)
    # save
    np.save(os.path.join(args.save_dir, 'scores.npy'), scores)
    # ground truth (text2video)
    video_vid = np.array([vid[:-4] for vid in video_id]) # (4901,) (str)

    # remove broken videos text
    broken = []
    for vid in text_id:
        if vid[2:] not in video_vid:
            broken.append(vid)
    

    R1 = 0
    R5 = 0
    R10 = 0
    R100 = 0
    for i, gt_id in tqdm(enumerate(text_id)):
        if gt_id in broken:
            continue
        sim = scores[i] # (Video-num,) (4901,)
        rank = np.argsort(sim)[::-1]
        # order the video_id according to the similarity
        sim = [video_vid[i] for i in rank][:100]
        gt_id = gt_id[2:]
        assert len(sim[0]) == len(gt_id)
        if gt_id in sim[:1]:
            R1 += 1
        if gt_id in sim[:5]:
            R5 += 1
        if gt_id in sim[:10]:
            R10 += 1
        if gt_id in sim[:100]:
            R100 += 1
    R1 /= len(text_id) - len(broken)
    R5 /= len(text_id) - len(broken)
    R10 /= len(text_id) - len(broken)
    R100 /= len(text_id) - len(broken)

    print("R@1:", R1*100)
    print("R@5:", R5*100)
    print("R@10:", R10*100)
    print("R@100:", R100*100)
    print("SumR:", (R1+R5+R10+R100)*100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_root', type=str, default='./tmp_result(video2text)', help='path to pickle files root')
    parser.add_argument('--save_dir', type=str, default='./retrival_result/clip_224', help='path to save similarity')
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()

    main(args)
