"""
Text Video Retrieval
query-attentive similarity: 
aggreate img features using text features and calculate cosine similarity
query: text features
"""
import os
import pickle as pkl
import torch
import numpy as np
from tqdm import tqdm

# calculate Query-attentive similarity by batches
def calc_attn(text_query, vis_features):
    """
    text_query: (B, 768)
    vis_features: (T, 768)
    output: (B, T)
    """
    attn = torch.matmul(text_query, vis_features.T) # (B, T)
    attn = torch.softmax(attn, dim=1)
    return attn

def aggrete_vis_features(vis_features, attn):
    """
    vis_features: (T, 768)
    attn: (B, T)
    output: (1, 768)
    """
    attn = attn.unsqueeze(2) # (B, T, 1)
    # attn -> (B, T, 768)
    attn = attn.repeat(1, 1, vis_features.shape[1])
    vis_features = vis_features.unsqueeze(0) # (1, T, 768)
    # vis_features -> (B, T, 768)
    vis_features = vis_features.repeat(attn.shape[0], 1, 1)
    vis_features = attn * vis_features
    vis_features = vis_features.sum(dim=1) # (B, 768)
    return vis_features

def calc_similarity(text_query, vis_features):
    """
    text_query: (B, 768)
    vis_features: (T, 768)
    output: (B, 1)
    """ 
    attn = calc_attn(text_query, vis_features)
    vis_features = aggrete_vis_features(vis_features, attn)
    cos_sim = torch.cosine_similarity(text_query, vis_features, dim=1)
    return cos_sim

def calc_similarity_all_text(text_pkl_root:str, vis_features):
    """
    text_pkl_root: the root of text features
    vis_features: (T, 768)
    output: 
        sim: (all_text_num,)
        vid_id: (all_text_num,)
    """
    assert os.path.exists(text_pkl_root), f'{text_pkl_root} not exists!'
    sim = []
    vid_id = []
    text_pkl_list = os.listdir(text_pkl_root)
    sorted(text_pkl_list,key=lambda x: int(x.split('_')[1].split('.')[0]))
    text_pkl_list = [os.path.join(text_pkl_root, pkl) for pkl in text_pkl_list]
    for text_pkl in tqdm(text_pkl_list):
        with open(text_pkl,'rb')as f:
            text_features = pkl.load(f)
        cos_sim = calc_similarity(torch.tensor(text_features['text_features']), vis_features)
        sim.append(cos_sim)
        vid_id.append(text_features['vid'])
    sim = torch.cat(sim, dim=0)
    vid_id = np.concatenate(vid_id, axis=0)
    return sim, vid_id


def calc_sim_all_videos(text_pkl_root:str, visual_root:str):
    # load visual features
    assert os.path.exists(visual_root), f'{visual_root} not exists!'
    assert os.path.exists(text_pkl_root), f'{text_pkl_root} not exists!'
    visual_feat_list = os.listdir(visual_root)
    
    tmp_result_path = './tmp_result(video2text)'
    if not os.path.exists(tmp_result_path):
        os.makedirs(tmp_result_path)

    for visual_feat in tqdm(visual_feat_list):
        visual_feat_path = os.path.join(visual_root, visual_feat)
        visual_features = np.load(visual_feat_path)
        visual_features = torch.tensor(visual_features)
        
        sim, vid_id = calc_similarity_all_text(text_pkl_root, visual_features)
        save_path = os.path.join(tmp_result_path, visual_feat[:-4] + '_sim.pkl')
        with open(save_path, 'wb') as f:
            pkl.dump({'sim': sim, 'vid_id': vid_id}, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_pkl_root', type=str, default='./features/text_features@224px')
    parser.add_argument('--visual_root', type=str, default='./features/image_features@224px')
    args = parser.parse_args()
    
    calc_sim_all_videos(args.text_pkl_root, args.visual_root)
