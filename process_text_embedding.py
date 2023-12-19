import os
import json
import numpy as np
import torch
import clip
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle as pkl

class DescriptionDataset(Dataset):
    def __init__(self, txt_path, preprocess,max_len=77):
        text_descriptions = read_all_text_descriptions(txt_path)
        self.preprocess = preprocess
        self.texts = [text[:max_len] for texts in list(text_descriptions.values()) for text in texts]
        if self.preprocess is not None:
            self.texts = [self.preprocess(text) for text in self.texts]
        self.text_vid = [vid for vid,texts in enumerate(text_descriptions) for _ in range(len(texts))]
        self.texts_tokens = clip.tokenize(self.texts,context_length=max_len)
        print("text tokens shape:", self.texts_tokens.shape)
        self.text_vid = torch.tensor(self.text_vid)

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts_tokens[idx].cuda(), self.text_vid[idx].cuda()

def read_all_text_descriptions(anno_file):
    """
    anno_file: json file
    return: dict, key: video name, value: list of text descriptions
    """
    with open(anno_file,'r') as f:
        anno = json.load(f)
    text_descriptions = {}
    print("anno length:", len(anno))
    for vid,values in anno.items():
        text_descriptions[vid] = values['sentences']
    return text_descriptions

def infer_worker(args):
    description_dataloader,model = args
    with torch.no_grad():
        for i, (text, vid) in tqdm(enumerate(description_dataloader)):
            text_features = model.encode_text(text).float()  # N_queries x 512
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy()
            vid = vid.cpu().numpy()

            # save the features by batch
            data = {'vid': vid, 'text_features': text_features}
            with open(f"./features/text_features@224px/batch_{i}.pkl", 'wb') as f:
                pkl.dump(data, f)

def infer():
    # load model first before creating dataset
    batch_size = 128
    # model, image_preprocess = clip.load("./pretrained_ckpt/ViT-L-14-336px.pt")
    model,image_preprocess = clip.load("ViT-L/14")
    # model,image_preprocess = clip.load("ViT-L/14px")
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    model.cuda().eval()
    description_dataset = DescriptionDataset("./annos/val_1.json", None)
    description_dataloader = DataLoader(description_dataset, batch_size=batch_size, shuffle=False)
    print("model loaded")
    
    infer_worker((description_dataloader,model))
    

if __name__ == '__main__':
    infer()

