import os
import json
import decord
from tqdm import tqdm
import multiprocessing as mp

def write_json(data,save_path):
    with open(save_path,'w')as f:
        json.dump(data,f,indent=4, ensure_ascii=False)


def get_video_info(video_path):
    try:
        vr = decord.VideoReader(video_path)
        video_info = {}
        video_info['video_name'] = os.path.basename(video_path)
        video_info['video_frames'] = len(vr)
        video_info['video_fps'] = vr.get_avg_fps()
        video_info['video_duration'] = video_info['video_frames']/video_info['video_fps']
    except:
        print(f'{video_path} error!')
        video_info = {}
        video_info['video_name'] = os.path.basename(video_path)
        video_info['video_frames'] = 0
        video_info['video_fps'] = 0
        video_info['video_duration'] = 0
    return video_info

def get_video_info_from_dir(video_dir):
    video_info_list = []
    for video_name in os.listdir(video_dir):
        video_path = os.path.join(video_dir,video_name)
        video_info = get_video_info(video_path)
        video_info_list.append(video_info)
    return video_info_list

def mp_get_video_info(video_dir,video_list=None):
    video_info_list = []
    if video_list is None:
        video_path_list = [os.path.join(video_dir,video_name) for video_name in os.listdir(video_dir)]
    else:
        video_path_list = [os.path.join(video_dir,video_name+'.mp4') for video_name in video_list]
        video_path_list = [video_path if os.path.exists(video_path) else (video_path[:-4] + '.mkv') for video_path in video_path_list ]
    for video_path in video_path_list:
        if not os.path.exists(video_path):
            print(f'{video_path} not exists!')
    pool = mp.Pool(mp.cpu_count())
    video_info_list = list(tqdm(pool.imap(get_video_info,video_path_list),total=len(video_path_list)))
    pool.close()
    pool.join()
    return video_info_list

if __name__ == '__main__':
    video_dir = './anet_videos'
    anno = './annos/val_1.json'
    with open(anno,'r') as f:
        anno = json.load(f)
    video_list = list(anno.keys())
    video_info_list = mp_get_video_info(video_dir,video_list)
    save_path = 'video_info.json'
    write_json(video_info_list,save_path)
    print('Done!')
