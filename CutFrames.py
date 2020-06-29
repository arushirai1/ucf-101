import os
from Utils import build_paths, create_frames, h5_dump, h5_load
import argparse
import numpy as np
import h5py

#### Paths #############################################################################################################

class_idxs, train_split, test_split, frames_root, remaining = build_paths()

#####
# Add: if pretrain == resume: error

### Data ###############################################################################################################

print('\n==> Preparing Data...\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CutFrames')

    parser.add_argument('--test', action='store_true', help='Test or train')
    parser.add_argument('--file', type=int, default=0,
                        help='file no')
    parser.add_argument('--batchsize', type=int, default=200,
                        help='file no')
    parser.add_argument('--h5file', type=str, default="",
                        help='path to h5 file')
    args=parser.parse_args()
    split=None
    if(args.test):
        split=test_split
    else:
        split=train_split

    split_path = split
    paths = []

    for pth in split_path:
        with open(pth) as f:
            for line in f:
                class_name, vid_id = line.strip().split()[0][:-4].split('/')
                # print(class_name, vid_id)
                paths.append((vid_id, class_name))
    class_dict = {}
    with open(class_idxs) as f:
        for line in f:
            label, class_name_key = line.strip().split()
            if class_name_key not in class_dict:
                class_dict[class_name_key] = []
            class_dict[class_name_key] = int(label) - 1
    i=0
    batch_size =args.batchsize
    while i < len(paths):
        all_frames=[]
        labels=[]
        for vid_id, class_name in paths[i:i+batch_size]:
            vid_dir = os.path.join(frames_root, class_name, vid_id)
            frame_count = 16
            frames = create_frames(vid_dir, frame_count)
            if frames is not None:
                label = np.array(class_dict[class_name], dtype=int)
                print(vid_id)
                all_frames.append(frames)
                labels.append(label)
                i += 1
        all_frames = np.array(all_frames).astype(np.float32)#np.append([all_frames[0]], [all_frames[1:]], axis=0).astype(np.float32)
        labels=np.array(labels)
        print(all_frames.shape)
        with h5py.File(args.h5file, 'a') as hf:
            hf.create_dataset('data_batch%d' %(i//batch_size),  data=all_frames)
            hf.create_dataset('labels_batch%d' %(i//batch_size), data=labels)
            print("Successfully created: ", "data+label")

    print('Number of Classes: %d' % len(class_dict))
    print('Number of Videos loaded: %d' % labels.shape)
    #print('Number of Testing Videos: %d' % len(testset))

