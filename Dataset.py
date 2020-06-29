import os, sys, select
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

from Utils import build_paths, create_frames, h5_dump, h5_load


class UCF10(Dataset):
    """
    Args:
        class_idxs (string): Path to list of class names and corresponding label (.txt)
        split (string): Path to train or test split (.txt)
        frames_root (string): Directory (root) of directories (classes) of directories (vid_id) of frames.

        UCF10
        ├── v_ApplyEyeMakeup_g01_c01
        │   ├── 1.jpg
        │   └── ...
        ├── v_ApplyLipstick_g01_c01
        │   ├── 1.jpg
        │   └── ...
        ├── v_Archery_g01_c01
        │   ├── 1.jpg
        │   └── ...

        clip_len (int): Number of frames per sample, i.e. depth of Model input.
        train (bool): Training vs. Testing model. Default is True
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, class_idxs, split, frames_root, train, clip_len=16, spatial_crop=True):

        self.class_idxs = class_idxs
        self.split_path = split
        self.frames_root = frames_root
        self.train = train
        self.clip_len = clip_len
        self.class_dict = self.read_class_ind()
        self.paths = self.read_split()
        self.data_list = self.build_data_list()
        self.do_crop=spatial_crop
        self.resize_height = 224
        self.resize_width = 224
        self.crop_size = 112


    # Reads .txt file w/ each line formatted as "1 ApplyEyeMakeup" and returns dictionary {'ApplyEyeMakeup': 0, ...}
    def read_class_ind(self):
        class_dict = {}
        with open(self.class_idxs) as f:
            for line in f:
                label, class_name_key = line.strip().split()
                if class_name_key not in class_dict:
                    class_dict[class_name_key] = []
                class_dict[class_name_key] = int(label) - 1  # .append(line.strip())
        print(class_dict)
        return class_dict


    # Reads train or test split.txt file and returns list [('v_ApplyEyeMakeup_g08_c01', array(0)), ...]
    def read_split(self):
        paths = []
        print(self.split_path)

        for pth in self.split_path:
            with open(pth) as f:
                for line in f:
                    class_name, vid_id = line.strip().split()[0][:-4].split('/')
                    vid_dir = os.path.join(self.frames_root, class_name, vid_id)
                    if os.path.exists("%s.avi" % vid_dir):
                        paths.append((vid_dir, class_name))

        return paths


    def build_data_list(self):
        paths = self.paths
        class_dict = self.class_dict
        data_list = []
        for vid_dir, class_name in paths:
            label = np.array(class_dict[class_name], dtype=int)
            data_list.append((vid_dir, label, self.clip_len, class_name))

        return data_list


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        vid_dir, label, frame_count, class_name = self.data_list[index]
        buffer = self.load_frames(vid_dir, frame_count)
        if self.do_crop:
            buffer = self.spatial_crop(buffer, self.crop_size)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        return buffer, label


    def load_frames(self, vid_dir, frame_count):
        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        frames = create_frames(vid_dir, self.clip_len)
        for i, frame in enumerate(frames):
            try:
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            except:
                print('The image %s is potentially corrupt!\nDo you wish to proceed? [y/n]\n' % vid_dir)
                response, _, _ = select.select([sys.stdin], [], [], 15)
                if response == 'n':
                    sys.exit()
                else:
                    frame = np.zeros((buffer.shape[1:]))

            frame = np.array(frame).astype(np.float32)
            buffer[i] = frame

        return buffer


    @staticmethod
    def spatial_crop(buffer, crop_size):
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)
        # Spatial crop is performed on the entire array, so each frame is cropped in the same location.
        buffer = buffer[:, height_index:height_index + crop_size, width_index:width_index + crop_size, :]

        return buffer


    @staticmethod
    def normalize(buffer):
        for i, frame in enumerate(buffer):
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            with np.errstate(all="raise"):
                frame -= np.array([[[90.0, 98.0, 102.0]]])  # BGR means
            buffer[i] = frame

        return buffer


    @staticmethod
    def to_tensor(buffer):
        buffer = buffer.transpose((3, 0, 1, 2))
        return torch.from_numpy(buffer)



if __name__ == '__main__':
    clip_len=16
    batch_size = 200
    class_idxs, train_split, test_split, frames_root, remaining = build_paths()
    dataset = UCF10(class_idxs=class_idxs, split=[train_split[0]], frames_root=frames_root,
                     clip_len=clip_len, train=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1) 
    for i, (batch, labels) in enumerate(dataloader):
        labels = np.array(labels)
        for j, clip in enumerate(batch):
            for img in clip:
                name = 'Class: %s' % str(labels[j] + 1)

