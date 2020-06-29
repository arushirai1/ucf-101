import os, sys
import numpy as np
import cv2
import errno
import h5py

def build_paths(print_paths=True):
    root= 'ActionRecognitionSplits'
    class_idxs = os.path.join(root, 'Splits/classInd.txt')
    train_split = [os.path.join(root, 'Splits/trainlist0%d.txt' % i) for i in range(1,4)] #change back to 4
    test_split = [os.path.join(root, 'Splits/testlist0%d.txt' % i) for i in range(1,4)]
    remainder_split = [os.path.join(root, 'Splits/remaining_files.txt')]
    frames_root = os.path.join(root, 'Frames')

    if print_paths:
        print('Class Index Path: %s' % class_idxs,
              '\nTrain Split Path: %s' % str(train_split),
              '\nTest Split Path: %s' % str(test_split),
              '\nFrames Root Dir: %s' % frames_root)
              #'\nPretrained Path: %s' % pretrained_model_path)

    return class_idxs, train_split, test_split, frames_root, remainder_split

import random
def rand(count, start, end, spacing):
    l = []
    if(end//spacing < count):
        return None
    for i in range(count):
        while True:
            num = random.randrange(start, end, spacing)
            if num not in l:
                l.append(num)
                break
    return l

#continuous
def create_frames(video_file, desired_frames=64, skip_rate=2, continuous_frames=8):
    if not os.path.exists(video_file+'.avi'):
        print("DNE", video_file)
        return None
    cap = cv2.VideoCapture("%s.avi" %(video_file))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #i = random.randrange(0, length, desired_frames*skip_rate)
    end = length - desired_frames*skip_rate - 1
    if end <= 0:
        i=0
        skip_rate=1
    else:
        i = random.randint(0, end)
    frames = []
    for j in range(0,desired_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i+skip_rate*j)
        success, frame = cap.read()
        if success and (frame is not None and frame.size != 0):
            frame=np.nan_to_num(frame, 0, posinf=255, neginf=0)
            low_mask = frame < 0
            high_mask = frame > 255
            frame[low_mask]=0
            frame[high_mask]=255
            frames.append(frame)
        else:
            if len(frames) is not 0:
                frames.append(frames[-1])
    while len(frames) != desired_frames:
        frames.append(frames[-1])
        print("video file err", video_file)
    frames=np.array(frames).astype(np.float32)
    return frames
'''
def create_frames(video_file, desired_frames=64, continuous_frames=8):
    if not os.path.exists(video_file+'.avi'):
        print("DNE", video_file)
        return None
    times = desired_frames//continuous_frames
    cap = cv2.VideoCapture("%s.avi" %(video_file))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_starts = rand(times,0,length-continuous_frames,continuous_frames)
    frames = []
    print(clip_starts)
    #step=max(length//desired_frames,2)
    for i in clip_starts: #np.linspace(0,length, num=desired_frames, dtype=int):
        for j in range(0, continuous_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i+j)
            success, frame = cap.read()
            if (frame is not None and frame.size != 0):
                #frame = cv2.patchNaNs(frame, 0)
                frames.append(frame)
    frames=np.array(frames).astype(np.float32)
    return frames
'''
'''
def create_frames(video_file, desired_frames=64):
    if not os.path.exists(video_file+'.avi'):
        print("DNE", video_file)
        return None
    cap = cv2.VideoCapture("%s.avi" %(video_file))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    #step=max(length//desired_frames,2)
    for i in np.linspace(0,length, num=desired_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if (frame is not None and frame.size != 0):
            frames.append(frame)
    frames=np.array(frames).astype(np.float32)
    return frames
'''
def h5_dump(dataset_name, data, h5_file):
    with h5py.File(h5_file, 'a') as hf:
        hf.create_dataset(dataset_name, data=data)
        print("Successfully created: ", dataset_name)

def h5_load(dataset_name, h5_file):
    frames=None
    with h5py.File(h5_file, 'r') as hf:
        frames=hf[dataset_name][:]
    return frames

# Computes Mean and Std Dev, across RGB channels, of all training images in a Dataset & returns averages
# Set Pytorch Transforms to None for this function
def calc_mean_and_std(dataset):
    mean = np.zeros((3,1))
    std = np.zeros((3,1))
    print('==> Computing mean and std...')
    for img in dataset:
        scaled_img = np.array(img[0])/255
        mean_tmp, std_tmp = cv2.meanStdDev(scaled_img)
        mean += mean_tmp
        std += std_tmp
    mean = mean/len(dataset)
    std = std/len(dataset)

    return mean, std


def cv2_imshow(img_path, wait_key=0, window_name='Test'):
    img = cv2.imread(img_path)
    cv2.imshow(window_name, img)
    cv2.waitKey(wait_key)
    cv2.destroyAllWindows()
