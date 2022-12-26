#package imports 
import json
import numpy as np
import cv2
import time

#custom imports
from detection import detect as detector
from read_videos import video_processor

with open("Settings/config.json", 'r') as _file:
    tracker_config = json.load(_file)

def get_bbox(frame):
    img = frame
    if img is None:
        return None
    if tracker_config.seg_mask is not None:
        while len(tracker_config.seg_mask.shape) != 3:
            pass
        if img.shape[0] == tracker_config.seg_mask.shape[0] and img.shape[1] == tracker_config.seg_mask.shape[1]:
            img = np.bitwise_and(
                img, tracker_config.seg_mask)
        else:
            tracker_config.seg_mask = cv2.resize(
                tracker_config.seg_mask, (frame[0], frame[1]))
            img = np.bitwise_and(
                img, tracker_config.seg_mask)
    bbox = detector(img)
    return bbox


def main():
    video_input =  video_processor()
    video_input.start_storing_frames()
    bbox = get_bbox(video_input.qin.get())

if __name__ == '__main__':
    main()