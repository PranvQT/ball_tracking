# Non custom imports
import pandas as pd
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# custom imports
from input.read_videos import video_processor
from process import get_detection,get_3d_point


class ball_tracker():
    def __init__(self):
        pass

    def start_tracker(self):
        imgs = self.video_processing.get_frames()
        detection = []
        for i in range(num_cams):
            detection.append(get_detection(imgs[i]))
        point_3d = get_3d_point(detection)
        ball_tracklet = track(point_3d)

if __name__ == '__main__':
    ball_tracking = ball_tracker()
    ball_tracking.start_tracker()
    