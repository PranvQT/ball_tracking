# Non custom imports
import cv2
import json
from imutils.video import FileVideoStream as Fvs
import threading
import numpy as np
import queue

# Custom imports 


class video_processor():
    '''
    creating the class for video processor 
    here all the things related to reading video and
    sotring the frames should take place
    '''
    def __init__(self):
        self.frm_count == 0
        with open("Settings/config.json", 'r') as _file:
            self.tracker_config = json.load(_file)

        if self.tracker_config.camera_model == 1:
            self.vid_reading_function = cv2.VideoCapture(self.source)
            self.source = self.tracker_config.source
        elif self.tracker_config.camera_model == 2:
            self.vid_reading_function = Fvs(path=self.source).start()
            if self.process_config.dk_12g_mode == 1:
                self.source = 'decklinkvideosrc mode=0 connection=0 ! videoconvert ! appsink'
            else:
                self.source = f'decklinkvideosrc device-number={self.process_config.dk_vno} profile=5 mode=0 connection=0 ! videoconvert ! appsink'
        
        self.frames_queue = queue.Queue(maxsize=5)

    def start_storing_frames(self):
        '''
        Store Frames in the queue 
        '''

        while True:
            frame = self.get_frame()
            if (frame is not None) and (self.FRAME_HT <= frame.shape[0]) and (self.FRAME_WD <= frame.shape[1]):                
                self.mutex1.acquire()
                if self.frames_queue.full() is True:
                    self.frames_queue.get()
                    self.dropped_frames += 1
                self.frames_queue.put(frame)
                self.mutex1.release()
            
            if self.stop_stream:
                self.vid_reading_function.stop()
                break
            # count1 += 1
        self.stop_stream = False

    def get_frame(self):
        '''
        get frames from either stored video or live video
        '''
        
        if self.tracker_config.camera_model == 1:
            ret, frame = self.vid_reading_function.read()
            ret, frame = self.vid_reading_function.read()
            if frame is None:
                print("CHECK VIDEO INPUT")
        elif self.tracker_config.camera_model == 2:
            frame = self.vid_reading_function.read()
            if frame is None:
                print("CHECK VIDEO INPUT")
            
        return frame
