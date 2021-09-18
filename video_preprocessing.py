import feature_extractor as ext
import os
import numpy as np
import subprocess
import multiprocessing
from datetime import datetime
import cv2
import ffmpy3
from ffmpy3 import FFmpeg
import math
import h5py
from tqdm import tqdm


class video_processing():
    def __init__(self):
        super().__init__()
    
    def extract(self, path, filename): 
       video_path = path
       e1 = ext.Extractor(video_path, 60, 1)
       try:
           e1.write_h5(filename)
           print("features extracted")
       except:
           print("error encountered")
    
    def load_videos_from_dir(self, path, width = 240, height=240, frame_rate=2, segment=32,normalize = True):
        """ This function is used to convert an entire directory of videos 
        into numpy array at a fixed fps and fixed size bags 

        Args: 
            path (str) : path to the directory containing videos 
            width (int) : width of the extracted frame(Default = 240)
            height (int) : height of the extracted frame(Default = 240)
            frame_rate (int) : this is the extracted fps (Deafault = 2)
            segment (int) : The maximum capacity of a bag/ batch, note a batch/bag can be less then the max capacity but not greater then it (Default = 32)
            normalize (Bool) : will convert the numpy frames into 0 to 1 range.

        Returns: 
             returns a 4 dimension numpy array which contains multiple bags/batches hold multiple video frames    

        """
        # load all the videos in a directory 
        vid_data = []

        segment = segment - 1 
        video_path = os.path.join(path, os.listdir(path)[0])
        e1 = ext.Extractor(video_path, segment, frame_rate)
        data = e1.load_vid_data(width,height,normalize)
        for i in tqdm (range (1,len(os.listdir(path)))):
            video_path = os.path.join(path, os.listdir(path)[i])
            # print("\nloading video number : ", i )
            # print("path = ", video_path)
            e1 = ext.Extractor(video_path, segment, frame_rate)
            data2 = e1.load_vid_data(width,height,normalize)
            # print("shape of Data : ",data2.shape)
            data = np.concatenate((data,data2), axis=0)
        return data

    def load_video( self, path, width=240,height=240, frame_rate = 2, segment=32, normalize=True):
        """ This function is used to convert an videos 
        into numpy array at a fixed fps and fixed size bags 

        Args: 
            path (str) : path to the directory containing videos 
            width (int) : width of the extracted frame(Default = 240)
            height (int) : height of the extracted frame(Default = 240)
            frame_rate (int) : this is the extracted fps (Deafault = 2)
            segment (int) : The maximum capacity of a bag/ batch, note a batch/bag can be less then the max capacity but not greater then it (Default = 32)
            normalize (Bool) : will convert the numpy frames into 0 to 1 range.

        Returns: 
             returns a 5 dimension numpy array which contains multiple bags/batches hold multiple video frames    

        """
        segment = segment-1
        e1 = ext.Extractor(path, segment, frame_rate)
        data = e1.load_vid_data(width,height,normalize)
        return data 

    def segment_videos(self, path , frame_rate):
        e1 = ext.Extractor(path, 60, 1)
        
        # this returns a list containing each segment from the video 
        # each segment has a frames (the frames are numpy array) so list[0] contains frames for 1st segment , list[1] contains frames for second segment and so on 
        data = e1.load_segmentated_video()

        # saving the segmentated frames as a video in the static folder
        dir_path = "/home/khawar/Documents/AutoEncoder/Samran_Code/api/static/"

        for i in range(len(data)):
            filename = "output" +str(i) +".webm"
            path = dir_path + filename
            out = cv2.VideoWriter(path , cv2.VideoWriter_fourcc(*"vp80"), frame_rate, (320, 240))
            for frame in data[i]:
                out.write(frame) # frame is a numpy.ndarray with shape (240, 320, 3)
            out.release()

        print(" video saved ! ")

        return len(data)

    def testing_function(self , path ):
        e1 = ext.Extractor(path, 60, 1)
        
        # this returns a list containing each segment from the video 
        # each segment has a frames (the frames are numpy array) so list[0] contains frames for 1st segment , list[1] contains frames for second segment and so on 
        

        return e1.segment_video_extract_features()
    
    
    def load_seg_and_classify(self, path):
        e1 = ext.Extractor(path, 60, 1)
        # this returns a list containing each segment from the video 
        # each segment has a frames (the frames are numpy array) so list[0] contains frames for 1st segment , list[1] contains frames for second segment and so on 
        data = e1.load_segmentated_video()
        dir_path = "/home/khawar/Documents/AutoEncoder/Samran_Code/api/static/"
        for i in range(len(data)):
            filename = "output" +str(i) +".webm"
            path = dir_path + filename
            out = cv2.VideoWriter(path , cv2.VideoWriter_fourcc(*"vp80"), frame_rate=1, dim = (320, 240))
            for frame in data[i]:
                out.write(frame) # frame is a numpy.ndarray with shape (240, 320, 3)
            out.release()

    def extract_feat_and_save_video(self , path , filename, frame_rate=2,segment = 32):

       video_path = path
       e1 = ext.Extractor(video_path, segment, frame_rate)
        # saving the segmentated frames as a video in the static folder
       dir_path = "/home/khawar/Documents/AutoEncoder/Samran_Code/api/static/"
    
       try:
           data = e1.write_h5_and_return_frames(filename)
           print("features extracted")
       except:
           print("error encountered")
           
       for i in range(len(data)):
           filename = "output" +str(i) +".webm"
           path = dir_path + filename
           out = cv2.VideoWriter(path , cv2.VideoWriter_fourcc(*"vp80"), frame_rate, (320, 240))
           for frame in data[i]:
               out.write(frame) # frame is a numpy.ndarray with shape (240, 320, 3)
           out.release()

       print(" video saved ! ")

       return len(data)

    def writefeatures(self , path , filename):
        video_path = path
        e1 = ext.Extractor(video_path, 60, 1)
        # saving the segmentated frames as a video in the static folder
        dir_path = "/home/khawar/Documents/AutoEncoder/Samran_Code/api/static/"
        try:
            data = e1.write_h5_and_return_frames(filename)
            print("features extracted")
        except:
            print("error encountered")

    def save_video(self , path, filename, frame_rate):
        video_path = path
        e1 = ext.Extractor(video_path, 60, 1)
        data = e1.write_h5_and_return_frames(filename)
        for i in range(len(data)):
            filename = "output" +str(i) +".webm"
            path = dir_path + filename
            out = cv2.VideoWriter(path , cv2.VideoWriter_fourcc(*"vp80"), frame_rate, (320, 240))
            for frame in data[i]:
                out.write(frame) # frame is a numpy.ndarray with shape (240, 320, 3)
            out.release()

        print(" video saved ! ")

        return len(data) 
    
    # the following function returns the duration of video in seconds 
    def get_length(self , path):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
        return float(result.stdout) 

    # The following function breaks the video into equal chuncks of a fixed timestep and returns a list of filenames
    def preprocess_video(self , path , sec, filename , target_dir):
        
        duration = self.get_length(path)
        # calculating video segments
        seg  = duration/sec
        seg = math.ceil(seg)
        print("number of segments : ", seg)
        names = []
        # creating file name and deleting video files if they already exist so no interupt is generated for video re-write
        for i in range(seg):
            file_name =  filename[:-4]
            print("checking file ",file_name)
            file_name= file_name +"-"+str(i+1)+ "-of-" +str(seg)+".mp4"
            deleting_files_path = "/".join([target_dir,file_name])

            if os.path.exists(deleting_files_path):
                os.remove(deleting_files_path)
            
            names.append(file_name)
        
        chunks = subprocess.run(["python","ffmpeg-split.py", "-f", path , "-s", str(sec)])
        
        return names
