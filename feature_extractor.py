import csv
import os
import sys
import cv2
import subprocess
import re
import numpy
# import tensorflow as tf
import math
import numpy as np

import h5py

# In OpenCV3.X, this is available as cv2.CAP_PROP_POS_MSEC
# In OpenCV2.X, this is available as cv2.cv.CV_CAP_PROP_POS_MSEC
CAP_PROP_POS_MSEC = 0


class Extractor:
    def __init__(self,video_file, segment_len, frame_rate):
        self.video_path = video_file
        self.segment_len = segment_len
        self.frame_rate = frame_rate

    def __calc_frames(self):
        process = subprocess.Popen(['ffmpeg', '-i', self.video_path], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        stdout, stderr = process.communicate()
        
        #testing comment 
        matches = re.search(r"Duration:\s{1}(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),",stdout.decode('utf-8'), re.DOTALL).groupdict()
        fps = float(re.findall(r"\d*\.?\d* fps,", stdout.decode('utf-8'))[0].split(' ')[0].rstrip("'").lstrip("'"))

        video_len = ((int(matches['hours']) * 3600) + (int(matches['minutes']) * 60) + float(matches['seconds']))
        total_frames = int(video_len * self.frame_rate)
        return total_frames

    def __calc_segments(self, total_frames):
        # total_frames = self.__calc_frames()
        segments = math.ceil(total_frames / (self.frame_rate * self.segment_len))
        return segments

    def __frame_iterator(self, max_prev_frames, every_ms=1000, max_num_frames=360):
        """Uses OpenCV to iterate over all frames of filename at a given frequency.

        Args:
          filename: Path to video file (e.g. mp4)
          every_ms: The duration (in milliseconds) to pick between frames.
          max_num_frames: Maximum number of frames to process, taken from the
            beginning of the video.

        Yields:
          RGB frame with shape (image height, image width, channels)
        """
        video_capture = cv2.VideoCapture()
        if not video_capture.open(self.video_path):
            print(sys.stderr, 'Error: Cannot open video file ' + self.video_path)
            return
        last_ts = -99999  # The timestamp of last retrieved frame.
        num_retrieved = 0

        while num_retrieved <= max_num_frames:
            # Skip frames
            while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
                if not video_capture.read()[0]:
                    return

            last_ts = video_capture.get(CAP_PROP_POS_MSEC)
            has_frames, frame = video_capture.read()
            if not has_frames:
                break
            if num_retrieved >= max_prev_frames:
                yield frame
                # num_retrieved += 1
            num_retrieved += 1

    def __quantize(self, features, min_quantized_value=-2.0, max_quantized_value=2.0):
        """Quantizes float32 `features` into string."""
        assert features.dtype == 'float32'
        assert len(features.shape) == 1  # 1-D array
        features = numpy.clip(features, min_quantized_value, max_quantized_value)
        quantize_range = max_quantized_value - min_quantized_value
        features = (features - min_quantized_value) * (255.0 / quantize_range)
        features = [int(round(f)) for f in features]

        return features

    def __extract_features(self):
        total_error = 0
        frames = self.__calc_frames()
        seg = self.__calc_segments(frames)
        seg_features = {}

        for iter in range(seg):
            rgb_features = []
            sum_rgb_features = None
            prev_max_frames= iter * self.segment_len
            if frames > prev_max_frames:
                for rgb in self.__frame_iterator(every_ms=1000.0 / self.frame_rate, max_prev_frames=prev_max_frames, max_num_frames=prev_max_frames + self.segment_len):
                    features = self.extractor.extract_rgb_frame_features(rgb[:, :, ::-1])
                    if sum_rgb_features is None:
                        sum_rgb_features = features
                    else:
                        sum_rgb_features += features
                    rgb_features.append(self.__quantize(features))

                    if not rgb_features:
                        print >> sys.stderr, 'Could not get features for ' + self.video_path
                        total_error +=1
                        continue
                mean_rgb_features = sum_rgb_features / len(rgb_features)
                seg_features['seg'+ str(iter +1)] = mean_rgb_features

        return seg_features


    def write_h5(self, file_name):
        features = self.__extract_features()
        print("*******feature shape ********", features['seg1'].shape)
        dt = h5py.special_dtype(vlen=str)
        h5f = h5py.File("/home/khawar/Documents/AutoEncoder/Samran_Code/"+file_name, 'w')
        h5f.create_dataset('id', data=self.video_path, dtype=dt)
        h5f.create_dataset('mean_rgb', data=np.array(list(features.values()), dtype=float))
        h5f.create_dataset('seg_num', data=len(features), dtype=int)
        h5f.close()
        print("features written")

# same function as above but it also returns a list of frames for each segment 
    def write_h5_and_return_frames(self, file_name):
        video_frame , features = self.segment_video_extract_features() # note this function returns extracted featuers plus video frames
        print("*******feature shape ********", features['seg1'].shape)
        dt = h5py.special_dtype(vlen=str)
        h5f = h5py.File("/home/khawar/Documents/AutoEncoder/Samran_Code/"+file_name, 'w')
        h5f.create_dataset('id', data=self.video_path, dtype=dt)
        h5f.create_dataset('mean_rgb', data=np.array(list(features.values()), dtype=float))
        h5f.create_dataset('seg_num', data=len(features), dtype=int)
        h5f.close()
        print("features written")
        return video_frame

    def load_vid_data(self, width, height, normalize = True):
        total_error = 0
        frames = self.__calc_frames()
        seg = self.__calc_segments(frames)
        seg_features = {}
        dim = (width, height)
        i = 0 
        for iter in range(seg):
            rgb_frames = []
            sum_rgb_features = None
            prev_max_frames= iter * self.segment_len
            if frames > prev_max_frames:
                for rgb in self.__frame_iterator(every_ms=1000.0 / self.frame_rate, max_prev_frames=prev_max_frames, max_num_frames=prev_max_frames + self.segment_len):
                    # resizing images
                    rgb = cv2.resize(rgb, dim, interpolation = cv2.INTER_AREA)
                    
                    if normalize:
                        # normalizing the frames
                        rgb = rgb/255.0
                    imgs = np.array(rgb)
                    rgb_frames.append(imgs)
            
            if i == 0:
                rgb_img = np.array(rgb_frames)
            else:
                img = np.array(rgb_frames)
                rgb_img = np.vstack((rgb_img,img))

            i+=1

        return rgb_img


    def load_segmentated_video(self):
        total_error = 0
        frames = self.__calc_frames()
        seg = self.__calc_segments(frames)
        seg_features = {}
       
        total_segs_rgb = []
        for iter in range(seg):
            rgb_features = []
            rgb_frames = []
            sum_rgb_features = None
            prev_max_frames= iter * self.segment_len
            if frames > prev_max_frames:
                for rgb in self.__frame_iterator(every_ms=1000.0 / self.frame_rate, max_prev_frames=prev_max_frames, max_num_frames=prev_max_frames + self.segment_len):
                    features = self.extractor.extract_rgb_frame_features(rgb[:, :, ::-1])
                    rgb_frames.append(rgb)

                    if sum_rgb_features is None:
                        sum_rgb_features = features
                    else:
                        sum_rgb_features += features
                    rgb_features.append(self.__quantize(features))

                    if not rgb_features:
                        print >> sys.stderr, 'Could not get features for ' + self.video_path
                        total_error +=1
                        continue
                mean_rgb_features = sum_rgb_features / len(rgb_features)
                seg_features['seg'+ str(iter +1)] = mean_rgb_features
                rgb_img= np.array(rgb_frames)
                total_segs_rgb.append(rgb_img) 
            

        print("segments +++++ ", seg)
        print("total_seg_rgb +++++ ",len(total_segs_rgb))

        return total_segs_rgb

    def segment_video_extract_features(self):
        total_error = 0
        frames = self.__calc_frames()
        seg = self.__calc_segments(frames)
        seg_features = {}
       
        total_segs_rgb = []
        for iter in range(seg):
            rgb_features = []
            rgb_frames = []
            sum_rgb_features = None
            prev_max_frames= iter * self.segment_len
            if frames > prev_max_frames:
                for rgb in self.__frame_iterator(every_ms=1000.0 / self.frame_rate, max_prev_frames=prev_max_frames, max_num_frames=prev_max_frames + self.segment_len):
                    features = self.extractor.extract_rgb_frame_features(rgb[:, :, ::-1])
                    rgb_frames.append(rgb)

                    if sum_rgb_features is None:
                        sum_rgb_features = features
                    else:
                        sum_rgb_features += features
                    rgb_features.append(self.__quantize(features))

                    if not rgb_features:
                        print >> sys.stderr, 'Could not get features for ' + self.video_path
                        total_error +=1
                        continue
                mean_rgb_features = sum_rgb_features / len(rgb_features)
                seg_features['seg'+ str(iter +1)] = mean_rgb_features
                rgb_img= np.array(rgb_frames)
                total_segs_rgb.append(rgb_img) 
            

        print("total_seg_rgb +++++ ",len(total_segs_rgb))
        print("testing ")
        print("length of seg_features", len(seg_features))
        print("segments +++++ ", seg)


        return total_segs_rgb , seg_features;


    def get_frames(self):
        total_error = 0
        frames = self.__calc_frames()
        seg = self.__calc_segments(frames)
        seg_features = {}
       
        total_segs_rgb = []
        for iter in range(seg):
            rgb_features = []
            rgb_frames = []
            sum_rgb_features = None
            prev_max_frames= iter * self.segment_len
            dim = (224, 224)
            if frames > prev_max_frames:
                for rgb in self.__frame_iterator(every_ms=1000.0 / self.frame_rate, max_prev_frames=prev_max_frames, max_num_frames=prev_max_frames + self.segment_len):
                    # resizing images to 224 X 224
                    rgb = cv2.resize(rgb, dim, interpolation = cv2.INTER_AREA)
                    
                    # normalizing the frames
                    rgb = rgb/255.0
                    rgb_frames.append(rgb)

            
                rgb_img= np.array(rgb_frames)
                total_segs_rgb.append(rgb_img) 
            
        print("total_seg_rgb +++++ ",len(total_segs_rgb))
        print("testing ")
        return np.array(total_segs_rgb)

    