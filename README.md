# Videolytics
This is Open Source Project I started because I could not find code for extracting video frames at fixed fps and segment that did not have some sort of memory issue, hence I
turned towards making this library. 

## Converting video frames into numpy array( for a single video ) 
 ``` python 
 from video_preprocessing import video_processing

 video_converter = video_processing()

 filename = "full path to your video"

 Converted_frames = video_converter.load_video(filename,frame_rate = 30, segment=60, normalization = False)


```


## Converting video frames into numpy array( for an entire directory of video ) 
 ``` python 
 from video_preprocessing import video_processing

 video_converter = video_processing()

 filename = "full path to your directory"

 Converted_frames = video_converter.load_video_from_dir(filename,frame_rate = 30, segment=60, normalization = False)


```

## Further features:
More features are being considered and will be added shortly. Moreover, I will also create a pip package soon. 

## Contribution deatils:
All sorts of contribiutions are more then welcome, I will add a contribution guide soon so stay tunned. 
