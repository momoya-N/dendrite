import os
import sys
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.collections as mc
import matplotlib.cm as cm

print(cv2.getBuildInformation())
#Video Source
Dir_path="/mnt/d/dendrite_data/edited_data/edited_movie/"
# Dir_path="/mnt/c/Users/PC/Desktop/"
file_path_list=glob.glob(Dir_path+"*.avi")
file_count=1
Total_file_count=len(file_path_list)

for path in file_path_list:
    print("Progress:"+ str(file_count) + "/" + str(Total_file_count))
    fname=os.path.basename(path)
    file_path=Dir_path + fname
    name_tag=fname.replace(".avi","")
    print(file_path)
    print(name_tag)
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Video Source Reading is... :",cap.isOpened())
        print("Video reading Error")
        sys.exit(1)
    else:
        print("Video Source Reading is... :",cap.isOpened())
        file_count+=1