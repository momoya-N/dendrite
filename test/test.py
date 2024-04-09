import random
import sys
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.collections as mc
import matplotlib.cm as cm
import time
import math
import gc

constellation = [
    "牡羊座",
    "金牛座",
    "双子座",
    "巨蟹座",
    "狮子座",
    "处女座",
    "天秤座",
    "天蝎座",
    "射手座",
    "摩羯座",
    "水瓶座",
    "双鱼座",
]
list = ["1位", "2位", "3位", "4位", "5位", "6位", "7位", "8位", "9位", "10位", "11位", "12位"]
random.shuffle(constellation)
print("星座占い")
for i in range(12):
    print(list[i] + ":" + constellation[i])
