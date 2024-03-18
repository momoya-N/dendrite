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
import time
import math
import gc

def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

a=[[[5, 4.079502041625272, 5486, 0, 0, 0], [5, 3.378330093494353, 5487, 0, 0, 0], [5, 2.076153618394076, 5488, 0, 0, 0]], [[4, 3.651300852650212, 5489, 0, 0, 0]], [[3, 3.604016039664557, 5490, 0, 0, 0]], [], [[0, 0, 5492, 0, 0, 0]]]
for i in range(len(a)):
    if a[i]==[]:
        print(a[i])
print(a)
