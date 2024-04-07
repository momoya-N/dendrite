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

a = [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0]
b = [i for i in range(len(a))]
c = [a for i in range(5)]
print(c)
test = [[y == 1 for y in x] for x in c]
test2 = [[2 if y == 1 else -1 for y in x] for x in c]
print(a)
print(test)
print(test2)
