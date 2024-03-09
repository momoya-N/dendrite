import numpy as np
import matplotlib.pyplot as plt
import math 
import pandas as pd
import copy

a=[i for i in range(5)]
print(a)
for i in range(len(a)):
    for j in range(i,min(i+3,len(a))):
        print(a[j])
    print("---------")