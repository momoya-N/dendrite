import numpy as np
import matplotlib.pyplot as plt
import math 
import pandas as pd

a=list(range(10))
print(a)
for i in reversed(range(len(a))):
    print("Iteration:"+str(i))
    for j in reversed(range(0,i)):
        print(a[j])
    print("-------------")
