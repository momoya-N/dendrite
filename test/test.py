import numpy as np
import matplotlib.pyplot as plt
import math 
import pandas as pd
import copy

r0=241
r1=232
r2=245
th0=3.990652542207994
th1=3.981459387415673
th2=3.9729796219037206

r01_2=pow(r0,2)+pow(r1,2)-2*r0*r1*math.cos(th0-th1)
r12_2=pow(r1,2)+pow(r2,2)-2*r1*r2*math.cos(th1-th2)
r20_2=pow(r2,2)+pow(r0,2)-2*r2*r0*math.cos(th2-th0)

print((th0-th1)/math.pi,(th1-th2)/math.pi,(th2-th0)/math.pi)
print((-r12_2+r01_2+r20_2))
print(2*math.sqrt(r01_2*r20_2))
print(math.acos((-r12_2+r01_2+r20_2)/(2*math.sqrt(r01_2*r20_2)))/math.pi)