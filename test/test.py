import numpy as np
import matplotlib.pyplot as plt
import math 
import pandas as pd

def branch_angle(node:list,branch1:list,branch2:list): #各点の極座標(theta,r)をわたす
    r0=node[1]
    r1=branch1[1]
    r2=branch2[1]
    th0=node[0]
    th1=branch1[0]
    th2=branch2[0]

    r01_2=pow(r0,2)+pow(r1,2)-2*r0*r1*math.cos(th0-th1)
    r12_2=pow(r1,2)+pow(r2,2)-2*r1*r2*math.cos(th1-th2)
    r20_2=pow(r2,2)+pow(r0,2)-2*r2*r0*math.cos(th2-th0)

    angle=math.acos((-r12_2+r01_2+r20_2)/(2*math.sqrt(r01_2*r20_2)))

    return angle

node=[0,0]
a=[0,6]
b=[math.pi*1.2,2]
angle=branch_angle(node,b,a)/math.pi
#ああああああ

print(angle)