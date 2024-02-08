import math
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.collections as mc
import matplotlib.cm as cm
import time

def N_Frame_Image(frameIndex):  # N番目のフレーム画像を返す
    # インデックスがフレームの範囲内なら…
    if 0<= frameIndex < Total_Frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        ret, image = cap.read() #ret:bool値(画像が読めれば True) image:画像のnbarray
        return image
    else:
        return None

def CM(n): #n:n_0,calculate the center of mass
    cap.set(cv2.CAP_PROP_POS_FRAMES,n)
    ret,image_cm=cap.read()
    gray_cm=cv2.cvtColor(image_cm,cv2.COLOR_BGR2GRAY)
    threshold,binary_cm=cv2.threshold(gray_cm,cut,255,cv2.THRESH_BINARY)

    binary_cm=Remove_Dust(binary_cm)
    
    m=cv2.moments(binary_cm,True) #bool値はbinary画像かどうか
    #重心の計算、四捨五入
    x,y=round(m['m10']/m['m00']) , round(m['m01']/m['m00'])
    
    return x,y

def First_Frame():  # ピクセル値を持つfirst frameを計算
    n = 0
    while 1:
        # making binary array
        gray = cv2.cvtColor(N_Frame_Image(n), cv2.COLOR_BGR2GRAY)
        threshold,binary=cv2.threshold(gray,cut,255,cv2.THRESH_BINARY)#完全に二値化
        binary=Remove_Dust(binary)
        
        if np.count_nonzero(binary) > 0:
            break
        n += 1
    return n

def Correlation_Function(k,power_specturm): #波数空間での相関関数
    d_theta=2*math.asin(0.5/k)
    theta=0
    rho_k=0
    
    if k>=max(Lx/2,Ly/2):
        return 

    while theta<2*math.pi:
        k_x=k*math.cos(theta)
        k_y=k*math.sin(theta)
        rho_k += power_specturm[int(Ly/2+k_y),int(Lx/2+k_x)]
        theta += d_theta
        
    return rho_k/(2*math.pi)

def Remove_Dust(Matrix):#remove dust funvtion
    for i in range(Ly):
        for j in range(Lx):
            if Matrix[i,j]==255:
                c_temp = 0
                for k in range(5):  # 1粒子あたり周囲25マスの探索
                    for l in range(5):
                        if (2 <= (j - 2 + l) < Lx - 2) and (2 <= (i - 2 + k) < Ly - 2)and (Matrix[i - 2 + k, j - 2 + l]==255) :
                            c_temp += 1
                if c_temp <= dust:  # 周囲25マスの粒子数が(dust)個より少なければゴミと判定
                    Matrix[i,j]=0
    return Matrix

def search_branch(binary,r,x_c,y_c):  # binsry data,radius r, cm_x,cm_y
    dth = 1 / r  # delta theta,十分大きいrではokそう？→f(x)=2arcsin(1/2x)-1/xは、x=2で0.005360...
    t = 0  # 角度のステップ数
    k = 0  # 角度のステップ数その2
    th = 0  # 角度
    phi = 0  # 太さ探索用角度
    # thick_d_array = []
    branch_cm=[]
    branch_th=[]

    r_x = int(r * math.cos(0) + x_c)
    r_y = int(r * math.sin(0) + y_c)

    while th < 2 * math.pi:
        th = dth * t
        r_x = int(r * math.cos(th) + x_c)
        r_y = int(r * math.sin(th) + y_c)

        cm_tmp=[]
        th_tmp=[]

        if (1 <= r_x < Lx-1) and (1 <= r_y < Ly-1):
            while binary[r_y, r_x] == 255:#枝の重心計測
                cm_tmp.append([r_x,r_y]) #cv2の描画の関係上ここだけx,yの順番が違う
                th_tmp.append(th)
                t += 1
                th = dth * t
                
                if th > 2 * math.pi:
                    break
                
                r_x = int(r * math.cos(th) + x_c)  # dth分回転させる
                r_y = int(r * math.sin(th) + y_c)

        if np.size(cm_tmp)!=0:
            cm=np.average(cm_tmp,axis=0)
            th=np.average(th_tmp)
            branch_cm.append(cm)
            branch_th.append(th)

        t += 1

    return branch_cm,branch_th

#Main

#time
start=time.time()

#constants
dust = 4  # チリの大きさ判定用変数
cut = 30  # threshold value,輝度値は0が黒色、255が白色。
# K = 15  # distance of pick up frame,2 s/frame ->30 s毎に取得

#Video Source
Dir_name="/mnt/c/Users/PC/Desktop/"
f_name="20230205_nonsur_77.2mN_No.1.avi"
f_name2="20230222_0.05sur_73.2mN_No.3.avi"
f_name3="20230221_nonsur_76.8mN_No.1.avi"

file_path=Dir_name + f_name3
name_tag=file_path.replace(Dir_name,"")
name_tag=name_tag.replace(".avi","")
window_name = file_path[len(Dir_name):]
cap = cv2.VideoCapture(file_path)
print(window_name)
print(type(cap))

Lx=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Ly=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS=cap.get(cv2.CAP_PROP_FPS)
print(cap.isOpened())
Total_Frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("File Name : ",window_name)
print("Frame Width : ", Lx)
print("Frame Hight : ", Ly)
print("FPS : ", FPS)
print("Frame Count : ", Total_Frames)

n0 = First_Frame()  # origin frame number, time=0

#Last Frame
cap.set(cv2.CAP_PROP_POS_FRAMES,Total_Frames-1)
ret,image=cap.read()

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#輝度値情報のみの画像
threshold,binary=cv2.threshold(gray,cut,255,cv2.THRESH_BINARY)#完全に二値化(RGB -> 輝度値算出 ->二値化)
threshold,nongray_binary=cv2.threshold(image,cut,255,cv2.THRESH_BINARY)#RGBを残したまま二値化

binary=Remove_Dust(binary)

# 画像
x,y=CM(n0) #重心計算

img_origin=image #original image
img_n=N_Frame_Image(n0) #N frames image
scalebar=ScaleBar(11/681,"cm",length_fraction=0.5,location="lower right")

# 画像として可視化する
r_max=max(x,(Lx-x),y,(Ly-y))#最大半径＝重心からの距離の最大値
fig = plt.figure(figsize=(9,9))
# ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(1,1,1,projection="polar")
# ax3=fig.add_subplot(2,2,4)
## fig, ax = plt.subplots(1,3,figsize=(18,6))

# #元画像(gray)
cv2.line(img_origin, (x-5,y-5), (x+5,y+5), (255, 0, 0), 2)
cv2.line(img_origin, (x+5,y-5), (x-5,y+5), (255, 0, 0), 2)
# ax1.imshow(img_origin,cmap='gray')
# ax1.set_title('Input Image')
# ax1.add_artist(scalebar)

#theta-半径グラフ
ax2.set_axis_off()
##ax2=plt.subplot(132,projection="polar")

ax2.set_axis_on()
ax2.set_xticks(
    [0, np.pi/4, np.pi/2, np.pi*3/4, np.pi, np.pi*5/4, np.pi*3/2, np.pi*7/4], 
    ["0", "\u03c0/4", "\u03c0/2", "3\u03c0/4", "\u03c0", "5\u03c0/4", "3\u03c0/2", "7\u03c0/4"]
)
# ax3.set_xlim(-0.1,2*math.pi+0.1)
# ax3.set_xlabel(r"$\theta$")
# ax3.set_ylabel(r"Radius $r$ pix")
search_range=[r for r in range(2,r_max,int(r_max/20))]
theta=[[]for i in range(len(search_range))]

for i , r in enumerate(search_range):
    branch_cm,branch_th=search_branch(binary,r,x,y)
    for j in range(len(branch_th)):
        # ax3.scatter(2*math.pi-branch_th[j],r,s=1,c="k")
        ax2.scatter(2*math.pi-branch_th[j],r,s=1,c="k")
        theta[i].append(2*math.pi-branch_th[j]) #探索の向きの関係上2Piから引くとimput imageと向きが一致

# ax3.set_box_aspect(0.8)

#枝のベクトル計算
vector=[]
for i in reversed(range(1,len(search_range))):
    rnow=search_range[i]
    rnext=search_range[i-1]
    for j in range(len(theta[i])):
        thnow=theta[i][j]
        dist_tmp2=pow(r_max,2) #十分大きな数で
        for k in range(len(theta[i-1])):
            thnext=theta[i-1][k]
            dist2=pow(rnow,2)+pow(rnext,2)-2*rnow*rnext*math.cos(thnow-thnext)
            if dist_tmp2 > dist2:
                dist_tmp2=dist2
                tmp1=rnext
                tmp2=thnext
        vector.append([[thnow,rnow],[tmp2,tmp1]])

print(vector)
node=[]
vector_pare=[]
# search node
for i in range(len(vector)-1):
    if vector[i][1][0] == vector[i+1][1][0]:
        node.append(vector[i][1])
        # vector_pare.append()

for i in range(len(node)):
    ax2.scatter(node[i][0],node[i][1],s=10,c="r")

lc1 = mc.LineCollection(vector, colors="k", linewidths=1)
# lc2 = mc.LineCollection(vector, colors="k", linewidths=1)

##r,theta,対応する枝の番号の分の情報をつけて再帰的に探索
ax2.add_collection(lc1)
# ax3.add_collection(lc2)

# plt.savefig(str(name_tag)+".png")
# plt.savefig(str(name_tag)+"_lareg.png")
finish=time.time()
plt.show()
total_time=finish-start
print("total time:",total_time)
print("x:",x,",(Lx-x):",(Lx-x),",y:",y,",(Ly-y):",(Ly-y))

#時間計測メモ file3で計測
# 系サイズ最大、1ステップ毎：75.58797311782837 sec ->20倍になっても時間は10倍ほど
# 系サイズ最大、r_max/20毎：7.566540241241455 sec
# 重心から系の端まで、1ステップ毎：77.58949756622314 sec ->こっちの方がむしろ長い？
# 重心から系の端まで、1ステップ毎、探索関数にif文をかませない：83.54880547523499 sec ->ifをかませない方が長い？
# 重心から系の端まで、r_max/20毎：6.782961368560791 sec
# 点数が増えるとplot関数により時間がかかる印象,cpuの使用状況に多少よるかも