import math
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.collections as mc
import matplotlib.cm as cm

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
#constants
dust = 4  # チリの大きさ判定用変数
cut = 30  # threshold value,輝度値は0が黒色、255が白色。
# K = 15  # distance of pick up frame,2 s/frame ->30 s毎に取得

#Video Source
Dir_name="/mnt/c/Users/PC/Desktop/"
f_name="20230205_nonsur_77.2mN_No.1.avi"
f_name2="20230222_0.05sur_73.2mN_No.3.avi"
f_name3="20230221_nonsur_76.8mN_No.1.avi"
#Dir_name="/mnt/d/dendrite_data/edited_data/edited_movie/"
#f_name="20230221_nonsur_76.8mN_No.1.avi"
# f_name2="20230213_0.05sur_71.4mN_No.4.avi"
#f_name3="20230221_nonsur_76.6mN_No.2.avi"

file_path=Dir_name + f_name2
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

# # 2 次元高速フーリエ変換で周波数領域の情報を取り出す
# f_uv = np.fft.fft2(binary)
# # 画像の中心に低周波数の成分がくるように並べかえる
# shifted_f_uv = np.fft.fftshift(f_uv)
# # パワースペクトルに変換する
# magnitude_spectrum2d = 20 * np.log(np.absolute(shifted_f_uv))

# k=[(min(int(Lx/2),int(Ly/2))/100*(i+1)) for i in range(100)]
# C_k=[[]for i in range(len(k))]
# for i in range(len(k)):
#     C_k[i]=Correlation_Function(k[i],magnitude_spectrum2d)
    
# C_k_log=np.log2(C_k)
# k_log=np.log2(k)

# # 元の並びに直す
# unshifted_f_uv = np.fft.fftshift(shifted_f_uv)

# # 2 次元逆高速フーリエ変換で空間領域の情報に戻す
# i_f_xy = np.fft.ifft2(unshifted_f_uv).real  # 実数部だけ使う

# 画像
x,y=CM(n0) #重心計算

img_origin=image #original image
img_n=N_Frame_Image(n0) #N frames image
scalebar=ScaleBar(11/681,"cm",length_fraction=0.5,location="lower right")

# 画像として可視化する
r_max=min(x,(Lx-x),y,(Ly-y))#最大半径＝重心からの距離の最小値
fig, ax = plt.subplots(1,3,figsize=(18,6))

#元画像(gray)
cv2.line(img_origin, (x-5,y-5), (x+5,y+5), (255, 0, 0), 2)
cv2.line(img_origin, (x+5,y-5), (x-5,y+5), (255, 0, 0), 2)
ax[0].imshow(img_origin,cmap='gray')
ax[0].set_title('Input Image')
ax[0].add_artist(scalebar)

#theta-半径グラフ
ax[2].set_axis_on()
ax[2].set_xticks(
    [0, np.pi/4, np.pi/2, np.pi*3/4, np.pi, np.pi*5/4, np.pi*3/2, np.pi*7/4, np.pi*2], 
    ["0", "\u03c0/4", "\u03c0/2", "3\u03c0/4", "\u03c0", "5\u03c0/4", "3\u03c0/2", "7\u03c0/4", "2\u03c0"]
)
ax[2].set_xlim(-0.1,2*math.pi+0.1)
ax[2].set_xlabel(r"$\theta$")
ax[2].set_ylabel(r"Radius $r$ pix")
search_range=[r for r in range(2,r_max)]
theta=[[]for i in range(len(search_range))]

for i , r in enumerate(search_range):
    branch_cm,branch_th=search_branch(binary,r,x,y)
    for j in range(len(branch_th)):
        ax[2].scatter(2*math.pi-branch_th[j],r,s=1,c="black")
        theta[i].append(2*math.pi-branch_th[j]) #探索の向きの関係上2Piから引くとimput imageと向きが一致

ax[2].set_box_aspect(0.5)

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
            if dist_tmp2>dist2:
                dist_tmp2=dist2
                tmp1=rnext
                tmp2=thnext
        vector.append([[thnow,rnow],[tmp2,tmp1]])

lc = mc.LineCollection(vector, colors="k", linewidths=1)
ax[2].add_collection(lc)

ax[1].set_axis_off()
ax[1]=plt.subplot(132,projection="polar")
for i , r in enumerate(search_range):
    branch_cm,branch_th=search_branch(binary,r,x,y)
    for j in range(len(branch_th)):
        ax[1].plot(2*math.pi-branch_th[j],r)
lc = mc.LineCollection(vector, colors="k", linewidths=1)
ax[1].add_collection(lc)

plt.savefig(str(name_tag)+".png")
plt.show()
