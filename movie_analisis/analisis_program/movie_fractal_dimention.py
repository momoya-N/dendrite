import math
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

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



#Main
#constants
dust = 4  # チリの大きさ判定用変数
cut = 30  # threshold value,輝度値は0が黒色、255が白色。
K = 15  # distance of pick up frame,2 s/frame ->30 s毎に取得

#Video Source
Dir_name="/mnt/c/Users/PC/Desktop"
f_name="/20230205_nonsur_77.2mN_No.1.avi"
f_name2="/20230222_0.05sur_73.2mN_No.3.avi"
f_name3="/20230221_nonsur_76.8mN_No.1.avi"
#Dir_name="/mnt/d/dendrite_data/edited_data/edited_movie/"
#f_name="20230221_nonsur_76.8mN_No.1.avi"
# f_name2="20230213_0.05sur_71.4mN_No.4.avi"
#f_name3="20230221_nonsur_76.6mN_No.2.avi"

file_path=Dir_name + f_name2
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

# 2 次元高速フーリエ変換で周波数領域の情報を取り出す
f_uv = np.fft.fft2(binary)
# 画像の中心に低周波数の成分がくるように並べかえる
shifted_f_uv = np.fft.fftshift(f_uv)
# パワースペクトルに変換する
magnitude_spectrum2d = 20 * np.log(np.absolute(shifted_f_uv))

k=[(min(int(Lx/2),int(Ly/2))/100*(i+1)) for i in range(100)]
C_k=[[]for i in range(len(k))]
for i in range(len(k)):
    C_k[i]=Correlation_Function(k[i],magnitude_spectrum2d)
    
C_k_log=np.log2(C_k)
k_log=np.log2(k)

# 元の並びに直す
unshifted_f_uv = np.fft.fftshift(shifted_f_uv)

# 2 次元逆高速フーリエ変換で空間領域の情報に戻す
i_f_xy = np.fft.ifft2(unshifted_f_uv).real  # 実数部だけ使う

# 画像
x,y=CM(n0) #重心計算
img_origin=image #original image
img_n=N_Frame_Image(n0) #N frames image

# 画像として可視化する
fig, axes = plt.subplots(1, 3, figsize=(18,6))

# 枠線と目盛りを消す
for ax in axes.flat:
    ax.set_axis_off()

#元画像(gray)
cv2.line(img_origin, (x-5,y-5), (x+5,y+5), (255, 0, 0), 2)
cv2.line(img_origin, (x+5,y-5), (x-5,y+5), (255, 0, 0), 2)
axes[0].imshow(img_origin,cmap='gray')
axes[0].set_title('Input Image')

# 二値化画像
cv2.line(nongray_binary, (x-5,y-5), (x+5,y+5), (255, 0, 0), 2)
cv2.line(nongray_binary, (x+5,y-5), (x-5,y+5), (255, 0, 0), 2)
axes[1].imshow(nongray_binary,cmap='gray')
axes[1].set_title('nongray_binary')

# 二値化画像
cv2.line(img_n, (x-5,y-5), (x+5,y+5), (255, 0, 0),1)#MatLikeに関数は入れられなさそう
cv2.line(img_n, (x+5,y-5), (x-5,y+5), (255, 0, 0),1)
axes[2].imshow(img_n,cmap='gray')
axes[2].set_title('binary')

plt.show()

# D = []
# step = []

# cutoff = (Total_Frames - n0) / 2  # 動画の真ん中のフレーム
# w = []  # 重みづけ関数

# # calculate frontline

# r_c = center(n0)
# print(r_c)

# while cap.isOpened():
#     if n >= Total_Frames:  # ＝がないと、ここは通るのに、N_FrameImageの条件を満たさず、grayの読み込みでエラーが出る。
#         break

#     s = 0  # 面積カウント用

#     ret, image = cap.read()

#     # making binary array
#     gray = cv2.cvtColor(N_Frame_Image(n), cv2.COLOR_BGR2GRAY)  # RGBの3次元情報を輝度値のみの1次元情報に変換
#     binary = np.zeros((Ly, Lx), dtype=np.int8)
#     for i in range(Ly):
#         for j in range(Lx):
#             if gray[i, j] > cut:
#                 c_temp = 0
#                 for k in range(5):  # 1粒子あたり周囲25マスの探索で単純に時間は25倍になる
#                     for l in range(5):
#                         if (2 <= (j - 2 + l) < gray.shape[1] - 2) and (2 <= (i - 2 + k) < gray.shape[0] - 2) and (gray[i - 2 + k, j - 2 + l] > cut):
#                             c_temp += 1

#                 if c_temp > dust:  # 周囲25マスの粒子数が(dust)個より多ければ粒子と判定
#                     binary[i,j]=1

#     # computing the fractal dimension
#     # considering only scales in a logarithmic list
#     scales = np.logspace(
#         1, 6, num=10, endpoint=False, base=2
#     )  # parametaの組み合わせは考慮の余地あり。2^n~L/(6~8)(ボックス6~8個分)ぐらいまでがいい感じ？分割幅は増やせば精度が上がる時もあるし、ない時もある。ワカラン。ただ、(end)-1だとscaleが整数値になって、なぜか精度が落ちた。Sierpinski gasketで調査。
#     Ns = []

#     # looping over several scales
#     for scale in scales:
#         H, edges = np.histogramdd(pixels, bins=(np.arange(0, Lx, scale), np.arange(0, Ly, scale)))
#         Ns.append(np.sum(H > 0))

#     # linear fit, polynomial of degree 1
#     coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
#     d_f = -coeffs[0]  # Haussdorff dimension
#     n += K
#     D.append(d_f)
#     step.append(n - n0)

#     if n < cutoff:
#         w.append(0)
#     else:
#         w.append(1)


# time = pl.array(step)
# dim = pl.array(D)
# mean_D = np.average(D, weights=w)
# print(mean_D)

# f = open(
#     "C:/Users/PC/Desktop/Master_Thesis/test/fractal_check.dat","w")
# f.write("#scale" + "\t" + "#Box_num" + "\n")
# for i in range(len(scales)):
#     f.write(str(np.log(scales[i])) + "\t" + str(np.log(Ns[i])) + "\n")
# f.close()

# f = open(
#     "/Volumes/HDPH-UT/dendrite_data/edited_data/housedolf_dim_data/" + str(g) + ".dat",
#     "w",
# )
# f.write("#time" + "\t" + "#Housedolf_dim" + "\n")
# for i in range(len(time)):
#     f.write(str(time[i]) + "\t" + str(dim[i]) + "\n")
# f.close()



