import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import matplotlib.colors as colors

cut=30
dust=5

def correlation_function(k,power_specturm): #波数空間での相関関数
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
    
    
#Video Source
Dir_name="/mnt/d/dendrite_data/edited_data/edited_movie/"
f_name="20230221_nonsur_76.8mN_No.1.avi"
f_name2="20230213_0.05sur_71.4mN_No.4.avi"
f_name3="20230221_nonsur_76.6mN_No.2.avi"

file_path=Dir_name + f_name
delay = 1
window_name = file_path[len(Dir_name):]
cap = cv2.VideoCapture(file_path)
print(window_name)
print(type(cap))

Lx=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Ly=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS=cap.get(cv2.CAP_PROP_FPS)
print(cap.isOpened())
Last_Frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
Total_Frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("File Name : ",window_name)
print("Frame Width : ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Frame Hight : ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS : ", cap.get(cv2.CAP_PROP_FPS))
print("Frame Count : ", cap.get(cv2.CAP_PROP_FRAME_COUNT))

#Last Frame
cap.set(cv2.CAP_PROP_POS_FRAMES,Last_Frame-1)
ret,image=cap.read()

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#輝度値情報のみの画像
threshold,binary=cv2.threshold(gray,cut,255,cv2.THRESH_BINARY)#完全に二値化(RGB -> 輝度値算出 ->二値化)
threshold,nongray_binary=cv2.threshold(image,cut,255,cv2.THRESH_BINARY)#RGBを残したまま二値化
# th_otsu,otsu_binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)　#輝度値が双方性を持たないので微妙

# #適応的閾値 -> ダメ
# thresh_mean=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
# thresh_gaussian=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)

print(binary[240,240])
# remove dust
for i in range(Ly):
    for j in range(Lx):
        if binary[i,j]==255:
            c_temp=0
            for k in range(5):  # 1粒子あたり周囲25マスの探索
                for l in range(5):
                    if (2 <= (j - 2 + l) < Lx - 2) and (2 <= (i - 2 + k) < Ly - 2)and (binary[i - 2 + k, j - 2 + l]==255) :
                        c_temp += 1
            if c_temp <= dust:  # 周囲25マスの粒子数が(dust)個より少なければゴミと判定
                binary[i,j]=0

#モーメントの計算（二値化画像を使用）
#Initial Frame
cap.set(cv2.CAP_PROP_POS_FRAMES,30)
ret,image_tmp=cap.read()
gray_tmp=cv2.cvtColor(image_tmp,cv2.COLOR_BGR2GRAY)
threshold,binary_tmp=cv2.threshold(gray_tmp,cut,255,cv2.THRESH_BINARY)

for i in range(Ly):
    for j in range(Lx):
        if binary_tmp[i,j]==255:
            c_temp = 0
            for k in range(5):  # 1粒子あたり周囲25マスの探索
                for l in range(5):
                    if (2 <= (j - 2 + l) < Lx - 2) and (2 <= (i - 2 + k) < Ly - 2)and (binary_tmp[i - 2 + k, j - 2 + l]==255) :
                        c_temp += 1
            if c_temp <= dust:  # 周囲25マスの粒子数が(dust)個より少なければゴミと判定
                binary_tmp[i,j]=0

m=cv2.moments(binary_tmp,True) #bool値はbinary画像かどうか
#重心の計算、四捨五入
x,y=round(m['m10']/m['m00']) , round(m['m01']/m['m00'])
print(x,y)

# 2 次元高速フーリエ変換で周波数領域の情報を取り出す
f_uv = np.fft.fft2(binary)
# 画像の中心に低周波数の成分がくるように並べかえる
shifted_f_uv = np.fft.fftshift(f_uv)
#print(shifted_f_uv)
# パワースペクトルに変換する
magnitude_spectrum2d = 20 * np.log(np.absolute(shifted_f_uv))

k=[(min(int(Lx/2),int(Ly/2))/100*(i+1)) for i in range(100)]
C_k=[[]for i in range(len(k))]
for i in range(len(k)):
    C_k[i]=correlation_function(k[i],magnitude_spectrum2d)
    
C_k_log=np.log2(C_k)
k_log=np.log2(k)

# 元の並びに直す
unshifted_f_uv = np.fft.fftshift(shifted_f_uv)
# 2 次元逆高速フーリエ変換で空間領域の情報に戻す
i_f_xy = np.fft.ifft2(unshifted_f_uv).real  # 実数部だけ使う

# 元画像
# img_disp = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
img_disp=image

# 元画像の重心位置に x印を書く
cv2.line(img_disp, (x-5,y-5), (x+5,y+5), (255, 0, 0), 2)
cv2.line(img_disp, (x+5,y-5), (x-5,y+5), (255, 0, 0), 2)

# 画像として可視化する
fig, axes = plt.subplots(1, 3, figsize=(18,6))
#fig, axes = plt.subplots(1, 2, figsize=(16,8))
# 枠線と目盛りを消す
for ax in axes.flat:
    ax.set_axis_off()

#元画像(gray)
axes[0].imshow(img_disp,cmap='gray')
axes[0].set_title('Input Image')

# 二値化画像
cv2.line(nongray_binary, (x-5,y-5), (x+5,y+5), (255, 0, 0), 2)
cv2.line(nongray_binary, (x+5,y-5), (x-5,y+5), (255, 0, 0), 2)
axes[1].imshow(nongray_binary,cmap='gray')
axes[1].set_title('nongray_binary')


# 二値化画像
axes[2].imshow(binary,cmap='gray')
axes[2].set_title('binary')

# # Addaptiv threshold(mean)
# axes[1,0].imshow(thresh_mean,cmap='gray')
# axes[1,0].set_title('mean')

# # Addaptiv threshold(gaussian)
# axes[1,1].imshow(thresh_gaussian,cmap='gray')
# axes[1,1].set_title('gaussian')

# # 二値化画像(大津の二値化)
# axes[2,0].imshow(otsu_binary,cmap='gray')
# axes[2,0].set_title('Otsu')

# # 輝度値のヒストグラム
# plt.hist(gray.ravel(),256)



# # 周波数領域のパワースペクトル
# axes[1].imshow(magnitude_spectrum2d,cmap='gray')
# axes[1].set_title('Magnitude Spectrum')
# # FFT -> IFFT した画像
# axes[2].imshow(i_f_xy,cmap='gray')
# axes[2].set_title('Reversed Image')
#correlation function
# axes[1].plot(k_log,C_k_log,"-o")
# axes[1].set_title('Correlation Function')

# グラフを表示する
plt.show()

# if not cap.isOpened():
#     sys.exit()

# while True:
#     ret, frame = cap.read()
#     if ret:
#         cv2.imshow(window_name, frame)
#         if cv2.waitKey(delay) & 0xFF == ord('q'):
#             break
#     else:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# cv2.destroyWindow(window_name)