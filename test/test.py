import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import matplotlib.colors as colors

cut=30
dust=4

def C_k(k,power_specturm): #波数空間での相関関数
    d_theta=2*math.asin(0.5/k)
    theta=0
    rho_k=0
    print(type(power_specturm))
    
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

delay = 1
window_name = f_name
cap = cv2.VideoCapture(Dir_name + f_name)
cap2 = cv2.VideoCapture(Dir_name + f_name2)

print(type(cap))
# <class 'cv2.VideoCapture'>

Lx=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Ly=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS=cap.get(cv2.CAP_PROP_FPS)
print(cap.isOpened())
Last_Frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
Last_Frame2=int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
Total_Frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
Total_Frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
print("Frame Width : ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Frame Hight : ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS : ", cap.get(cv2.CAP_PROP_FPS))
print("Frame Count : ", cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.set(cv2.CAP_PROP_POS_FRAMES,Last_Frame-1)
ret,image=cap.read()
#print(ret)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#print(gray,len(gray),len(gray[1]))

cap2.set(cv2.CAP_PROP_POS_FRAMES,Last_Frame2-1)
ret,image=cap2.read()
#print(ret)

gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#print(gray2,len(gray2),len(gray2[1]))

# plt.subplot(121).imshow(gray)
# plt.subplot(122).imshow(gray2)


# plt.show()

# making binary array
#gray = cv2.cvtColor(Total_Frames, cv2.COLOR_BGR2GRAY)  # RGBの3次元情報を輝度値のみの1次元情報に変換
binary = np.zeros((Ly, Lx), dtype=np.int8)
for i in range(Ly):
    for j in range(Lx):
        if gray[i, j] > cut:
            c_temp = 0
            for k in range(5):  # 1粒子あたり周囲25マスの探索で単純に時間は25倍になる
                for l in range(5):
                    if (2 <= (j - 2 + l) < gray.shape[1] - 2) and (2 <= (i - 2 + k) < gray.shape[0] - 2) and (gray[i - 2 + k, j - 2 + l] > cut):
                        c_temp += 1

            if c_temp > dust:  # 周囲25マスの粒子数が(dust)個より多ければ粒子と判定
                binary[i,j]=1

#モーメントの計算（輝度値情報を持つ画像を使用）
m=cv2.moments(gray,False)
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

# 元の並びに直す
unshifted_f_uv = np.fft.fftshift(shifted_f_uv)
# 2 次元逆高速フーリエ変換で空間領域の情報に戻す
i_f_xy = np.fft.ifft2(unshifted_f_uv).real  # 実数部だけ使う


# 上記を画像として可視化する
fig, axes = plt.subplots(1, 3, figsize=(16,8))
# 枠線と目盛りを消す
for ax in axes:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
# 元画像
img_disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# 元画像の重心位置に x印を書く
cv2.line(img_disp, (x-5,y-5), (x+5,y+5), (255, 0, 0), 3)
cv2.line(img_disp, (x+5,y-5), (x-5,y+5), (255, 0, 0), 3)

#元画像
axes[0].imshow(img_disp,cmap='gray')
axes[0].set_title('Input Image')
# 周波数領域のパワースペクトル
axes[1].imshow(magnitude_spectrum2d,cmap='gray')
axes[1].set_title('Magnitude Spectrum')
# FFT -> IFFT した画像
axes[2].imshow(i_f_xy,cmap='gray')
axes[2].set_title('Reversed Image')
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