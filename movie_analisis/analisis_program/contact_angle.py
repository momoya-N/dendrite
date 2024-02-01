import numpy as np
import math
from matplotlib import pyplot as plt
import cv2
from PIL import Image

# Bernstein多項式を計算する関数
def bernstein(n, t):
    B = []
    for k in range(n + 1):
        # 二項係数を計算してからBernstein多項式を計算
        nCk = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
        B.append(nCk * t ** k * (1 - t) ** (n - k))
    return B

# ベジェ曲線を描く関数
def bezie_curve(Q):
    n = len(Q) - 1
    dt = 0.01
    t = np.arange(0, 1 + dt, dt)
    B = bernstein(n, t)
    px = 0
    py = 0
    for i in range(len(Q)):
        px += np.dot(B[i], Q[i][0])
        py += np.dot(B[i], Q[i][1])
    return px, py

def color_hist(filename):
    img = np.asarray(Image.open(filename).convert("L")).reshape(-1,1)
    plt.hist(img, bins=128)
    plt.show()

#Video Source
Dir_name="/mnt/c/Users/PC/Desktop/data_for_contact_angle/"
fname_list=[]
for i in range(4):
    fname="contact_angle_H2O_Plronic_TWEEN_00"+str(i+1)+".tif"
    print(fname)
    fname_list.append(fname)

file_path=Dir_name + fname_list[0]
name_tag=file_path.replace(Dir_name,"")
name_tag=name_tag.replace(".tif","")
window_name = file_path[len(Dir_name):]
img=cv2.imread(file_path,1)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

print(window_name)
print(type(img))

hight, width, channels=img.shape
print("File Name : ",window_name)
print("Frame Width : ", width)
print("Frame Hight : ", hight)

# # 大津の二値化
ret,th = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# スレッショルドを切る
# ret,th = cv2.threshold(img_gray,40,255,cv2.THRESH_BINARY)


# カーネルの設定
kernel = np.ones((5,5),np.uint8)

# モルフォロジー変換（膨張）
th_dilation = cv2.dilate(th,kernel,iterations = 1)

# モルフォロジー変換（収縮）
th_elosion = cv2.erode(th,kernel,iterations = 1)

#オープニング処理(収縮->膨張),極板の除去
th_opening=cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel)

#クロージング処理(膨張->収縮),黒い穴の除去
kernel=np.ones((20,20),np.uint8) #カーネルの変更
th_closing=cv2.morphologyEx(th_opening,cv2.MORPH_CLOSE,kernel)

# 輪郭抽出
contours, hierarchy = cv2.findContours(th_dilation,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

# 輪郭を元画像に描画
img_contour = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)


# color_hist(file_path)

# 点座標を準備
# q1 = [0., 0.]
# q2 = [0.5, 1.]
# q3 = [1., 0.]and
# Q = [q1, q2, q3]

q1 = [0., 0.]
q2 = [0.5, 0.]
q3 = [0.5, 1.]
q4 = [1., 1.]
Q = [q1, q2, q3, q4]

# ベジェ曲線を描く関数を実行
px, py = bezie_curve(Q)
print(px,py)
# ここからグラフ描画-------------------------------------
fig, ax = plt.subplots(1,3,figsize=(18,6))

#元画像(gray)
ax[0].imshow(img_contour,cmap='gray')
ax[0].set_title('Input Image')

#二値化画像(binary)
ax[1].imshow(th_closing,cmap='gray')
ax[1].set_title('Binary Image')

# # フォントのサイズを設定する。
# plt.rcParams['font.size'] = 14

# # 目盛を内側にする。
# ax[2].rcParams['xtick.direction'] = 'in'
# ax[2].rcParams['ytick.direction'] = 'in'

# グラフの上下左右に目盛線を付ける。
# fig = plt.figure()
# ax[2] = fig.add_subplot(133)
ax[2].yaxis.set_ticks_position('both')
ax[2].xaxis.set_ticks_position('both')

# 軸のラベルを設定する。
ax[2].set_xlabel('x')
ax[2].set_ylabel('y')

# スケールの設定をする。
ax[2].set_xlim(-0.1, 1.1)
ax[2].set_ylim(-0.1, 1.1)

# ベジェ曲線をプロット
ax[2].plot(px, py, color='red', label='Bezie curve')

# 制御点をプロット
qx = []
qy = []
for i in range(len(Q)):
    qx.append(Q[i][0])
    qy.append(Q[i][1])
ax[2].plot(qx, qy, color='blue', marker='o', linestyle='--', label='Control point')
ax[2].legend()
#ax1.axis('off')

# レイアウト設定
fig.tight_layout()

# グラフを表示する。
plt.show()
plt.close()
# ---------------------------------------------------
