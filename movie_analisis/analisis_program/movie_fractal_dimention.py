import math
import sys

import cv2
import matplotlib.pylab as pl
import numpy as np

def N_FrameImage(frameIndex):  # N番目のフレーム画像を返す
    # インデックスがフレームの範囲内なら…
    if frameIndex >= 0 and frameIndex < Total_Frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        ret, image = cap.read() #ret:bool値(画像が読めれば True) image:画像のnbarray
        return image
    else:
        return None


def center(n):  # 重心を返す,n=n0=n_initを入れる
    center = []

    for k in range(30):  # time=0からxフレーム分計算,x=30なのは経験と勘による。
        gray = cv2.cvtColor(N_FrameImage(n + k), cv2.COLOR_BGR2GRAY)
        # finding all the zero(black) pixels
        pixels = []
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                if gray[i, j] > cut:
                    pixels.append((i, j))
        center.append(pl.mean(pixels, 0))  # 重心(平均値)の中央値なので、外れ値(チリ)の影響はあまりない...はず。

    center = np.array(center, dtype=int)

    return np.median(center, axis=0)


def n_init():  # ピクセル値を持つfirst frameを計算
    n = 0
    while 1:
        # making binary array
        gray = cv2.cvtColor(N_FrameImage(n), cv2.COLOR_BGR2GRAY)
        binary = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.int8)

        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                if gray[i, j] > cut:
                    c_temp = 0
                    for k in range(5):
                        for l in range(5):
                            if (2 <= (j - 2 + l) < gray.shape[1] - 2) and (2 <= (i - 2 + k) < gray.shape[0] - 2) and (gray[i - 2 + k, j - 2 + l] > cut):
                                c_temp += 1

                    if c_temp <= dust:  # 周囲25マスの粒子数が(dust)個以下ならチリと判定。初めて値を持つフレームをn0=n_initにすると、他の計算でエラーが出る。(初期フレームは析出粒子が小さくチリとして判定され、結果的にまっさらな画像の判定になり、空配列の判定を行うことになってエラーが出る。)
                        binary[i, j] = 0
                    else:
                        binary[i, j] = 1

        if np.count_nonzero(binary == 1) > 0:
            break

        n += 1

    return n

#Main
#Video Source
Dir_name="/mnt/d/dendrite_data/edited_data/edited_movie/"
f_name="20230221_nonsur_76.8mN_No.1.avi"
delay = 1
window_name = f_name
cap = cv2.VideoCapture(Dir_name + f_name)

Lx=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
Ly=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
FPS=cap.get(cv2.CAP_PROP_FPS)
Total_Frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(type(cap)) # <class 'cv2.VideoCapture'>
print(cap.isOpened()) #True or False
print("Frame Width : ", Lx)
print("Frame Hight : ", Ly)
print("FPS : ", FPS)
print("Frame Count : ",Total_Frames) # get total frame number of movie

'''
if not cap.isOpened():
    sys.exit()

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cv2.destroyWindow(window_name)

'''

dust = 4  # チリの大きさ判定用変数
cut = 30  # threshold value,輝度値は0が黒色、255が白色。思ったより画像が暗いので、白を取るときは割と大きめで。

n0 = n_init()  # origin frame number, time=0
n = n0  # time=0
K = 15  # distance of pick up frame,1フレームあたり2秒なので、30秒ごとに取っている
D = []
step = []

cutoff = (Total_Frames - n0) / 2  # 動画の真ん中のフレーム
w = []  # 重みづけ関数

# calculate frontline

r_c = center(n0)
print(r_c)

while cap.isOpened():
    if n >= Total_Frames:  # ＝がないと、ここは通るのに、N_FrameImageの条件を満たさず、grayの読み込みでエラーが出る。
        break

    s = 0  # 面積カウント用

    ret, image = cap.read()

    # making binary array
    gray = cv2.cvtColor(N_FrameImage(n), cv2.COLOR_BGR2GRAY)  # RGBの3次元情報を輝度値のみの1次元情報に変換
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

    # computing the fractal dimension
    # considering only scales in a logarithmic list
    scales = np.logspace(
        1, 6, num=10, endpoint=False, base=2
    )  # parametaの組み合わせは考慮の余地あり。2^n~L/(6~8)(ボックス6~8個分)ぐらいまでがいい感じ？分割幅は増やせば精度が上がる時もあるし、ない時もある。ワカラン。ただ、(end)-1だとscaleが整数値になって、なぜか精度が落ちた。Sierpinski gasketで調査。
    Ns = []

    # looping over several scales
    for scale in scales:
        H, edges = np.histogramdd(pixels, bins=(np.arange(0, Lx, scale), np.arange(0, Ly, scale)))
        Ns.append(np.sum(H > 0))

    # linear fit, polynomial of degree 1
    coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
    d_f = -coeffs[0]  # Haussdorff dimension
    n += K
    D.append(d_f)
    step.append(n - n0)

    if n < cutoff:
        w.append(0)
    else:
        w.append(1)


time = pl.array(step)
dim = pl.array(D)
mean_D = np.average(D, weights=w)
print(mean_D)

f = open(
    "C:/Users/PC/Desktop/Master_Thesis/test/fractal_check.dat","w")
f.write("#scale" + "\t" + "#Box_num" + "\n")
for i in range(len(scales)):
    f.write(str(np.log(scales[i])) + "\t" + str(np.log(Ns[i])) + "\n")
f.close()

# f = open(
#     "/Volumes/HDPH-UT/dendrite_data/edited_data/housedolf_dim_data/" + str(g) + ".dat",
#     "w",
# )
# f.write("#time" + "\t" + "#Housedolf_dim" + "\n")
# for i in range(len(time)):
#     f.write(str(time[i]) + "\t" + str(dim[i]) + "\n")
# f.close()



