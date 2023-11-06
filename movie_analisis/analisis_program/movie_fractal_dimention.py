import math

import cv2
import matplotlib.pylab as pl
import numpy as np


def N_FrameImage(frameIndex):  # N番目のフレーム画像を返す
    # インデックスがフレームの範囲内なら…
    if frameIndex >= 0 and frameIndex < totalFrames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        ret, image = cap.read()
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


# main function
g = "20230223_0.03sur_75.7mN_No.4"  # ここを変えれば読み込みファイルを変えられる。読み込みはedited_movieから。
filename = "/Volumes/HDPH-UT/dendrite_data/edited_data/edited_movie/" + str(g) + ".avi"  # 黒地に白の画像 ファイル名を入れる
cap = cv2.VideoCapture(filename)  # 動画読み込み
totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # get total frame number of movie
dust = 4  # チリの大きさ判定用変数
cut = 30  # threshold value,輝度値は0が黒色、255が白色。思ったより画像が暗いので、白を取るときは割と大きめで。

n0 = n_init()  # origin frame number, time=0
n = n0  # time=0
K = 15  # distance of pick up frame,1フレームあたり2秒なので、30秒ごとに取っている
D = []
step = []

cutoff = (totalFrames - n0) / 2  # 動画の真ん中のフレーム
w = []  # 重みづけ関数

# calculate frontline

r_c = center(n0)
print(r_c)

while cap.isOpened():
    if n >= totalFrames:  # ＝がないと、ここは通るのに、N_FrameImageの条件を満たさず、grayの読み込みでエラーが出る。
        break

    s = 0  # 面積カウント用

    ret, image = cap.read()

    # making binary array
    gray = cv2.cvtColor(N_FrameImage(n), cv2.COLOR_BGR2GRAY)  # RGBの3次元情報を輝度値のみの1次元情報に変換
    binary = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.int8)
    pixels = []
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i, j] > cut:
                c_temp = 0
                for k in range(5):  # 1粒子あたり周囲25マスの探索で単純に時間は25倍になる。しんど。
                    for l in range(5):
                        if (2 <= (j - 2 + l) < gray.shape[1] - 2) and (2 <= (i - 2 + k) < gray.shape[0] - 2) and (gray[i - 2 + k, j - 2 + l] > cut):
                            c_temp += 1

                if c_temp > dust:  # 周囲25マスの粒子数が(dust)個より多ければ粒子と判定
                    pixels.append((i, j))

    Lx = gray.shape[1]
    Ly = gray.shape[0]
    pixels = pl.array(pixels)

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
    "/Volumes/HDPH-UT/dendrite_data/edited_data/housedolf_dim_data/" + str(g) + ".dat",
    "w",
)
f.write("#time" + "\t" + "#Housedolf_dim" + "\n")
for i in range(len(time)):
    f.write(str(time[i]) + "\t" + str(dim[i]) + "\n")
f.close()


# 全体的にもっと効率化&可読性を上げることはできる気がするが、なにぶん初めてなので許して下さいなんでも(以下略)


"""
g = "20230206_0.005sur_76.6mN_No.4"  # ここを変えれば読み込みファイルを変えられる
filename = "/Volumes/HDPH-UT/dendrite_data/edited_data/edited_movie/" + str(g) + "_edited.avi"  # 黒地に白の画像 ファイル名を入れる
cap = cv2.VideoCapture(filename)  # 動画読み込み
totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # get total frame number of movie

k = 100  # number of sample,10の倍数の時だけ、D_fの個数とTの分割の個数が合わない？なぜ？→n=0から始めていた影響説or(k*K)noteq(totalframe),数値誤差の影響か？
K = totalFrames / k  # distance of pick up frame
n = 3 * K  # frame index,初期値はnon0(ある程度の大きさ)の方がいい。画面が単一色だとgray[i,j]を満たすピクセルが存在しなくなり、histogramのbinの次元と合わなくなりエラーが出る。
D = []  # fractral dimention strage
w = []  # weight strage
m = 2  # cut-off index, cut-off under 1/m frame of movie
cutoff = k * K / m  # =(totalframs)/m, cut-off frame

# frame_n = cv2.cvtColor(N_FrameImage(n), cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    if n >= totalFrames:  # ＝がないと、ここは通るのに、N_FrameImageの条件を満たさず、grayの読み込みでエラーが出る。
        break
    ret, image = cap.read()

    gray = cv2.cvtColor(N_FrameImage(n), cv2.COLOR_BGR2GRAY)  # RGBの3次元情報を輝度値のみの1次元情報に変換

    # finding all the zero(black) pixels
    pixels = []
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i, j] > 20:  # 輝度値は0が黒色、255が白色。どっちを探索するかで<>を変える。思ったより画像が暗いので、白を取るときは20以上くらいがいいかも。cv2.thresholdは3次元なのでgrayに再変換すれば==0or255とできる...と思いました。(1敗)
                pixels.append((i, j))

    Lx = gray.shape[1]
    Ly = gray.shape[0]
    pixels = pl.array(pixels)

    # computing the fractal dimension
    # considering only scales in a logarithmic list
    scales = np.logspace(1, 6, num=10, endpoint=False, base=2)  # parametaの組み合わせは考慮の余地あり。2^n~L/(6~8)(ボックス6~8個分)ぐらいまでがいい感じ？分割幅は増やせば精度が上がる時もあるし、ない時もある。ワカラン。ただ、(end)-1だとscaleが整数値になって、なぜか精度が落ちた。Sierpinski gasketで調査。
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

    if n < cutoff:
        w.append(0)
    else:
        w.append(1)


T = np.linspace(0, totalFrames, k - 2)  # 始めのframe=0の時の分も加えて k個のHousdolf demention。k≡0(mod10)だとk+1にしないとダメっぽい？


fig = pl.figure(figsize=(7, 7))
pl.plot(T, D, label="")
pl.xlabel("$t$", fontsize=14)
pl.ylabel("$D$", fontsize=14)
mean_D = np.average(D, weights=w)

pl.title("Mean Hausdorff dimension is " + str(mean_D), fontsize=14)
pl.savefig(str(g) + ".png")
pl.close()

cap.release()
cv2.destroyAllWindows()

"""
