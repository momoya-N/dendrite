import math
import sys
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.collections as mc
import matplotlib.cm as cm
import time
import copy


def N_Frame_Image(frameIndex: int):  # N番目のフレーム画像を返す
    # インデックスがフレームの範囲内なら…
    if 0 <= frameIndex < Total_Frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        ret, image = cap.read()  # ret:bool値(画像が読めれば True) image:画像のnbarray
    else:
        print("Out of video frames")
        sys.exit(1)

    return image


def CM(n):  # n:n_0,calculate the center of mass
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, image_cm = cap.read()
    gray_cm = cv2.cvtColor(image_cm, cv2.COLOR_BGR2GRAY)
    threshold, binary_cm = cv2.threshold(gray_cm, cut, 255, cv2.THRESH_BINARY)

    binary_cm = Remove_Dust(binary_cm)

    m = cv2.moments(binary_cm, True)  # bool値はbinary画像かどうか
    # 重心の計算、四捨五入
    x, y = round(m["m10"] / m["m00"]), round(m["m01"] / m["m00"])

    return x, y


def First_Frame():  # ピクセル値を持つfirst frameを計算
    n = 0
    while 1:
        # making binary array
        gray = cv2.cvtColor(N_Frame_Image(n), cv2.COLOR_BGR2GRAY)  # このままでも動く。
        threshold, binary = cv2.threshold(gray, cut, 255, cv2.THRESH_BINARY)  # 完全に二値化
        binary = Remove_Dust(binary)

        if np.count_nonzero(binary) > 0:
            break
        n += 1
    return n


def Correlation_Function(k, power_specturm):  # 波数空間での相関関数
    d_theta = 2 * math.asin(0.5 / k)
    theta = 0
    rho_k = 0

    if k >= max(Lx / 2, Ly / 2):
        return

    while theta < 2 * math.pi:
        k_x = k * math.cos(theta)
        k_y = k * math.sin(theta)
        rho_k += power_specturm[int(Ly / 2 + k_y), int(Lx / 2 + k_x)]
        theta += d_theta

    return rho_k / (2 * math.pi)


def Remove_Dust(Matrix):  # remove dust function
    for i in range(2, Ly - 2):
        for j in range(2, Lx - 2):
            if Matrix[i, j] == 255:
                count = search(i, j, 5, Matrix)
                if count <= dust:  # 周囲25マスの粒子数が(dust)個より少なければゴミと判定
                    Matrix[i, j] = 0
    return Matrix


def search(i: int, j: int, k: int, image):  # Search branch, edge and node, k is search range:odd number,image array
    count = 0
    width = int((k - 1) / 2)
    for p in range(i - width, i + width + 1):
        for q in range(j - width, j + width + 1):
            if image[p][q] == 255:
                count += 1

    return count  # 自分も含めた周囲の粒子数


# Main
# time
start = time.time()

# constants
dust = 4  # チリの大きさ判定用変数
cut = 30  # threshold value,輝度値は0が黒色、255が白色。

# Video Source
Video_dir_path = "/mnt/d/dendrite_data/edited_data/edited_movie/"
file_path_list = glob.glob(Video_dir_path + "*.avi")
file_count = 1
Total_file_count = len(file_path_list)

print("Start Analize")
file_count = 1
# for path in file_path_list:
for p in [5]:
    path = file_path_list[p]
    print("Progress:" + str(file_count) + "/" + str(Total_file_count))
    fname = os.path.basename(path)
    file_path = Video_dir_path + fname
    name_tag = fname.replace(".avi", "")
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Video reading Error :" + file_path)
        sys.exit(1)

    Lx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Ly = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    print("Video Source Reading is... :", cap.isOpened())
    Total_Frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("File Name : ", fname)
    print("Frame Width : ", Lx)
    print("Frame Hight : ", Ly)
    print("FPS : ", FPS)
    print("Frame Count : ", Total_Frames)

    n0 = First_Frame()  # origin frame number, time=0

    # Getting Last Frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, Total_Frames - 1)
    ret, img_origin = cap.read()

    color_img = img_origin.copy()
    gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)  # 輝度値情報のみの画像
    threshold, binary = cv2.threshold(gray, cut, 255, cv2.THRESH_BINARY)  # 完全に二値化(RGB -> 輝度値算出 ->二値化)
    threshold, nongray_binary = cv2.threshold(img_origin, cut, 255, cv2.THRESH_BINARY)  # RGBを残したまま二値化

    binary = Remove_Dust(binary)

    # Making thinning image by THINNING_GUOHALL method
    skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    edited_skelton = list(copy.deepcopy(skeleton))
    edited_skelton = [[[0, 0, 0, 0, 0] if y == 255 else [] for y in x] for x in edited_skelton]
    for i in range(1, len(skeleton) - 1):
        for j in range(1, len(skeleton[i]) - 1):
            if skeleton[i][j] != 0:
                count = search(i, j, 3, skeleton)
                if count <= 1:  # dust(大きさ１の孤立ピクセル)の消去処理。万が一のerror処理
                    edited_skelton[i][j] = []
                elif count < 4:
                    edited_skelton[i][j][0] = count  # 2:edge,3:branch flag
                else:
                    edited_skelton[i][j][0] = 4  # 4:node flag

    for i in range(len(edited_skelton)):
        for j in range(len(edited_skelton[i])):
            if edited_skelton[i][j] != [] and edited_skelton[i][j][0] == 2:  # edge
                color_img[i][j] = (0, 0, 255)
            elif edited_skelton[i][j] != [] and edited_skelton[i][j][0] == 3:  # branch
                color_img[i][j] = (0, 255, 0)
            elif edited_skelton[i][j] != [] and edited_skelton[i][j][0] == 4:  # node
                color_img[i][j] = (255, 0, 0)

    print("Start Making Figure")
    # Making Images
    x, y = CM(n0)  # 重心計算
    scalebar = ScaleBar(11 / 681, "cm", length_fraction=0.5, location="lower right")

    # Making figure
    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.imshow(img_origin, cmap="gray")
    ax1.set_title("Thinning GUOHALL method")
    ax1.add_artist(scalebar)
    ax1.imshow(color_img)
    # color_img[skeleton == 255] = (0, 255, 0)

    plt.savefig(str(name_tag) + "_GUOHALL.pdf")

    finish = time.time()
    print("Finish Making Figure")
    total_time = finish - start
    print("total time:", total_time)
    file_count += 1
    print("--------------")

    # skeleton_list=[skeleton1,skeleton2]
    # name_list=["_ZHANGSUEN","_GUOHALL"]

    # for i in range(len(name_list)):
    #     #元画像(gray)
    #     # cv2.line(img_origin, (x-5,y-5), (x+5,y+5), (255, 0, 0), 2)
    #     # cv2.line(img_origin, (x+5,y-5), (x-5,y+5), (255, 0, 0), 2)
    #     ax1.imshow(img_origin,cmap='gray')
    #     ax1.set_title('Thinning_' + name_list[i])
    #     ax1.add_artist(scalebar)
    #     color_img = img_origin.copy()
    #     color_img[skeleton_list[i] == 255] = (0, 255, 0)
    #     ax1.imshow(color_img)
    #     plt.savefig(str(name_tag) + name_list[i] + ".pdf")

    # ax1.imshow(img_origin,cmap='gray')
    # ax1.set_title('Thinning Method')
    # ax1.add_artist(scalebar)
    # method0=skeleton_list[0].copy()
    # method1=skeleton_list[1].copy()
    # common=skeleton_list[0].copy()

    # method0[skeleton_list[1]==255]=0 #0にあって１にないもの
    # method1[skeleton_list[0]==255]=0 #1にあって0にないもの
    # common[method0==255]=0
    # common[method1==255]=0 #片方にしかないものを消した

    # color_img0 = img_origin.copy()

    # color_img0[method0 == 255] = (255, 0, 0) #赤色
    # color_img0[method1 == 255] = (0, 255, 0) #緑色
    # color_img0[common==255] = (0,0,255) #青色、両方にあるもの
    # ax1.imshow(color_img0)
    # plt.savefig(str(name_tag)+ "_test.pdf")
    # #GUOHALL methodの方が端の方がよく取れている->採用#
