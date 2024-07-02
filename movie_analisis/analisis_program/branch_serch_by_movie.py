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


def n_frame_image(frame_index: int):  # return n frame image
    # インデックスがフレームの範囲内なら…
    if 0 <= frame_index < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, img = cap.read()  # ret:bool値(画像が読めれば True) img:画像のndarray
    else:
        print("Out of video frames")
        sys.exit(1)

    return img


# def CM(n):  # n:n_0,calculate the center of mass
#     cap.set(cv2.CAP_PROP_POS_FRAMES, n)
#     ret, image_cm = cap.read()
#     gray_cm = cv2.cvtColor(image_cm, cv2.COLOR_BGR2GRAY)
#     threshold, binary_cm = cv2.threshold(gray_cm, cut, 255, cv2.THRESH_BINARY)

#     binary_cm = Remove_Dust(binary_cm)

#     m = cv2.moments(binary_cm, True)  # bool値はbinary画像かどうか
#     # 重心の計算、四捨五入
#     x, y = round(m["m10"] / m["m00"]), round(m["m01"] / m["m00"])

#     return x, y


def first_frame(binary_img):  # get first frame
    n = 0
    while 1:
        binary = Remove_Dust(binary)

        if np.count_nonzero(binary) > 0:
            break
        n += 1
    return n


# def Correlation_Function(k, power_specturm):  # 波数空間での相関関数
#     d_theta = 2 * math.asin(0.5 / k)
#     theta = 0
#     rho_k = 0

#     if k >= max(Lx / 2, Ly / 2):
#         return

#     while theta < 2 * math.pi:
#         k_x = k * math.cos(theta)
#         k_y = k * math.sin(theta)
#         rho_k += power_specturm[int(Ly / 2 + k_y), int(Lx / 2 + k_x)]
#         theta += d_theta

#     return rho_k / (2 * math.pi)

# def remove_dust(binary_img):  # remove dust from binary image
#     serch_width = 5  # search range from center pixel
#     dust = 4  # dust size
#     tmp = int((serch_width - 1) / 2)
#     for i in range(tmp, Ly - tmp):
#         for j in range(tmp, Lx - tmp):
#             if binary_img[i, j] == 255:
#                 count = search_dust(i, j, serch_width, binary_img)
#                 if count <= dust:  # 周囲25マスの粒子数が(dust)個より少なければゴミと判定
#                     binary_img[i, j] = 0
#     return binary_img


def search_dust(i: int, j: int, k: int, image):  # k is search range:odd number,image array
    count = 0
    width = int((k - 1) / 2)
    for p in range(i - width, i + width + 1):
        for q in range(j - width, j + width + 1):
            if image[p][q] == 255:
                count += 1

    return count  # 自分も含めた周囲の粒子数


def mk_gray_binary_img():  # Making Gray And Binary Movie Data
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gray_movie_data = []
    binary_movie_data = []

    # Get Otsu Threshold
    img = n_frame_image(total_frames - 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_ostu, _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    print("Otsu Threshold Value:", thresh_ostu)

    for frame_i in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, thresh_ostu, 255, cv2.THRESH_BINARY)
        # if frame_i > int(total_frames / 10):
        #     binary = remove_dust(binary)

        gray_movie_data.append(gray)
        binary_movie_data.append(binary)

    return gray_movie_data, binary_movie_data


def save_movie_img(img_data, img_path: str):
    for i, img in enumerate(img_data):
        path = img_path + str(i) + ".png"
        cv2.imwrite(img_path, img)


def check_video_source(file_path: str):
    if not cap.isOpened():
        print("Video reading error :" + file_path)
        sys.exit(1)
    else:
        print("Video was readed :" + file_path)

    print("----------------------------")


def video_info(file_path: str):
    fname = os.path.basename(file_path)
    Lx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Ly = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("File Name : ", fname)
    print("Frame Width : ", Lx)
    print("Frame Hight : ", Ly)
    print("FPS : ", FPS)
    print("Frame Count : ", total_frames)
    print("----------------------------")

    return fname, Lx, Ly, FPS, total_frames


# main
unit = 182.5 / 5.0  # 5cm=182.5px
# time
start_time = time.time()

# constants
dust = 4  # チリの大きさ判定用変数
# cut = 30  # threshold value,輝度値は0が黒色、255が白色。

# Video Source
file_path = "/mnt/d/master_thesis_data/experment_data/movie_data/movie_data/data_for_analisis/TWEEN/20240428_2ML_TWEEN20_0.005sur_4.78mPas_41.1mN_0001/20240422_2ML_TWEEN20_0.05sur_4.69mPas_33.2mN_0001_edited.avi"

# Reading Video
cap = cv2.VideoCapture(file_path)

# Check Video Source
check_video_source(file_path)

# Video Information
fname, Lx, Ly, FPS, total_frames = video_info(file_path)

# Start Movie Analize
print("Start Analize")
# making VideoWriter
name_tag = fname.replace(".avi", "")
output_movie = name_tag + "_binary.avi"  # 保存する動画ファイル名
cv2.VideoWriter
# カラー画像をグレースケールに変換して動画として保存するためisColor=Falseとしています。
out = cv2.VideoWriter(output_movie, fourcc, fps, (Lx, Ly), False)


# # Making Gray And Binary Movie Data
# gray_img, binary_img = mk_gray_binary_img()

# # Save Gray And Binary Movie Data
# gray_movie_data_path = "/mnt/c/Users/PC/Desktop/Master_Thesis/movie_analisis/analisis_program/gray_movie_data/"
# binary_movie_data_path = "/mnt/c/Users/PC/Desktop/Master_Thesis/movie_analisis/analisis_program/binary_movie_data/"
# os.makedirs(gray_movie_data_path, exist_ok=True)
# os.makedirs(binary_movie_data_path, exist_ok=True)

# # Start Analize


# n0 = First_Frame()  # origin frame number, time=0
# for path in file_path_list:
# for p in range(len(file_path_list)):
# path = file_path_list[p]
# print("Progress:" + str(file_count) + "/" + str(Total_file_count))
# fname = os.path.basename(path)
# file_path = Video_dir_path + fname
# name_tag = fname.replace(".avi", "")
# cap = cv2.VideoCapture(file_path)

# Lx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# Ly = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# FPS = cap.get(cv2.CAP_PROP_FPS)
# print("Video Source Reading is... :", cap.isOpened())
# Total_Frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# n0 = First_Frame()  # origin frame number, time=0

# # Getting Last Frame
# cap.set(cv2.CAP_PROP_POS_FRAMES, Total_Frames - 1)
# ret, img_origin = cap.read()

# color_img = img_origin.copy()
# gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)  # 輝度値情報のみの画像
# threshold, binary = cv2.threshold(gray, cut, 255, cv2.THRESH_BINARY)  # 完全に二値化(RGB -> 輝度値算出 ->二値化)
# threshold, nongray_binary = cv2.threshold(img_origin, cut, 255, cv2.THRESH_BINARY)  # RGBを残したまま二値化

# binary = Remove_Dust(binary)

# # Making thinning image by THINNING_GUOHALL method
# skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
# edited_skelton = list(copy.deepcopy(skeleton))
# edited_skelton = [[[0, 0, 0, 0, 0] if y == 255 else [] for y in x] for x in edited_skelton]
# for i in range(1, len(skeleton) - 1):
#     for j in range(1, len(skeleton[i]) - 1):
#         if skeleton[i][j] != 0:
#             count = search(i, j, 3, skeleton)
#             if count <= 1:  # dust(大きさ１の孤立ピクセル)の消去処理。万が一のerror処理
#                 edited_skelton[i][j] = []
#             elif count < 4:
#                 edited_skelton[i][j][0] = count  # 2:edge,3:branch flag
#             else:
#                 edited_skelton[i][j][0] = 4  # 4:node flag

# for i in range(len(edited_skelton)):
#     for j in range(len(edited_skelton[i])):
#         if edited_skelton[i][j] != [] and edited_skelton[i][j][0] == 2:  # edge
#             color_img[i][j] = (0, 0, 255)
#         elif edited_skelton[i][j] != [] and edited_skelton[i][j][0] == 3:  # branch
#             color_img[i][j] = (0, 255, 0)
#         elif edited_skelton[i][j] != [] and edited_skelton[i][j][0] == 4:  # node
#             color_img[i][j] = (255, 0, 0)

# print("Start Making Figure")
# # Making Images
# x, y = CM(n0)  # 重心計算
# scalebar = ScaleBar(11 / 681, "cm", length_fraction=0.5, location="lower right")

# # Making figure
# fig = plt.figure(figsize=(9, 9))
# ax1 = fig.add_subplot(1, 1, 1)

# ax1.imshow(img_origin, cmap="gray")
# ax1.set_title("Thinning GUOHALL method")
# ax1.add_artist(scalebar)
# ax1.imshow(color_img)
# # color_img[skeleton == 255] = (0, 255, 0)
# img_path = Data_dir_path + "pdf_img_thinning/"
# os.makedirs(img_path, exist_ok=True)
# plt.savefig(img_path + str(name_tag) + "_GUOHALL.pdf")
# img_path = Data_dir_path + "png_img_thinning/"
# os.makedirs(img_path, exist_ok=True)
# plt.savefig(img_path + str(name_tag) + "_GUOHALL.png")

# finish = time.time()
# print("Finish Making Figure")
# total_time = finish - start
# print("total time:", total_time)
# file_count += 1
# print("--------------")

# # skeleton_list=[skeleton1,skeleton2]
# # name_list=["_ZHANGSUEN","_GUOHALL"]

# # for i in range(len(name_list)):
# #     #元画像(gray)
# #     # cv2.line(img_origin, (x-5,y-5), (x+5,y+5), (255, 0, 0), 2)
# #     # cv2.line(img_origin, (x+5,y-5), (x-5,y+5), (255, 0, 0), 2)
# #     ax1.imshow(img_origin,cmap='gray')
# #     ax1.set_title('Thinning_' + name_list[i])
# #     ax1.add_artist(scalebar)
# #     color_img = img_origin.copy()
# #     color_img[skeleton_list[i] == 255] = (0, 255, 0)
# #     ax1.imshow(color_img)
# #     plt.savefig(str(name_tag) + name_list[i] + ".pdf")

# # ax1.imshow(img_origin,cmap='gray')
# # ax1.set_title('Thinning Method')
# # ax1.add_artist(scalebar)
# # method0=skeleton_list[0].copy()
# # method1=skeleton_list[1].copy()
# # common=skeleton_list[0].copy()

# # method0[skeleton_list[1]==255]=0 #0にあって１にないもの
# # method1[skeleton_list[0]==255]=0 #1にあって0にないもの
# # common[method0==255]=0
# # common[method1==255]=0 #片方にしかないものを消した

# # color_img0 = img_origin.copy()

# # color_img0[method0 == 255] = (255, 0, 0) #赤色
# # color_img0[method1 == 255] = (0, 255, 0) #緑色
# # color_img0[common==255] = (0,0,255) #青色、両方にあるもの
# # ax1.imshow(color_img0)
# plt.savefig(str(name_tag) + "_test.pdf")
# # #GUOHALL methodの方が端の方がよく取れている->採用#
finish_time = time.time()
print("Total time : ", finish_time - start_time)
