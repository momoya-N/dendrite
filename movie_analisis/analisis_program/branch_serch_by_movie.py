import math
import sys
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.collections as mc

# import matplotlib.cm as cm
import time
import copy
from tqdm import tqdm
import pandas as pd
import re


def n_frame_image(frame_index: int):  # return n frame image
    # インデックスがフレームの範囲内なら…
    if 0 <= frame_index < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # フレームの位置変更，全体で開始フレームが変わるので注意
        ret, img = cap.read()  # ret:bool値(画像が読めれば True) img:画像のndarray
    else:
        print("Out of video frames")
        sys.exit(1)

    return img


def center():
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_otsu, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # initialization
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while 1:
        ret, img = cap.read()
        img = remove_dust(img, 4, thresh_otsu)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, thresh_otsu, 255, cv2.THRESH_BINARY)
        count = np.count_nonzero(binary)
        if count > 4:
            m = cv2.moments(binary, True)
            x, y = round(m["m10"] / m["m00"]), round(m["m01"] / m["m00"])
            break

    return x, y


def r_g(frame: int, thresh_otsu: float, x: int, y: int):
    tmp_rg = 0
    img = n_frame_image(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_otsu, binary = cv2.threshold(gray, thresh_otsu, 255, cv2.THRESH_BINARY)
    nonzero = np.nonzero(binary)
    for i in range(len(nonzero[0])):
        tmp_rg += (nonzero[0][i] - y) ** 2 + (nonzero[1][i] - x) ** 2
    tmp_rg = tmp_rg / len(nonzero[0])
    r_g = np.sqrt(tmp_rg)

    return r_g


def check_video_source(file_path_avi: str):
    if not cap.isOpened():
        print("Video reading error :" + file_path_avi)
        sys.exit(1)
    elif cap.isOpened():
        print("Video was readed :" + file_path_avi)

    print("----------------------------")


def video_info(file_path_avi: str):
    fname = os.path.basename(file_path_avi)
    Lx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Ly = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) / 10
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sec_per_frame = 10
    frame_step = int(4 * total_frames / (10 * unit))  # 4pix成長するのに必要なフレーム数

    print("File Name : ", fname)
    print("Frame Width : ", Lx)
    print("Frame Hight : ", Ly)
    print("FPS : ", fps)
    print("Frame Count : ", total_frames)
    print("Total time : ", total_frames * sec_per_frame)
    print("Frame Step : ", frame_step)
    print("----------------------------")

    return fname, Lx, Ly, fps, total_frames, total_frames * sec_per_frame, frame_step


def remove_dust(frame, dust_area: int, thresh_otsu: float):
    # チリを除去する関数
    # チリの面積がdust_area以下の輪郭を除去する
    # frame:入力画像
    # dust_area:チリとみなす面積の閾値
    # 出力:チリを除去したgray画像(値としては二値化)
    black = np.zeros_like(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh_otsu, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour_i in range(len(contours)):
        if cv2.contourArea(contours[contour_i]) > dust_area:
            cv2.fillPoly(black, [contours[contour_i]], (255, 255, 255))
    return black


def search(img, i: int, j: int, k: int):  # Search branch, edge and node, k is search range:odd number,input image array
    count = 0
    width = int((k - 1) / 2)
    for p in range(i - width, i + width + 1):
        for q in range(j - width, j + width + 1):
            if img[p][q] == 255:
                count += 1
    return count  # 自分中心に周囲ｋ×ｋの白画素数を返す


# def mk_tracking_movie(start_frame: int, end_frame: int, frame_step: int, dust: int):
#     # get Otsu threshold value
#     frame = n_frame_image(total_frames - 1)
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     thresh_otsu, _ = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_OTSU)
#     # # get initital frame
#     # frame_befor = remove_dust(n_frame_image(start_frame), dust, thresh_otsu)

#     for frame_i in tqdm(range(start_frame + frame_step, end_frame - frame_step, frame_step), desc="Tracking"):
#         r_mean_tmp = []
#         x_list = []
#         y_list = []
#         frame_befor = remove_dust(n_frame_image(frame_i), dust, thresh_otsu)
#         frame_after = remove_dust(n_frame_image(frame_i + frame_step), dust, thresh_otsu)
#         frame_diff = cv2.subtract(frame_after, frame_befor)
#         frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
#         _, frame_diff = cv2.threshold(frame_diff, thresh_otsu, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(frame_after, contours, -1, (0, 255, 0), 3)
#         # get radius of gyration
#         radius_gyr = round(r_g(frame_i, thresh_otsu, x_c, y_c))
#         cv2.circle(frame_after, (x_c, y_c), radius_gyr, (0, 255, 0), 3)

#         for contour_i in range(len(contours)):
#             m = cv2.moments(contours[contour_i])
#             if m["m00"] == 0:
#                 continue
#             x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]
#             x, y = round(x), round(y)
#             r_tmp = np.sqrt((x_c - x) ** 2 + (y_c - y) ** 2)
#             if r_tmp > 0.5 * radius_gyr:  # 重心半径の半分よりも大きいものを採用
#                 r_mean_tmp.append(np.sqrt((x_c - x) ** 2 + (y_c - y) ** 2))  # 中心からの距離を追加
#                 cv2.line(frame_after, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
#                 cv2.line(frame_after, (x + 5, y - 5), (x - 5, y + 5), (0, 0, 255), 2)
#                 x_list.append(x)
#                 y_list.append(y)

#         # plot center of mass
#         cv2.line(frame_after, (x_c - 5, y_c - 5), (x_c + 5, y_c + 5), (0, 255, 0), 2)
#         cv2.line(frame_after, (x_c + 5, y_c - 5), (x_c - 5, y_c + 5), (0, 255, 0), 2)

#         # if len(r_mean_tmp) > 0:
#         #     r_mean = round(np.mean(r_mean_tmp))
#         #     r_std = round(np.std(r_mean_tmp))
#         #     x_list = [x_list[i] for i in range(len(r_mean_tmp)) if r_mean - r_std < r_mean_tmp[i]]
#         #     y_list = [y_list[i] for i in range(len(r_mean_tmp)) if r_mean - r_std < r_mean_tmp[i]]
#         #     for i in range(len(x_list)):
#         #         cv2.line(frame_after, (x_list[i] - 5, y_list[i] - 5), (x_list[i] + 5, y_list[i] + 5), (0, 0, 255), 2)
#         #         cv2.line(frame_after, (x_list[i] + 5, y_list[i] - 5), (x_list[i] - 5, y_list[i] + 5), (0, 0, 255), 2)
#         #     cv2.circle(frame_after, (x_c, y_c), r_mean, (0, 0, 255), 3)

#         out_tracking.write(frame_after)

#     print("Tracking Movie is done.")


def mk_tracking_movie(start_frame: int, end_frame: int, frame_step: int, dust: int):  # thinngを使うver
    # get Otsu threshold value
    frame = n_frame_image(total_frames - 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_otsu, _ = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_OTSU)

    for frame_i in tqdm(range(start_frame + frame_step, end_frame - frame_step, frame_step), desc="Tracking"):
        frame_RGB = remove_dust(n_frame_image(frame_i), dust, thresh_otsu)
        frame_gray = cv2.cvtColor(frame_RGB, cv2.COLOR_BGR2GRAY)
        _, frame_binary = cv2.threshold(frame_gray, thresh_otsu, 255, cv2.THRESH_BINARY)
        skelton = cv2.ximgproc.thinning(frame_binary, thinningType=cv2.ximgproc.THINNING_GUOHALL)  # 細線化処理
        for i in range(1, len(skelton) - 1):
            for j in range(1, len(skelton[i]) - 1):
                if skelton[i][j] != 0:
                    count = search(skelton, i, j, 3)
                    if count <= 1:  # dust(大きさ１の孤立ピクセル)の消去処理。万が一のerror処理
                        skelton[i][j] = 0
                    elif count < 4:
                        skelton[i][j] = count  # 2:edge,3:branch point
                    elif count >= 4:
                        skelton[i][j] = 4  # 4:node flag
        # get radius of gyration
        radius_gyr = round(r_g(frame_i, thresh_otsu, x_c, y_c))
        cv2.circle(frame_RGB, (x_c, y_c), radius_gyr, (0, 255, 0), 3)

        # plot center of mass
        cv2.line(frame_RGB, (x_c - 5, y_c - 5), (x_c + 5, y_c + 5), (0, 255, 0), 2)
        cv2.line(frame_RGB, (x_c + 5, y_c - 5), (x_c - 5, y_c + 5), (0, 255, 0), 2)

        # plot thinning image
        frame_RGB[skelton == 2] = [0, 0, 255]  # edge
        frame_RGB[skelton == 3] = [0, 255, 0]  # branch
        frame_RGB[skelton == 4] = [255, 0, 0]  # node

        out_tracking.write(frame_RGB)

    print("Tracking Movie is done.")


# main
unit = 231.34 / 5.0  # 5cm=231.34px

# time
start_time = time.time()

# constants
dust = 4  # チリの大きさ判定用変数

# Video Source
dir_path = "/mnt/d/master_thesis_data/experment_data/movie_data/movie_data/data_for_analisis"
surfactant_list = os.listdir(dir_path)
for surafactant in surfactant_list:
    if not surafactant == surfactant_list[0]:
        continue
    surafactant_path = os.path.join(dir_path, surafactant)
    file_list = os.listdir(surafactant_path)
    for file in file_list:
        if not file == file_list[1]:
            continue
        file_path = os.path.join(surafactant_path, file)
        get_parms = re.findall(r"(.*?)_", file)
        file_path_avi = file_path + "/" + file + "_edited.avi"
        base = os.path.basename(file_path_avi)

        # Reading Video
        cap = cv2.VideoCapture(file_path_avi)

        # Check Video Source
        check_video_source(file_path_avi)

        # Video Information
        fname, Lx, Ly, fps, total_frames, total_time, frame_step = video_info(file_path_avi)

        # Start Movie Analize
        print("Start Analize")

        # get center of mass
        x_c, y_c = center()

        # Making Front Tracking VideoWriter & Movie
        output_tracking_movie = file_path_avi.replace(".avi", "_tracking.avi")  # 保存する動画ファイル名
        fourcc = cv2.VideoWriter.fourcc("I", "4", "2", "0")  # 動画保存時のfourcc設定、無圧縮avi
        if os.path.exists(output_tracking_movie):
            os.remove(output_tracking_movie)
        # 録画データにトラッキング点を追加して保存するためI420かつisColor=Trueとする。データはY800のはずなのに3つ引数があるのは謎？
        out_tracking = cv2.VideoWriter(output_tracking_movie, fourcc, fps, (Lx, Ly), isColor=True)
        mk_tracking_movie(
            0, total_frames - 1, frame_step, 4
        )  # FFmpegはI420形式の動画を読み込めないので一括で処理すること。

        cap.release()

        finish_time = time.time()
        print("Total time : ", finish_time - start_time)
