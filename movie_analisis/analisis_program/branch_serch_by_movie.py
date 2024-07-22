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
from tqdm import tqdm


def n_frame_image(frame_index: int):  # return n frame image
    # インデックスがフレームの範囲内なら…
    if 0 <= frame_index < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # フレームの位置変更，全体で開始フレームが変わるので注意
        ret, img = cap.read()  # ret:bool値(画像が読めれば True) img:画像のndarray
    else:
        print("Out of video frames")
        sys.exit(1)

    return img


def check_video_source(file_path: str):
    if not cap.isOpened():
        print("Video reading error :" + file_path)
        sys.exit(1)
    elif cap.isOpened():
        print("Video was readed :" + file_path)

    print("----------------------------")


def video_info(file_path: str):
    fname = os.path.basename(file_path)
    Lx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Ly = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) / 10
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sec_per_frame = 10
    frame_step = int(total_frames / (10 * unit)) + 2

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


def mk_tracking_movie(start_frame: int, end_frame: int, frame_step: int, dust: int):
    # get Otsu threshold value
    frame = n_frame_image(total_frames - 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_otsu, _ = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_OTSU)
    # get initital frame
    frame_befor = remove_dust(n_frame_image(start_frame), dust, thresh_otsu)

    for frame_i in tqdm(range(start_frame + frame_step, end_frame - frame_step, frame_step), desc="Tracking"):
        # frame_befor = remove_dust(n_frame_image(frame_i), dust, thresh_otsu)
        frame_after = remove_dust(n_frame_image(frame_i + frame_step), dust, thresh_otsu)
        frame_diff = cv2.subtract(frame_after, frame_befor)
        # frame_diff = cv2.absdiff(frame_befor, frame_after)
        frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, frame_diff = cv2.threshold(frame_diff, thresh_otsu, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_after, contours, -1, (0, 255, 0), 3)

        for contour_i in range(len(contours)):
            m = cv2.moments(contours[contour_i])
            if m["m00"] == 0:
                continue
            # elif m["m00"] < 1:
            #     continue
            x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]
            x, y = round(x), round(y)
            cv2.line(frame_after, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
            cv2.line(frame_after, (x + 5, y - 5), (x - 5, y + 5), (0, 0, 255), 2)

        out_tracking.write(frame_after)

        frame_befor = cv2.add(frame_befor, frame_after)

    print("Tracking Movie is done.")


# main
unit = 231.34 / 5.0  # 5cm=231.34px

# time
start_time = time.time()

# constants
dust = 4  # チリの大きさ判定用変数

# Video Source
file_path = "/mnt/d/master_thesis_data/experment_data/movie_data/movie_data/data_for_analisis/NOTWEEN/20240630_2M_NOTWEEN_0.00mM_5.99mPas_76.4mN_0001/20240630_2M_NOTWEEN_0.00mM_5.99mPas_76.4mN_0001_edited.avi"
# file_path = "/mnt/d/master_thesis_data/experment_data/movie_data/movie_data/data_for_analisis/TWEEN20/20240716_2M_TWEEN20_0.005mM_5.92mPas_58.7mN_0001/20240716_2M_TWEEN20_0.005mM_5.92mPas_58.7mN_0001_edited.avi"
# file_path = "/mnt/d/master_thesis_data/experment_data/movie_data/movie_data/data_for_analisis/TWEEN20/20240704_2M_TWEEN20_0.04mM_6.23mPas_40.9mN_0001/20240704_2M_TWEEN20_0.04mM_6.23mPas_40.9mN_0001_edited.avi"

# Reading Video
cap = cv2.VideoCapture(file_path)

# Check Video Source
check_video_source(file_path)

# Video Information
fname, Lx, Ly, fps, total_frames, total_time, frame_step = video_info(file_path)

# Start Movie Analize
print("Start Analize")

# Making Front Tracking VideoWriter & Movie
name_tag = fname.replace(".avi", "")
output_tracking_movie = file_path.replace(".avi", "") + "_tracking.avi"  # 保存する動画ファイル名
fourcc = cv2.VideoWriter.fourcc("I", "4", "2", "0")  # 動画保存時のfourcc設定、無圧縮avi
if os.path.exists(output_tracking_movie):
    os.remove(output_tracking_movie)
# 録画データにトラッキング点を追加して保存するためI420かつisColor=Trueとする。データはY800のはずなのに3つ引数があるのは謎？
out_tracking = cv2.VideoWriter(output_tracking_movie, fourcc, fps, (Lx, Ly), isColor=True)
mk_tracking_movie(0, total_frames - 1, frame_step, 4)  # FFmpegはI420形式の動画を読み込めないので一括で処理すること。

cap.release()

finish_time = time.time()
print("Total time : ", finish_time - start_time)
