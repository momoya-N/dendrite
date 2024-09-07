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


class Particle:
    def __init__(self, x=0, y=0, z=0, track_nr=0, in_track=False):
        self.x = x
        self.y = y
        self.z = z
        self.track_nr = track_nr
        self.in_track = in_track

    def distance(self, p):
        return math.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2)


class Video:
    def __init__(self, file_path_avi: str):
        cap = cv2.VideoCapture(file_path_avi)
        fname = os.path.basename(file_path_avi)
        param = fname.split("_")
        # get Otsu threshold value
        last_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame)
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh_otsu, _ = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_OTSU)

        self.bool = cap.isOpened()
        self.path = file_path_avi
        self.fname = fname
        self.unit = float(param[6].replace("pix", "")) / 5.0
        self.Lx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.Ly = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) / 10
        self.otsu = thresh_otsu
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_time = self.total_frames / self.fps
        self.frame_step = int(5 * self.total_frames / (10 * self.unit))

    def show_info(self):
        if not self.bool:
            print("Video reading error :" + self.fname)
            sys.exit(1)
        print("File Name : ", self.fname)
        print("Unit : ", self.unit, " pix/cm")
        print("Frame Width : ", self.Lx, " pix")
        print("Frame Hight : ", self.Ly, " pix")
        print("FPS : ", self.fps, " frame/s")
        print("Otsu Threshold : ", self.otsu)
        print("Frame Count : ", self.total_frames, " frame")
        h = int(self.total_time // 3600)
        m = int((self.total_time % 3600) // 60)
        s = int(self.total_time % 60)
        print("Total time  : ", self.total_time, "s (", h, "h", m, "m", s, "s)")
        print("Frame Step : ", self.frame_step, " frame")
        print("----------------------------")


# 動画操作関連
# フレーム取得
def n_frame_image(frame_index: int):  # return n frame image
    # インデックスがフレームの範囲内なら…
    if 0 <= frame_index < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # フレームの位置変更，全体で開始フレームが変わるので注意
        ret, img = cap.read()  # ret:bool値(画像が読めれば True) img:画像のndarray
    else:
        print("Out of video frames")
        sys.exit(1)

    return img


# 解析開始・終了点の設定
def set_start_end_point(file_path_dci: str, dci_mA_diff_thresh: float):  # dci_thresh: 単位 mA
    df = pd.read_excel(file_path_dci)
    time = df["#Time(sec.msec)"]
    dci = df["#DCI Value(A)"]
    dci_mA = dci * 1000
    window_size = len(time[time < 60])  # 60秒間の移動平均を取る
    dci_mA_mean = dci_mA.rolling(window=window_size, min_periods=1).mean()
    dci_mA_diff = np.gradient(dci_mA_mean, time)
    for i in range(len(dci_mA_diff)):
        if dci_mA_diff[i] < dci_mA_diff_thresh:
            # plt.vlines(time[i], 0, 700, "red", linestyles="dashed", alpha=0.5)
            start_frame = int(time[i] // 10 + 1)
            break
    for i in range(int(len(dci_mA_diff) / 2), len(dci_mA_diff)):
        if dci_mA_diff[i] > dci_mA_diff_thresh:
            # plt.vlines(time[i], 0, 700, "blue", linestyles="dashed", alpha=0.5)
            end_frame = int(time[i] // 10 + 1)
            break

    return start_frame, end_frame


# 動画解析関連
# チリ除去
# def remove_dust(frame, dust_area: int, thresh_otsu: float):
#     # チリを除去する関数
#     # チリの面積がdust_area以下の輪郭を除去する
#     # frame:入力画像,dust_area:チリとみなす面積の閾値
#     # 出力:チリを除去したbinary画像
#     black = np.zeros_like(frame, dtype=np.uint8)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, thresh_otsu, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for contour_i in range(len(contours)):
#         if cv2.contourArea(contours[contour_i]) > dust_area:
#             cv2.fillPoly(black, [contours[contour_i]], (255, 255, 255))

#     return black


def mk_dust_removed_frames(file_path_avi: str, thresh_otsu: float):
    dust = 4
    cap = cv2.VideoCapture(file_path_avi)
    frames = []
    # チリの除去＆二値化
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        black = np.zeros_like(frame, dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, thresh_otsu, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour_i in range(len(contours)):
            if cv2.contourArea(contours[contour_i]) > dust:
                cv2.fillPoly(black, [contours[contour_i]], (255, 255, 255))
        remove_dust_gray = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
        _, remove_dust_binary = cv2.threshold(remove_dust_gray, 127, 255, cv2.THRESH_BINARY)
        frames.append(remove_dust_binary)

    return frames


# 回転半径計算
def r_g(frame: int, thresh_otsu: float, x: int, y: int):
    tmp_rg = 0
    img = remove_dust(n_frame_image(frame), dust, thresh_otsu)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    nonzero = np.nonzero(binary)

    if len(nonzero[0]) == 0:
        return 0

    for i in range(len(nonzero[0])):
        tmp_rg += (nonzero[0][i] - y) ** 2 + (nonzero[1][i] - x) ** 2
    tmp_rg = tmp_rg / len(nonzero[0])
    r_g = np.sqrt(tmp_rg)

    return r_g


# 成長点のトラッキング
def mk_tracking_movie(start_frame: int, end_frame: int, frame_step: int, dust: int):
    # get Otsu threshold value
    frame = n_frame_image(total_frames - 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_otsu, _ = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_OTSU)
    # バックの輝度値は常に一定か？ー＞後半はほぼ一定で，最後の輝度値を取るので問題なさそう

    # make empty data array
    branch_pos = []
    data_len = len(list(range(start_frame, end_frame, frame_step)))
    time = np.zeros(data_len)
    particle = np.zeros(data_len)
    area = np.zeros(data_len)
    R_g = np.zeros(data_len)
    data = pd.DataFrame()

    # get initital frame
    frame_before = remove_dust(n_frame_image(start_frame), dust, thresh_otsu)
    befor_gray = cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY)
    _, before_binary = cv2.threshold(befor_gray, 127, 255, cv2.THRESH_BINARY)

    for frame_i in tqdm(range(start_frame + frame_step, end_frame, frame_step), desc="Tracking"):
        itr = (frame_i - start_frame) // frame_step - 1
        time[itr] = frame_i / fps
        frame_after = remove_dust(n_frame_image(frame_i), dust, thresh_otsu)
        after_gray = cv2.cvtColor(frame_after, cv2.COLOR_BGR2GRAY)
        _, after_binary = cv2.threshold(after_gray, 127, 255, cv2.THRESH_BINARY)

        # contureで領域がくっついているものを同じ枝とみなすとよい
        frame_xor = cv2.bitwise_xor(after_binary, before_binary)
        frame_diff = cv2.bitwise_and(after_binary, frame_xor)
        contours, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 0]
        cv2.drawContours(frame_after, contours, -1, (0, 255, 0), 3)
        # get radius of gyration
        radius_gyr = r_g(frame_i, thresh_otsu, x_c, y_c)
        R_g[itr] = radius_gyr

        for contour_i in range(len(contours)):
            m = cv2.moments(contours[contour_i])
            if m["m00"] == 0:
                continue
            x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]
            x, y = round(x), round(y)
            branch_pos.append([x, y])

        for i in range(len(branch_pos)):
            cv2.circle(frame_after, (branch_pos[i][0], branch_pos[i][1]), 2, (0, 0, 255), -1)

        # plot center of mass
        cv2.line(frame_after, (x_c - 5, y_c - 5), (x_c + 5, y_c + 5), (0, 255, 0), 2)
        cv2.line(frame_after, (x_c + 5, y_c - 5), (x_c - 5, y_c + 5), (0, 255, 0), 2)

        # get area
        particle_count = np.nonzero(after_binary)[0].shape[0]
        particle[itr] = particle_count
        area[itr] = particle_count / (unit**2)

        out_tracking.write(frame_after)
        before_binary = cv2.bitwise_or(before_binary, after_binary)

    # save data
    data["time"] = time
    data["particle"] = particle
    data["area"] = area
    data["R_g"] = R_g
    data.to_csv(file_path + "/data.csv", index=False)

    print("Tracking Movie is done.")


# メイン処理
if __name__ == "__main__":
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
            file_path_avi = file_path + "/" + file + "_edited.avi"
            file_path_dci = file_path + "/" + file + ".xlsm"
            video = Video(file_path_avi)
            video.show_info()
            start_frame, end_frame = set_start_end_point(file_path_dci, 0.2)
            sys.exit(1)

    start_frame = 0
    end_frame = len(frames)
    frame_step = 10  # 必要に応じて調整

    mk_tracking_movie(frames, start_frame, end_frame, frame_step, dust=4)


# 動画情報取得・初期化
def get_video_frames(file_path_avi):
    cap_tmp = cv2.VideoCapture(file_path_avi)
    # get Otsu threshold value
    frame_tmp = cap_tmp.set(
        cv2.CAP_PROP_POS_FRAMES, cap_tmp.get()
    )  # フレームの位置変更，全体で開始フレームが変わるので注意
    ret, img = cap.read()  # ret:bool値(画像が読めれば True) img:画像のndarray
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_otsu, _ = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_OTSU)
    cap = cv2.VideoCapture(file_path_avi)
    frames = []

    if not cap.isOpened():
        print("Video reading error :" + file_path_avi)
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


# 重心計算
# def center():
#     cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     thresh_otsu, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     # initialization
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     while 1:
#         ret, img = cap.read()
#         img = remove_dust(img, 4, thresh_otsu)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         _, binary = cv2.threshold(gray, thresh_otsu, 255, cv2.THRESH_BINARY)
#         count = np.count_nonzero(binary)
#         if count > 4:
#             m = cv2.moments(binary, True)
#             x, y = round(m["m10"] / m["m00"]), round(m["m01"] / m["m00"])
#             break

#     return x, y
# main
# time
start_time = time.time()

# constants
dust = 4  # チリの大きさ判定用変数

# Video Source


# Reading Video
cap = cv2.VideoCapture(file_path_avi)

# Check Video Source
check_video_source(file_path_avi)

# Video Information
fname, unit, Lx, Ly, fps, total_frames, total_time, frame_step = video_info(file_path_avi)

# get start and end frame
start_frame, end_frame = set_start_end_point(0.2)
print("Start Frame : ", start_frame, " End Frame : ", end_frame)

# Start Movie Analize
print("Start Analize")

# get center of mass
x_c, y_c = center()

# Making Front Tracking VideoWriter & Movie
output_tracking_movie = file_path_avi.replace(".avi", "_tracking.avi")  # 保存する動画ファイル名
fourcc = cv2.VideoWriter.fourcc("I", "4", "2", "0")  # 動画保存時のfourcc設定、無圧縮avi
# fourcc = cv2.VideoWriter.fourcc(*"MJPG")
if os.path.exists(output_tracking_movie):
    os.remove(output_tracking_movie)
# 録画データにトラッキング点を追加して保存するためI420かつisColor=Trueとする。データはY800のはずなのに3つ引数があるのは謎？
out_tracking = cv2.VideoWriter(output_tracking_movie, fourcc, fps, (Lx, Ly), isColor=True)
# FFmpegはI420形式の動画を読み込めないので一括で処理すること。
mk_tracking_movie(start_frame, end_frame, frame_step, 4)

cap.release()

finish_time = time.time()
print("Total time : ", finish_time - start_time)


# 以下はMtrack2のようなプログラムを作成するためのコード
import math
import sys
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def analyze_frame(frame, dust_area, thresh_otsu):
    particles = []
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > dust_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                particles.append(Particle(cx, cy))
    return particles


def track_particles(frames, min_size, max_velocity):
    n_frames = len(frames)
    all_particles = []

    # 各フレームで粒子を検出
    for i in range(n_frames):
        particles = analyze_frame(frames[i], min_size, 127)
        all_particles.append(particles)

    tracks = []
    track_count = 0

    for i in range(n_frames):
        for particle in all_particles[i]:
            if not particle.in_track:
                track_count += 1
                particle.in_track = True
                particle.track_nr = track_count
                track = [particle]

                for j in range(i + 1, n_frames):
                    found_particle = None
                    for next_particle in all_particles[j]:
                        if not next_particle.in_track and particle.distance(next_particle) < max_velocity:
                            if found_particle is None or particle.distance(next_particle) < particle.distance(
                                found_particle
                            ):
                                found_particle = next_particle

                    if found_particle:
                        found_particle.in_track = True
                        found_particle.track_nr = track_count
                        track.append(found_particle)
                        particle = found_particle
                    else:
                        break

                if len(track) >= 2:  # 最低トラック長の設定
                    tracks.append(track)

    return tracks


def save_tracking_results(tracks, file_path):
    data = {"Frame": [], "X": [], "Y": [], "Track": []}
    for track in tracks:
        for particle in track:
            data["Frame"].append(particle.z)
            data["X"].append(particle.x)
            data["Y"].append(particle.y)
            data["Track"].append(particle.track_nr)

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")


# 成長点のトラッキングと結果保存
def mk_tracking_movie(frames, start_frame, end_frame, frame_step, dust):
    # get Otsu threshold value
    frame = frames[-1]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_otsu, _ = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_OTSU)

    # create list of frames for tracking
    selected_frames = []
    for frame_i in range(start_frame, end_frame, frame_step):
        selected_frames.append(frames[frame_i])

    tracks = track_particles(selected_frames, dust, max_velocity=10.0)

    # トラッキング結果を保存
    output_csv = file_path_avi.replace(".avi", "_tracking_results.csv")
    save_tracking_results(tracks, output_csv)

    print("Tracking and saving results is done.")
