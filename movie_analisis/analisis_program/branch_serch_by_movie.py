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
    def __init__(self, x=0.0, y=0.0, time=0.0, branch_no=0, in_branch=False, edge=False, node=False):
        self.x = x
        self.y = y
        self.time = time
        self.branch_no = branch_no
        self.in_branch = in_branch
        self.edge = edge
        self.node = node

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

        self.last_img = frame
        self.bool = cap.isOpened()
        self.path = file_path_avi
        self.fname = fname
        # 5cmあたりのピクセル数
        self.unit = float(param[6].replace("pix", "")) / 5.0
        self.Lx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.Ly = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) / 10
        self.otsu = thresh_otsu
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_time = self.total_frames / self.fps
        # おおよそ6ピクセル成長するフレーム数
        self.frame_delta = int(6 * self.total_frames / (10 * self.unit))

    # 動画情報の表示
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
        print("Frame Delta : ", self.frame_delta, " frame")
        print("----------------------------")

    # 解析開始・終了点の設定
    def get_start_end(self, file_path_dci: str, dci_mA_diff_thresh: float):
        df = pd.read_excel(file_path_dci)
        time = df["#Time(sec.msec)"]
        dci = df["#DCI Value(A)"]
        dci_mA = dci * 1000
        window_size = len(time[time < 60])  # 60秒間(分のデータ数)の移動平均を取る
        dci_mA_mean = dci_mA.rolling(window=window_size, min_periods=1).mean()
        dci_mA_diff = np.gradient(dci_mA_mean, time)
        for i in range(len(dci_mA_diff)):
            if dci_mA_diff[i] < dci_mA_diff_thresh:
                # plt.vlines(time[i], 0, 700, "red", linestyles="dashed", alpha=0.5)
                start_frame = int(time[i] // 10)
                break
        for i in range(int(len(dci_mA_diff) / 2), len(dci_mA_diff)):
            if dci_mA_diff[i] > dci_mA_diff_thresh:
                # plt.vlines(time[i], 0, 700, "blue", linestyles="dashed", alpha=0.5)
                end_frame = int(time[i] // 10)
                break
            else:
                end_frame = self.total_frames

        return start_frame, end_frame


def mk_dust_removed_frames(file_path_avi: str, thresh_otsu: float, frame_index: int):
    dust = 4
    cap = cv2.VideoCapture(file_path_avi)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    # チリの除去＆二値化
    black = np.zeros_like(frame, dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh_otsu, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour_i in range(len(contours)):
        if cv2.contourArea(contours[contour_i]) > dust:
            cv2.fillPoly(black, [contours[contour_i]], (255, 255, 255))
    dust_removed_gray = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
    _, dust_removed_binary = cv2.threshold(dust_removed_gray, 127, 255, cv2.THRESH_BINARY)

    return dust_removed_binary


def mk_particl_frames(
    file_path_avi: str, thresh_otsu: float, total_frame: int, start_frame: int, end_frame: int, frame_delta: int
):
    particle_frames = []
    for frame_i in tqdm(range(start_frame, end_frame), desc="Making Particle Frames"):
        before_frame = mk_dust_removed_frames(file_path_avi, thresh_otsu, frame_i)
        if not frame_i + frame_delta > total_frame:
            after_frame = mk_dust_removed_frames(file_path_avi, thresh_otsu, frame_i + frame_delta)
        else:
            after_frame = mk_dust_removed_frames(file_path_avi, thresh_otsu, total_frame)
        frame_xor = cv2.bitwise_xor(after_frame, before_frame)
        frame_diff = cv2.bitwise_and(after_frame, frame_xor)
        binary_movie.write(after_frame)
        particle_frames.append(frame_diff)

    return particle_frames


def analyze_frame(particle_frame, time, min_size, max_size):
    particles = []
    contours, _ = cv2.findContours(particle_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_size <= area <= max_size:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                particles.append(Particle(cx, cy, time))
    return particles


def tree_search(all_particles, particle, n_frame, max_velocity):
    particle.in_branch = True
    # max_velocity未満の粒子を探す
    next_particles = [
        next_particle
        for sublist in all_particles[n_frame + 1 : min(n_frame + 4, len(all_particles))]
        for next_particle in sublist
    ]
    next_particles = [
        next_particle for next_particle in next_particles if particle.distance(next_particle) < max_velocity
    ]
    for next_particle in next_particles:
        if next_particle.in_branch:
            continue
        tree_search(all_particles, next_particle, n_frame + 1, max_velocity)
    # distance_tmp = 1000  # 十分大きな値
    # found_particle = None
    # for next_particle in next_particles:
    #     if next_particle.in_branch:
    #         continue
    #     distance = particle.distance(next_particle)
    #     # 速度がmax_velocity未満かつ最短距離の粒子を見つける
    #     if distance < distance_tmp:
    #         distance_tmp = distance
    #         found_particle = next_particle
    # if found_particle:
    #     tree_search(all_particles, found_particle, n_frame + 1, max_velocity)
    # else:
    #     particle.edge = True


def track_particles(particle_frames, start_frame, min_size, max_size, max_velocity, min_track_length, last_img):
    n_frames = len(particle_frames)
    all_particles = []
    time = (start_frame + 1) * 10

    # 各フレームで粒子を検出
    for frame_i in range(n_frames):
        particles = analyze_frame(particle_frames[frame_i], time, min_size, max_size)
        # Plot particles on the frame
        for particle in particles:
            cv2.circle(last_img, (int(particle.x), int(particle.y)), 2, (0, 255, 0), -1)
        tracking_movie.write(last_img)
        all_particles.append(particles)
        time = time + 10

    branches = []
    edge = []
    node = []
    branch_count = 0

    # 枝の検出&再構成
    # tracks = []
    # track_count = 0

    # for i in range(n_frames):
    #     for particle in all_particles[i]:
    #         if not particle.in_branch:
    #             track_count += 1
    #             particle.in_branch = True
    #             particle.branch_no = track_count
    #             track = [particle]

    #             for j in range(i + 1, n_frames):
    #                 found_particle = None
    #                 for next_particle in all_particles[j]:
    #                     if not next_particle.in_branch and particle.distance(next_particle) < max_velocity:
    #                         if found_particle is None or particle.distance(next_particle) < particle.distance(
    #                             found_particle
    #                         ):
    #                             found_particle = next_particle

    #                 if found_particle:
    #                     found_particle.in_branch = True
    #                     found_particle.branch_no = track_count
    #                     track.append(found_particle)
    #                     particle = found_particle
    #                 else:
    #                     break

    #             if len(track) >= min_track_length:
    #                 tracks.append(track)

    # return tracks
    # for first_particle in all_particles[0]:
    #     first_particle.edge = True
    #     tree_search(all_particles, first_particle, 0, max_velocity)  # 枝のrootの粒子から開始

    # for all_particle in all_particles:
    #     for particle in all_particle:
    #         if particle.edge:
    #             edge.append(particle)
    #         elif particle.node:
    #             node.append(particle)
    # return edge, node
    # for frame_i in reversed(range(n_frames)):
    #     for particle in all_particles[frame_i]:
    #         if particle.in_branch:
    #             continue
    #         branch_count += 1
    #         particle.edge = True
    #         particle.in_branch = True
    #         particle.branch_no = branch_count
    #         branch = [particle]
    #         for frame_j in reversed(range(frame_i)):
    #             found_particle = None
    #             # 距離がmax_velocity未満の粒子をリストアップ
    #             next_particles = [
    #                 next_particle
    #                 for sublist in all_particles[max(0, frame_j - 4) : frame_j]
    #                 for next_particle in sublist
    #                 if particle.distance(next_particle) < max_velocity
    #             ]
    #             distance_tmp = 1000  # 十分大きな値
    #             for next_particle in next_particles:
    #                 distance = particle.distance(next_particle)
    #                 # 速度がmax_velocity以下の場合かつ最短距離の粒子を見つける
    #                 if distance < distance_tmp:
    #                     distance_tmp = distance
    #                     found_particle = next_particle

    #             # 空でない文字列はTrue
    #             if found_particle and not found_particle.in_branch:
    #                 found_particle.in_branch = True
    #                 found_particle.branch_no = branch_count
    #                 branch.append(found_particle)
    #                 particle = found_particle
    #             elif found_particle and found_particle.in_branch:
    #                 found_particle.node = True
    #                 branch.append(found_particle)
    #                 break
    #             else:
    #                 break

    #         if len(branch) >= min_track_length:  # 最低トラック長の設定
    #             branches.append(branch)

    return branches


# メイン処理
if __name__ == "__main__":
    # time
    start_time = time.time()

    # set parameters
    min_size = 1
    max_size = 999999
    min_track_length = 1
    max_velocity = 10.0
    show_labels = False
    show_positions = False
    show_paths = True
    show_path_lengths = False
    save_results_file = False
    # setting for video writer
    fourcc = cv2.VideoWriter.fourcc("M", "J", "P", "G")  # 動画保存時のfourcc設定、確認のためだけなので圧縮形式

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

            # get video file path
            file_path = os.path.join(surafactant_path, file)
            file_path_avi = file_path + "/" + file + "_edited.avi"
            file_path_dci = file_path + "/" + file + ".xlsm"
            file_path_png = file_path + "/" + file + "_tracking.png"
            video = Video(file_path_avi)
            video.show_info()
            file_path_tracking = file_path_avi.replace(".avi", "_tracking.avi")  # 保存する動画ファイル名
            file_path_tracking_binary = file_path_avi.replace(".avi", "_tracking_binary.avi")  # 保存する動画ファイル名
            tracking_movie = cv2.VideoWriter(file_path_tracking, fourcc, video.fps, (video.Lx, video.Ly), isColor=True)
            binary_movie = cv2.VideoWriter(
                file_path_tracking_binary, fourcc, video.fps, (video.Lx, video.Ly), isColor=False
            )

            # get particle frames
            print("Start Analize")
            start_frame, end_frame = video.get_start_end(file_path_dci, 0.25)
            # end_frame = video.total_frames - video.frame_step
            start_frame = 600
            end_frame = 700
            particle_frames = mk_particl_frames(
                file_path_avi, video.otsu, video.total_frames, start_frame, end_frame, video.frame_delta
            )
            get_particle_time = time.time()
            print("Get Particle Frames : ", get_particle_time - start_time)

            # tracking particles
            branches = track_particles(
                particle_frames, start_frame, min_size, max_size, max_velocity, min_track_length, video.last_img
            )
            get_tracks_time = time.time()
            print("Get Tracks : ", get_tracks_time - get_particle_time)
            # if show_paths:
            #     origin_frames = video.last_img
            #     for track in tracks:
            #         for i in range(len(track) - 1):
            #             cv2.line(
            #                 origin_frames,
            #                 (int(track[i].x), int(track[i].y)),
            #                 (int(track[i + 1].x), int(track[i + 1].y)),
            #                 (255, 0, 0),
            #                 2,
            #             )
            #     cv2.imwrite(file_path_png, origin_frames)
            if show_paths:
                origin_frames = video.last_img
                for branch in branches:
                    cv2.circle(origin_frames, (int(branch[0].x), int(branch[0].y)), 2, (255, 0, 0), -1)
                    cv2.circle(origin_frames, (int(branch[-1].x), int(branch[-1].y)), 2, (0, 0, 255), -1)
                # for edge in edges:
                #     cv2.circle(origin_frames, (int(edge.x), int(edge.y)), 2, (0, 0, 255), -1)
                # for node in nodes:
                #     cv2.circle(origin_frames, (int(node.x), int(node.y)), 2, (255, 0, 0), -1)
                cv2.imwrite(file_path_png, origin_frames)
            print("Total time : ", time.time() - start_time)

sys.exit(0)
