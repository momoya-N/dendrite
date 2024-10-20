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
import csv


class Particle:
    def __init__(self, x=0.0, y=0.0, time=0.0, branch_no=[], in_branch=False, edge=False, node=False):
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


def track_particles(particle_frames, start_frame, min_size, max_size, max_velocity, min_branch_length, last_img):
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
    branch_no = 0

    for i in range(n_frames):
        for particle in all_particles[i]:
            if particle.in_branch:
                continue
            branch_no += 1
            particle.in_branch = True
            particle.branch_no = particle.branch_no + [branch_no]
            particle.edge = True
            branch = [particle]

            before_particles = [
                before_particle
                for tmp in all_particles[max(i - 3, 0) : i]
                for before_particle in tmp
                if particle.distance(before_particle) < max_velocity and before_particle.in_branch
            ]
            found_particle = None
            dist_tmp = 1000  # とりあえず大きな値を入れておく
            # 3つ前までのフレームに存在する粒子の中で最も近い粒子を探す
            for before_particle in before_particles:
                if particle.distance(before_particle) < dist_tmp:
                    dist_tmp = particle.distance(before_particle)
                    found_particle = before_particle
            if found_particle:
                found_particle.in_branch = True
                found_particle.branch_no = found_particle.branch_no + [branch_no]
                found_particle.node = True
                particle.edge = False
                branch.insert(0, found_particle)

            for j in range(i + 1, n_frames):
                found_particle = None
                next_particles = [
                    next_particle
                    for tmp in all_particles[j : min(j + 2, len(all_particles))]
                    for next_particle in tmp
                    if not next_particle.in_branch and particle.distance(next_particle) < max_velocity
                ]
                # 最短距離のnext particleを探す
                for next_particle in next_particles:
                    if found_particle is None or particle.distance(next_particle) < particle.distance(found_particle):
                        found_particle = next_particle

                if found_particle:
                    found_particle.in_branch = True
                    found_particle.branch_no = found_particle.branch_no + [branch_no]
                    branch.append(found_particle)
                    particle = found_particle
                else:
                    particle.edge = True
                    break

            if len(branch) >= min_branch_length:
                branches.append(branch)

    return branches


# メイン処理
if __name__ == "__main__":
    # time
    start_time = time.time()

    # set parameters
    min_size = 1
    max_size = 999999
    min_branch_length = 3
    max_velocity = 6.0
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
            start_frame = 700
            end_frame = 800
            particle_frames = mk_particl_frames(
                file_path_avi, video.otsu, video.total_frames, start_frame, end_frame, video.frame_delta
            )
            get_particle_time = time.time()
            print("Get Particle Frames : ", get_particle_time - start_time)

            # tracking particles
            branches = track_particles(
                particle_frames, start_frame, min_size, max_size, max_velocity, min_branch_length, video.last_img
            )
            # Write branches to CSV
            csv_file_path = file_path_avi.replace(".avi", "_branches.csv")
            with open(csv_file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Branch No", "X", "Y", "Time", "In Branch", "Edge", "Node"])
                for branch_no, branch in enumerate(branches, start=1):
                    for particle_no, particle in enumerate(branch, start=1):
                        writer.writerow(
                            [
                                particle.branch_no,
                                particle.x,
                                particle.y,
                                particle.time,
                                particle.in_branch,
                                particle.edge,
                                particle.node,
                            ]
                        )
            print(f"Branches data saved to {csv_file_path}")
            get_tracks_time = time.time()
            print("Get Tracks : ", get_tracks_time - get_particle_time)
            if show_paths:
                origin_frames = video.last_img
                for branch in branches:
                    # node
                    cv2.circle(origin_frames, (int(branch[0].x), int(branch[0].y)), 2, (255, 0, 0), -1)
                    # edge
                    cv2.circle(origin_frames, (int(branch[-1].x), int(branch[-1].y)), 2, (0, 0, 255), -1)
                    # branch
                    node = branch[0]
                    # 枝の途中にnodeがある場合の分割描画処理
                    for i in range(len(branch)):
                        if branch[i].node:
                            edge = branch[i]
                            cv2.line(
                                origin_frames,
                                (int(node.x), int(node.y)),
                                (int(edge.x), int(edge.y)),
                                (255, 255, 0),
                                1,
                            )
                            node = branch[i]
                    # 最後のnodeからedgeまでの描画処理
                    edge = branch[-1]
                    cv2.line(
                        origin_frames,
                        (int(node.x), int(node.y)),
                        (int(edge.x), int(edge.y)),
                        (255, 255, 0),
                        1,
                    )
                cv2.imwrite(file_path_png, origin_frames)
            print("Total time : ", time.time() - start_time)

sys.exit(0)
