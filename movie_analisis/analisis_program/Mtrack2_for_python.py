import cv2
import numpy as np
import math
import os

# パラメータの設定
min_size = 1
max_size = 999999
min_track_length = 2
max_velocity = 10.0
show_labels = False
show_positions = False
show_paths = False
show_path_lengths = False
save_results_file = False


class Particle:
    def __init__(self, x=0, y=0, z=0, track_nr=0, in_track=False, flag=False):
        self.x = x
        self.y = y
        self.z = z
        self.track_nr = track_nr
        self.in_track = in_track
        self.flag = flag

    def copy(self, source):
        self.x = source.x
        self.y = source.y
        self.z = source.z
        self.in_track = source.in_track
        self.flag = source.flag

    def distance(self, p):
        return math.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2)


def analyze_frame(frame, min_size, max_size):
    particles = []
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_size <= area <= max_size:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                particles.append(Particle(cx, cy))
    return particles


def track_particles(frames, min_size, max_size, max_velocity, min_track_length):
    n_frames = len(frames)
    all_particles = []

    for i in range(n_frames):
        particles = analyze_frame(frames[i], min_size, max_size)
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

                if len(track) >= min_track_length:
                    tracks.append(track)

    return tracks


def save_results(tracks, filename):
    with open(filename, "w") as f:
        f.write("Frame\tX\tY\tTrack\n")
        for track in tracks:
            for particle in track:
                f.write(f"{particle.z+1}\t{particle.x}\t{particle.y}\t{particle.track_nr}\n")


# メイン関数
def main():
    # ここで画像を読み込み、各フレームを処理する
    video = cv2.VideoCapture("video_file_path")  # 例: 動画からフレームを抽出
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        frames.append(thresh)

    tracks = track_particles(frames, min_size, max_size, max_velocity, min_track_length)

    if save_results_file:
        directory = "output_directory"
        filename = os.path.join(directory, "track_results.txt")
        save_results(tracks, filename)

    if show_paths:
        for track in tracks:
            for i in range(len(track) - 1):
                cv2.line(frames[track[i].z], (track[i].x, track[i].y), (track[i + 1].x, track[i + 1].y), (255, 0, 0), 2)
        cv2.imshow("Tracks", frames[-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# 説明
# ・Particleクラス: 各粒子の位置と追跡ステータスを保持する。
# ・analyze_frame関数: 各フレームで粒子を検出し、位置情報を取得します。
# ・track_particles関数: 各フレームの粒子を追跡し、トラックを形成します。
# ・save_results関数: 追跡結果をテキストファイルに保存します。
# 注意
# ・このコードはシンプルな実装であり、オリジナルのJavaコードにあったすべての機能を完全には再現していません。例えば、パスの描画、結果の表示、詳細なオプション設定などが含まれていません。必要に応じて追加・調整してください。
