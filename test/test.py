import numpy as np
import cv2
from scipy.spatial import distance


# 前処理した各フレームでオブジェクトの重心を取得する関数
def get_centroids(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
    return centroids


# オブジェクト追跡を行う関数
def track_objects(frames):
    tracks = []
    max_dist = 50  # 適切な値に調整
    for i, frame in enumerate(frames):
        centroids = get_centroids(frame)
        if i == 0:
            for centroid in centroids:
                tracks.append([centroid])
        else:
            for track in tracks:
                last_position = track[-1]
                distances = distance.cdist([last_position], centroids)
                min_index = np.argmin(distances)
                if distances[0][min_index] < max_dist:
                    track.append(centroids[min_index])
                    centroids.pop(min_index)
                else:
                    track.append(None)
            for centroid in centroids:
                tracks.append([None] * i + [centroid])
    return tracks


# フレームリストを生成（例として二値化した画像を用いる）
frames = [cv2.threshold(cv2.imread(f"frame{i}.png", 0), 127, 255, cv2.THRESH_BINARY)[1] for i in range(1, 6)]

# オブジェクトを追跡
tracks = track_objects(frames)

# 結果の出力
for i, track in enumerate(tracks):
    print(f"Track {i+1}: {track}")
