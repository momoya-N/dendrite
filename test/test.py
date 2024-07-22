def n_frame_image(frame_index: int):  # return n frame image
    # インデックスがフレームの範囲内なら…
    if 0 <= frame_index < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, img = cap.read()  # ret:bool値(画像が読めれば True) img:画像のndarray
    else:
        print("Out of video frames")
        sys.exit(1)

    return img


import cv2
import numpy as np
import sys
import os


# Video Source
file_path = "/mnt/c/Users/PC/Desktop/Master_Thesis/test/Result of flat02_cropped_short.avi"
# file_path = "/mnt/d/master_thesis_data/experment_data/movie_data/movie_data/data_for_analisis/NOTWEEN/20240624_2M_NOTWEEN_0.00mM_5.87mPas_77.1mN_0001/20240624_2M_NOTWEEN_0.00mM_5.87mPas_77.1mN_0001_edited.avi"

# Reading Video
if not os.path.exists(file_path):
    print("File not found")
    sys.exit(1)
cap = cv2.VideoCapture(file_path)
print(cap.isOpened())
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
Lx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Ly = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
output_movie = file_path.replace(".avi", "") + "_binary_test.avi"  # 保存する動画ファイル名
# fourcc = cv2.VideoWriter.fourcc("I", "4", "2", "0")  # 動画保存時のfourcc設定、無圧縮avi
fourcc = cv2.VideoWriter.fourcc("X", "V", "I", "D")  # 動画保存時のfourcc設定、無圧縮avi
# カラー画像をグレースケールに変換して動画として保存するためisColor=Falseとする。
out = cv2.VideoWriter(output_movie, 0, fps, (Lx, Ly), isColor=True)
print(out.isOpened())

img = n_frame_image(total_frames - 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh_otsu, _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
print("Otsu Threshold Value:", thresh_otsu)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while 1:
    ret, frame = cap.read()
    if not cap.isOpened():
        print("Video reading error :" + file_path)
        sys.exit(1)

    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame_binary = cv2.threshold(frame_gray, thresh_otsu, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(frame_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = [contour for contour in contours if cv2.contourArea(contour) > 4]
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 4:
            # 重心の計算
            m = cv2.moments(contour)
            x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]
            # 座標を四捨五入
            x, y = round(x), round(y)
            # 重心位置に x印を書く
            cv2.line(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
            cv2.line(frame, (x + 5, y - 5), (x - 5, y + 5), (0, 0, 255), 2)
        else:
            frame = cv2.fillPoly(frame, [contour], (0, 0, 0))

    out.write(frame)

cap = cv2.VideoCapture(output_movie)

cap.release()
cv2.destroyAllWindows()

# 参考：https://qiita.com/odaman68000/items/73eb101ba27af26057f1
