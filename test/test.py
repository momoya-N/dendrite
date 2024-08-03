import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


def n_frame_image(frame_index: int):  # return n frame image
    # インデックスがフレームの範囲内なら…
    if 0 <= frame_index < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # フレームの位置変更，全体で開始フレームが変わるので注意
        ret, img = cap.read()  # ret:bool値(画像が読めれば True) img:画像のndarray
    else:
        print("Out of video frames")
        sys.exit(1)

    return img


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


def cm():
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


def r_g(frame: int, x: int, y: int):
    tmp_rg = 0
    img = n_frame_image(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_otsu, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nonzero = np.nonzero(binary)
    for i in range(len(nonzero[0])):
        tmp_rg += (nonzero[0][i] - y) ** 2 + (nonzero[1][i] - x) ** 2
    tmp_rg = tmp_rg / len(nonzero[0])
    r_g = np.sqrt(tmp_rg)

    return r_g


dir_path = "/mnt/d/master_thesis_data/experment_data/movie_data/movie_data/data_for_analisis"
surafactant_list = os.listdir(dir_path)
for surafactant in surafactant_list:
    if not surafactant == surafactant_list[0]:
        continue
    surafactant_path = os.path.join(dir_path, surafactant)
    file_list = os.listdir(surafactant_path)
    for file in file_list:
        if not file == file_list[1]:
            continue
        file_path = os.path.join(surafactant_path, file)
        print(file_path)
        get_parms = re.findall(r"(.*?)_", file)
        file_path_avi = file_path + "/" + file + "_edited.avi"
        print(file_path_avi)
        print(os.path.exists(file_path_avi))
        base = os.path.basename(file_path_avi)
        print(base)
        cap = cv2.VideoCapture(file_path_avi)
        x, y = cm()
        radius = round(r_g(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1), x, y))
        frame = n_frame_image(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1))
        frame_RGB = remove_dust(frame, 4, 51)
        frame_gray = cv2.cvtColor(frame_RGB, cv2.COLOR_BGR2GRAY)
        _, frame_binary = cv2.threshold(frame_gray, 51, 255, cv2.THRESH_BINARY)
        skeleton = cv2.ximgproc.thinning(frame_binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)  # 細線化処理
        print(skeleton)
        print(np.shape(skeleton))
        print(x, y, radius)

        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        ret, img = cap.read()
        cv2.line(img, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), 2)
        cv2.line(img, (x + 5, y - 5), (x - 5, y + 5), (0, 255, 0), 2)
        cv2.circle(img, (x, y), radius, (0, 255, 0), 3)
        img[skeleton == 255] = (255, 0, 0)  # 赤色
        plt.imshow(img)
        plt.show()
