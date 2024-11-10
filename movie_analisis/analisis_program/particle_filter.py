import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random

def particle_filter(file_path_avi: str,obj: list[int]):

    cap = cv2.VideoCapture(file_path_avi)

    if cap.isOpened() == False:
        sys.exit

    ret, frame = cap.read()
    h, w =frame.shape[:2]#大きさを取得

    fourcc = cv2.VideoWriter.fourcc("M", "J", "P", "G")  # 動画保存時のfourcc設定、確認のためだけなので圧縮形式
    output_dst = cv2.VideoWriter("C:/Users/PC/Desktop/Master_Thesis/test/Result of flat02_cropped_short_test.avi",fourcc,5.0,(w,h))#動画出力の設定

    np.random.seed(100)#乱数の初期化,毎回同じ乱数になる
    Np = 400#粒子の数
    # obj = [175,33] #目的の（追いかける）標的の座標0~512
    WD = 10

    px = np.zeros((Np),dtype=np.int64)#粒子のx座標
    py = np.zeros((Np),dtype=np.int64)#粒子のy座標
    lc = np.zeros((Np))#粒子の色の尤度
    ls = np.zeros((Np))#粒子の空間の尤度
    lt = np.zeros((Np))#粒子の尤度total
    index = np.arange(Np)

    #objの周りに撒く
    d = 2
    px = np.random.normal(obj[0], d, Np).astype(np.int64)
    py = np.random.normal(obj[1], d, Np).astype(np.int64)

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Analysing")
    while True:
        ret, frame = cap.read()#１枚読み込み
        if ret == False:
            break#最後になったらループから抜ける

        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#色の変換
        thresh,frame_binary=cv2.threshold(frame_gray, 200, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)#二値化

        gx = np.average(px)
        gy = np.average(py)#１フレーム前の粒子の重心

        for i in range(Np):
            # lc[i] = frame_binary[py[i],px[i]]
            lc[i] = frame[py[i],px[i]][1] / 255.0#輝度の尤度
            ls[i] = np.exp(-((px[i] - gx) ** 2 + (py[i] - gy) ** 2)/(WD ** 2))
            lt[i] = lc[i] * ls[i]
        lt = lt / lt.sum()

        pnew_index = np.array(random.choices(population=index,weights=lt,k=Np))
        pxnew = px[pnew_index] + np.random.randint(-15,15,Np)
        pynew = py[pnew_index] + np.random.randint(-15,15,Np)

        #リサンプリングした,ある程度ランダムウォーク
        #ランダムウォークで画面外に出る場合の処理
        px = np.where(pxnew > w-1, w-1, pxnew)
        py = np.where(pynew > h-1, h-1, pynew)
        px = np.where(px < 0, 0, px)
        py = np.where(py < 0, 0, py)

        for i in range(Np): #画像の中に粒子を描く
            cv2.circle(frame,(px[i],py[i]),1,(0,255,0),-1)
        cv2.circle(frame,(int(gx),int(gy)),5,(0,0,255),-1)#重心を赤い円で描く

        output_dst.write(frame)
        pbar.update()
    pbar.close()
#一粒子のみの場合のParticle Filter ->　ぶつかったり，分裂したりするものには使えない
#多粒子の場合は，要検討。
#参考文献：https://www.researchgate.net/publication/224711429_Multiple_Particle_Filtering
if __name__ == "__main__":
    file_path="C:/Users/PC/Desktop/Master_Thesis/test/Result of flat02_cropped_short.avi"
    # file_path="D:/master_thesis_data/experment_data/movie_data/movie_data/data_for_analisis/NOTWEEN/20240630_2M_NOTWEEN_0.00mM_5.99mPas_76.4mN_232.67pix/20240630_2M_NOTWEEN_0.00mM_5.99mPas_76.4mN_232.67pix_edited_tracking_gray.avi"
    print("Start")
    particle_filter(file_path,[172,32])#目的の標的の座標を指定
    print("Finish")