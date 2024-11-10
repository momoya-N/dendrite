import sys
import os
import shutil
import cv2
import numpy as np
import time
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import math



class Video:
    def __init__(self, file_path_avi: str):
        cap = cv2.VideoCapture(file_path_avi)
        fname = os.path.basename(file_path_avi)
        ret, frame = cap.read()

        self.bool = cap.isOpened()
        self.path = file_path_avi
        self.fname = fname
        self.scale=1/360 #cm/pix
        self.Lx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.Ly = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_time = self.total_frames / self.fps

    # 動画情報の表示
    def show_info(self):
        if not self.bool:
            print("Video reading error :" + self.fname)
            sys.exit(1)
        print("File Name : ", self.fname)
        print("Frame Width : ", self.Lx, " pix")
        print("Frame Hight : ", self.Ly, " pix")
        print("FPS : ", self.fps, " frame/s")
        print("Frame Count : ", self.total_frames, " frame")
        h = int(self.total_time // 3600)
        m = int((self.total_time % 3600) // 60)
        s = int(self.total_time % 60)
        print("Total time  : ", self.total_time, "s (", h, "h", m, "m", s, "s)")
        print("----------------------------")

def dust_remove(file_path,fps,Lx,Ly):
    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter.fourcc("M", "J", "P", "G")  # 動画保存時のfourcc設定、確認のためだけなので圧縮形式
    file_path_dust_removed_avi = file_path.replace(".avi", "_dust_removed.avi")
    dust_removed_movie = cv2.VideoWriter(file_path_dust_removed_avi, fourcc, fps, (Lx, Ly), isColor=False)
    binary_frames = []

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Removing dust")
    while True:
        ret, frame = cap.read()  # １枚読み込み
        if ret == False:
            break  # 最後になったらループから抜ける

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 色の変換
        thresh, frame_binary = cv2.threshold(frame_gray, 127, 255, cv2.THRESH_BINARY)  # 二値化
        negative_frame_binary = cv2.bitwise_not(frame_binary)#ネガポジ反転
        black=np.zeros_like(frame_gray)
        contour,_=cv2.findContours(negative_frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contour:
            if cv2.contourArea(cont)>10000:
                cv2.fillPoly(black, [cont], (255,255,255))
        binary_frames.append(black)
        dust_removed_movie.write(black)
        pbar.update()
    pbar.close()
    cap.release()

    return binary_frames

def circle(X,cx,cy,r):
    return -np.sqrt(np.abs(r**2-(X-cx)**2))+cy

def getxy_RD(x, y, X, Y):
    _x, _y = (X-x), (Y-y)
    r = math.sqrt(_x**2+_y**2)
    rad = math.atan2(_y, _x)
    return r, rad

def get_circle(file_path,Lx,Ly):
    cap=cv2.VideoCapture(file_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    ret,img=cap.read()
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    surface_point=[]
    x=np.arange(0,Lx)
    y=np.arange(0,Ly)
    for x in x:
        if not np.all(img[...,x] == 255):
            y=np.min(np.where(img[...,x] != 255)[0])
            surface_point.append([x,y])
    surface_point=np.array(surface_point)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    X= surface_point[:,0]
    Y= surface_point[:,1]
    parms,_=curve_fit(circle,X,Y,p0=[int(Lx/2),int(Ly*1.1),int(Lx)])
    Y_fit=circle(X,*parms)

    print(parms)
    for i in range(0,len(X)):
        cv2.circle(img,(X[i],int(Y_fit[i])),3,(0,255,0),-1)

    cv2.imwrite(file_path.replace(".avi", "_circle.png"), img)

    return parms

def get_point_data(binary_frames,Lx:int,Ly:int,scale:float):
    X=[]
    Y=[]
    first_frame=binary_frames[0]
    for frame in tqdm(binary_frames,desc="get surface data"):
        x=[]
        y=[]
        for i in range(0, Lx):
            if not np.all(frame[...,i] == 0):
                hight=(len(np.nonzero(frame[...,i])[0])-len(np.nonzero(first_frame[...,i])[0]))*scale
                if hight<0:#成長していないところが光の加減でへこんだと判定されたときに0にする
                    hight=0
                x.append(i*scale)
                y.append(hight)
        X.append(x)
        Y.append(y)
    return X,Y

# メイン処理
if __name__ == "__main__":
    # time
    start_time = time.time()

    # set parameters

    # setting for video writer
    fourcc = cv2.VideoWriter.fourcc("M", "J", "P", "G")  # 動画保存時のfourcc設定、確認のためだけなので圧縮形式

    # Video Source
    # file_path="D:/master_thesis_data/experment_data/movie_data/movie_data/data_for_test/surface_growth/20241105/20241105_NOTWEEN_0.00mM_5V_02/20241105_NOTWEEN_0.00mM_5V_02.avi"
    file_path="D:/master_thesis_data/experment_data/movie_data/movie_data/data_for_test/surface_growth/20241105/20241105_TWEEN20_0.005mM_5V_01/20241105_TWEEN20_0.005mM_5V_01.avi"
    # file_path_avi=file_path.replace(".avi", "_average.avi")
    video = Video(file_path)
    video.show_info()
    cap = cv2.VideoCapture(file_path)
    # movie=cv2.VideoWriter(file_path_avi, fourcc, video.fps, (video.Lx, video.Ly))
    csv_file_path = file_path.replace(".avi", "_data.csv")
    binary_frames = dust_remove(file_path, video.fps, video.Lx, video.Ly)
    # cx,cy,r0 = get_circle(file_path, video.Lx, video.Ly)
    X,Y = get_point_data(binary_frames, video.Lx,video.Ly, video.scale)
    # mkdir
    dir_path = os.path.dirname(file_path)
    fourie_dir_path = dir_path + "/fourie_data"
    if not os.path.exists(fourie_dir_path):
        os.makedirs(fourie_dir_path)
        print("make directory: ", fourie_dir_path)
    else:
        shutil.rmtree(fourie_dir_path)
        os.makedirs(fourie_dir_path)
        print("remove and make directory: ", fourie_dir_path)

    # #DFT(離散フーリエ変換)
    # #set list
    # FFT_t=[]
    # Inv_FFT_t=[]
    # x_plot=[i*video.scale for i in range(0,video.Lx)]

    # #set parameter
    # N=video.Lx
    # L=N*video.scale #cm
    # dx=video.scale #cm
    # FFT_max=0
    # h_t_max=0
    
    # for frame_i in range(len(X)):
    #     if frame_i%10==0:
    #         Y_fft=np.fft.fft(Y[frame_i])/N
    #         Y_fft_shift=np.fft.fftshift(Y_fft)
    #         k=np.fft.fftfreq(N, d=dx)*2*np.pi
    #         shifted_k=np.fft.fftshift(k)
    #         Y_ifft=np.fft.ifft(Y_fft)*N
            
    #         FFT_t.append(np.abs(Y_fft_shift))
    #         Inv_FFT_t.append(Y_ifft)

    #         if np.max(np.abs(Y_fft_shift))>FFT_max:
    #             FFT_max=np.max(np.abs(Y_fft_shift))
    #         if np.max(np.abs(Y_ifft))>h_t_max:
    #             h_t_max=np.max(np.abs(Y_ifft))

    # for i in tqdm(range(1,len(FFT_t))):
    #     fig,ax=plt.subplots(1, 2, figsize=(12, 6))
    #     if np.where(FFT_t[i]<0)[0].size>0:
    #         print(np.where(FFT_t[i]<0)[0])
    #         sys.exit()
    #     ax[0].stem(shifted_k, np.abs(FFT_t[i]), label="FFT", markerfmt=' ')
    #     ax[0].set_ylim(1e-5,1e-1)
    #     ax[0].set_title("FFT,time={0:04d}".format(i*10)+" sec")
    #     ax[0].set_yscale("log")
    #     ax[0].set_xlabel("k "+r"cm$^{-1}$")
    #     x_min=2*np.pi/0.5
    #     x_max=2*np.pi/0.01
    #     ax[0].set_xlim(x_min,x_max)

    #     ax[1].plot(x_plot,np.abs(Inv_FFT_t[i]), label="IFFT")
    #     ax[1].set_ylim(0,h_t_max)
    #     ax[1].set_title("IFFT,time={0:04d}".format(i*10)+" sec")
    #     ax[1].set_xlabel("width cm")
    #     plt.savefig(fourie_dir_path + "/fourie_{0:04d}.png".format(i))
    #     plt.close()

    # save surface data
    
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Height_mean (cm)", "Std (cm)"])
        for frame_i in range(len(Y)):
            writer.writerow([frame_i*video.fps, np.mean(Y[frame_i]), np.std(Y[frame_i])])
