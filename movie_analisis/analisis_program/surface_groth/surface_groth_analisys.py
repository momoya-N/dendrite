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

# メイン処理
if __name__ == "__main__":
    # time
    start_time = time.time()

    # set parameters

    # setting for video writer
    fourcc = cv2.VideoWriter.fourcc("M", "J", "P", "G")  # 動画保存時のfourcc設定、確認のためだけなので圧縮形式

    # Video Source
    # file_path = "D:/master_thesis_data/experment_data/movie_data/movie_data/data_for_test/surface_growth/20241010/surface_groth_test_5V_TWEEN20_0.005mM_20241010/surface_groth_test_5V_TWEEN20_0.005mM_20241010_edited.avi"
    # file_path="D:/master_thesis_data/experment_data/movie_data/movie_data/data_for_test/surface_growth/20241010/surface_groth_test_5V_0.00mM_20241010/surface_groth_test_5V_0.00mM_20241010_edited.avi"
    # file_path="D:/master_thesis_data/experment_data/movie_data/movie_data/data_for_test/surface_growth/20241101/20241101_TWEEN20_0.005mM_5V_02_edited.avi"
    file_path="D:/master_thesis_data/experment_data/movie_data/movie_data/data_for_test/surface_growth/20241105/20241105_NOTWEEN_0.00mM_5V_02_edited.avi"
    # file_path="D:/master_thesis_data/experment_data/movie_data/movie_data/data_for_test/surface_growth/20241105/20241105_TWEEN20_0.005mM_5V_02_edited.avi"
    file_path_avi=file_path.replace(".avi", "_average.avi")
    video = Video(file_path)
    video.show_info()
    cap = cv2.VideoCapture(file_path)
    movie=cv2.VideoWriter(file_path_avi, fourcc, video.fps, (video.Lx, video.Ly))
    csv_file_path = file_path.replace(".avi", "_data.csv")

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
    
    #set list
    X=[video.scale*i for i in range(video.Lx)]
    h_t_pix=[]
    h_t_cm=[]
    h_t_cm_plot=[]
    h_std=[]
    FFT_t=[]
    Inv_FFT_t=[]
    FFT_t_data_csv=pd.DataFrame()

    #set parameter
    N=video.Lx
    L=N*video.scale
    dx=video.scale
    FFT_max=0
    h_t_max=0

    pbar = tqdm(total=video.total_frames, desc="Surface Growth Analysis")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h_t_x_pix=[]
        h_t_x_cm=[]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        theresh,frame_binary=cv2.threshold(frame_gray, 127, 255, cv2.THRESH_BINARY)

        for i in range(0, video.Lx):
            for j in range(0, video.Ly):
                if frame_binary[j,i]==0:
                    h_t_x_pix.append(j) #映像上部からの距離がｊ
                    h_t_x_cm.append((video.Ly-j)*video.scale)#影像下部からの距離(cm)
                    cv2.circle(frame, (i, j), 1, (0, 0, 255), -1)
                    break
        t=int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if t%10==0:
            #FFT&IFFT
            Y_fft=np.fft.fft(np.array(h_t_x_cm)-np.mean(h_t_x_cm))/N
            Y_fft_shift=np.fft.fftshift(Y_fft)
            Y_ifft=np.fft.ifft(Y_fft)*N+np.mean(h_t_x_cm)
            k=np.fft.fftfreq(N, d=dx)*2*np.pi
            shifted_k=np.fft.fftshift(k)
            Y_fft_shift=Y_fft_shift[np.abs(shifted_k)<300]
            shifted_k=shifted_k[np.abs(shifted_k)<300]
            

            FFT_t.append(np.abs(Y_fft_shift))
            Inv_FFT_t.append(Y_ifft)
            h_t_cm_plot.append(h_t_x_cm)

            FFT_t_data_csv = pd.concat([FFT_t_data_csv, pd.DataFrame(np.abs(Y_fft_shift)).T], axis=0)

            if np.max(np.abs(Y_fft_shift))>FFT_max:
                FFT_max=np.max(np.abs(Y_fft_shift))
            if np.max(h_t_x_cm)>h_t_max:
                h_t_max=np.max(h_t_x_cm)
        print(len(k))
        #save data
        h_t_cm.append(np.mean(h_t_x_cm))
        sigma=np.std(h_t_x_cm)
        h_std.append(sigma)
        cv2.line(frame, (0, int(np.mean(h_t_x_pix))), (video.Lx, int(np.mean(h_t_x_pix))), (0, 255, 0), 1)
        movie.write(frame)
        pbar.update()
    pbar.close()
    
    for i in tqdm(range(len(FFT_t))):
        fig,ax=plt.subplots(1, 2, figsize=(12, 6))
        ax[0].stem(shifted_k, np.abs(FFT_t[i]), label="FFT", markerfmt=' ')
        # ax[0].set_xlim(-500, 500)
        ax[0].set_ylim(0, FFT_max)
        ax[0].set_title("FFT,time={0:04d}".format(i*10)+" sec")

        ax[1].plot(np.abs(Inv_FFT_t[i]), label="IFFT")
        ax[1].plot(h_t_cm_plot[i], label="h_t_cm")
        ax[1].set_xlim(0, N)
        ax[1].set_ylim(0, h_t_max)
        ax[1].set_title("IFFT,time={0:04d}".format(i*10)+" sec")
        plt.savefig(fourie_dir_path + "/fourie_{0:04d}.png".format(i))
        plt.close()

    # save surface data
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Height_mean (cm)", "Std (cm)"])
        for frame_i in range(len(h_t_cm)):
            writer.writerow([frame_i*video.fps, h_t_cm[frame_i], h_std[frame_i]])

    # save FFT data
    # FFT_t_data_csv.to_csv(fourie_dir_path + "/FFT_data.csv", index=False)
    

    # Release resources
    cap.release()
    movie.release()
    cv2.destroyAllWindows()

    # Print elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")