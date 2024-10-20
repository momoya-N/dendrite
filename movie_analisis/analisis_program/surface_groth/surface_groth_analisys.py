import sys
import os
import cv2
import numpy as np
import time
from tqdm import tqdm
import csv



class Video:
    def __init__(self, file_path_avi: str):
        cap = cv2.VideoCapture(file_path_avi)
        fname = os.path.basename(file_path_avi)
        ret, frame = cap.read()

        self.bool = cap.isOpened()
        self.path = file_path_avi
        self.fname = fname
        self.scale=1/362 #cm/pix
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
    dir_path = "D:/master_thesis_data/experment_data/movie_data/movie_data/data_for_test/surface_growth/20241010/surface_groth_test_5V_TWEEN20_0.005mM_20241010/surface_groth_test_5V_TWEEN20_0.005mM_20241010_edited.avi"
    # dir_path="D:/master_thesis_data/experment_data/movie_data/movie_data/data_for_test/surface_growth/20241010/surface_groth_test_5V_0.00mM_20241010/surface_groth_test_5V_0.00mM_20241010_edited.avi"
    dir_path_avi=dir_path.replace(".avi", "_average.avi")
    video = Video(dir_path)
    video.show_info()
    cap = cv2.VideoCapture(dir_path)
    movie=cv2.VideoWriter(dir_path_avi, fourcc, video.fps, (video.Lx, video.Ly))
    #set list
    h_t_pix=[]
    h_t_cm=[]
    h_scatter=[]
    FFT_t=[]
    inv_FFT_t=[]

    #set parameter
    N=video.Lx
    L=N*video.scale
    dx=video.scale
    X=np.linspace(0,L,N)

    pbar = tqdm(total=video.total_frames)
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
        #FFT&IFFT
        Y_fft=np.fft.fft(h_t_x_cm)/(N/2)
        Y_IFFT=np.fft.ifft(Y_fft)*(N/2)
        k=np.fft.fftfreq(N, d=dx)*2*np.pi
        shifted_k=np.fft.fftshift(k)

        
        h_t_cm.append(np.mean(h_t_x_cm))
        deviation_2=[(h_t_x_cm[i]-np.mean(h_t_x_cm))**2 for i in range(len(h_t_x_cm))]
        h_scatter.append(np.sqrt(np.sum(deviation_2)))
        cv2.line(frame, (0, int(np.mean(h_t_x_pix))), (video.Lx, int(np.mean(h_t_x_pix))), (0, 255, 0), 1)
        movie.write(frame)
        pbar.update()
    pbar.close()
    
    # Save data to CSV

    csv_file_path = dir_path.replace(".avi", "_data.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Height_mean (cm)", "Scatter"])
        for frame_i in range(len(h_t_cm)):
            writer.writerow([frame_i*video.fps, h_t_cm[frame_i], h_scatter[frame_i]])

    # Release resources
    cap.release()
    movie.release()
    cv2.destroyAllWindows()

    # Print elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")