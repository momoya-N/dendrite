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
from matplotlib.ticker import ScalarFormatter

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

class range_value:
    def __init__(self,r_min=0.0,theta_center=0.0,r_L_max=0.0,r_R_max=0.0,theta_L_lim=0.0,theta_R_lim=0.0,theta_L_0=0.0,theta_R_0=0.0):
        self.r_min=r_min
        self.theta_center=theta_center
        self.r_L_max=r_L_max
        self.r_R_max=r_R_max
        self.theta_L_lim=theta_L_lim
        self.theta_R_lim=theta_R_lim
        self.theta_L_0=theta_L_0
        self.theta_R_0=theta_R_0
        self.theta_center=theta_center

def dust_remove(file_path,fps,Lx,Ly,scale):
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
        thresh, frame_binary = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#2値化
        negative_frame_binary = cv2.bitwise_not(frame_binary)#ネガポジ反転
        black=np.zeros_like(frame_gray)
        
        if cap.get(cv2.CAP_PROP_POS_FRAMES)==1:
            first_frame=black

        contour,_=cv2.findContours(negative_frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contour:
            if cv2.contourArea(cont)>10000:
                cv2.fillPoly(black, [cont], (255,255,255))

        #overhungの判定，overhungがある場合はreturn
        for x in range(0,Lx):
            y_init=first_frame[...,x]
            y=black[...,x]
            nonzero=np.where(y!=0)[0]
            nonzero_init=np.where(y_init!=0)[0]
            hight=len(nonzero)-len(nonzero_init)
            if Ly-len(nonzero)!=nonzero[0] and hight*scale>0.07:#初期からの成長高さが0.07cm=700umより大きく，かつoverhungがある場合
                return binary_frames
            
        binary_frames.append(black)
        dust_removed_movie.write(black)
        pbar.update()
    pbar.close()
    cap.release()

    return binary_frames

def circle(X,cx,cy,r):
    return -np.sqrt(np.abs(r**2-(X-cx)**2))+cy

def get_xy2RTh(x0, y0, X, Y):
    _x, _y = (X-x0), (Y-y0)
    r = math.sqrt(_x**2+_y**2)
    theta = math.atan2(_y, _x)
    return r, theta

def get_rth2XY(x0, y0, r, theta):
    X = x0 + r*math.cos(theta)
    Y = y0 + r*math.sin(theta)
    return X, Y

def get_circle(file_path,binary_frames,Lx,Ly):
    #estimating circle
    cap=cv2.VideoCapture(file_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    ret,img=cap.read()
    img_binary=binary_frames[0]
    surface_point=[]
    for x in range(0,Lx):
        if not np.all(img_binary[...,x] == 0):
            y=np.max(np.where(img_binary[...,x] ==0)[0])
            surface_point.append([x,y])
    surface_point=np.array(surface_point)

    X= surface_point[:,0]
    Y= surface_point[:,1]
    parms,_=curve_fit(circle,X,Y,p0=[int(Lx/2),int(Ly*1.1),int(Lx)])
    #極座標の代表点を求める
    cx,cy,r0=parms
    r_min=cy-Ly
    _ , theta_center=get_xy2RTh(cx,cy,cx,r_min)
    r_L_max,theta_L_0=get_xy2RTh(cx,cy,0,0)
    r_R_max,theta_R_0=get_xy2RTh(cx,cy,Lx,0)
    r_L_min,theta_L_lim=get_xy2RTh(cx,cy,0,Ly)
    r_R_min,theta_R_lim=get_xy2RTh(cx,cy,Lx,Ly)
    r_L_edge,theta_L_edge=get_xy2RTh(cx,cy,X[0],Y[0])
    r_R_edge,theta_R_edge=get_xy2RTh(cx,cy,X[-1],Y[-1])
    
    Y_fit=circle(X,*parms)
    
    for i in range(0,len(X)):
        cv2.circle(img,(X[i],int(Y_fit[i])),3,(0,255,0),-1)
    cv2.circle(img,(int(get_rth2XY(cx,cy,r_L_edge,theta_L_edge)[0]),int(get_rth2XY(cx,cy,r_L_edge,theta_L_edge)[1])),3,(0,0,255),-1)
    cv2.circle(img,(int(get_rth2XY(cx,cy,r_R_edge,theta_R_edge)[0]),int(get_rth2XY(cx,cy,r_R_edge,theta_R_edge)[1])),3,(0,0,255),-1)

    cv2.imwrite(file_path.replace(".avi", "_circle.png"), img)
    polor_range=range_value(r_min,theta_center,r_L_max,r_R_max,theta_L_lim,theta_R_lim,theta_L_0,theta_R_0)

    return cx,cy,r0,polor_range

def correlation_func(Y):
    Y=np.array(Y)-np.mean(Y)
    N=len(Y)
    cor=np.zeros(N)
    for i in range(int(N/2)):
        cor[i]=np.sum(Y[:N-i]*Y[i:])/np.sum(Y[:N-i]**2)
    return cor

def decart_img2polar_img_analisys_plot(file_path,video,cx,cy,r0,polor_range,time_range):#動画の座標変換
    #set parameter
    Lx=video.Lx
    Ly=video.Ly
    scale=video.scale
    theta_min=polor_range.theta_L_lim
    theta_max=polor_range.theta_R_lim
    r_min=polor_range.r_min
    r_max=max(polor_range.r_L_max,polor_range.r_R_max)
    Theta_lim=np.linspace(theta_min,theta_max,Lx)
    Theta=[theta for theta in Theta_lim if theta>polor_range.theta_L_0 and theta<polor_range.theta_R_0]
    delta_theta=Theta-min(Theta)#角度の差分,左端を0にする
    R=np.linspace(r_min,r_max,Ly)
    r_bottom=0.0
    h_mean=[]#界面高さの平均
    h_std=[]#界面高さの標準偏差
    cap=cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter.fourcc("M", "J", "P", "G")  # 動画保存時のfourcc設定、確認のためだけなので圧縮形式
    file_path_polar_avi = file_path.replace(".avi", "_polar.avi")
    polar_movie = cv2.VideoWriter(file_path_polar_avi, fourcc, video.fps, (len(Theta), Ly), isColor=False)
    file_path_binary_avi = file_path.replace(".avi", "_binary.avi")
    binary_polar_movie = cv2.VideoWriter(file_path_binary_avi, fourcc, video.fps, (len(Theta), Ly), isColor=False)
    dir_path = os.path.dirname(file_path)
    corr_dir_path = dir_path + "/correlation_data"

    #make directory
    if not os.path.exists(corr_dir_path):
        os.makedirs(corr_dir_path)
        print("make directory: ", corr_dir_path)
    else:
        shutil.rmtree(corr_dir_path)
        os.makedirs(corr_dir_path)
        print("remove and make directory: ", corr_dir_path)

    #start analisis
    pbar = tqdm(total=time_range-1, desc="Polor mapping")
    while True:
        ret,img=cap.read()
        n_frame=int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if n_frame==time_range:
            break
        if not n_frame==51:
            pbar.update()
            continue

        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        black=np.zeros((Ly,len(Theta)),dtype=np.uint8)
        black_surface=np.zeros((Ly,len(Theta)),dtype=np.uint8)
        for i,theta in enumerate(Theta):
            for j,r in enumerate(reversed(R)):
                x,y=get_rth2XY(cx,cy,r,theta)
                if x>0 and x<Lx and y>0 and y<Ly:
                    black[j,i]=img_gray[int(y),int(x)]
        polar_movie.write(black)
        cv2.imwrite(file_path.replace(".avi", "_polar_tmp.png"), black)
        _,img_binary=cv2.threshold(black, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_negative_binary=cv2.bitwise_not(img_binary)
        contoures ,_=cv2.findContours(img_negative_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contoures:
            if cv2.contourArea(cont)>40000:
                cv2.fillPoly(black_surface, [cont], (255,255,255))
        cv2.imwrite(file_path.replace(".avi", "_polar_tmp_fill.png"), black_surface)
        binary_polar_movie.write(black_surface)
        #界面高さトラッキング
        h=[]#界面高さ
        for i,theta in enumerate(Theta):
            R_rev=np.flip(R)
            top=np.nonzero(black_surface[...,i])[0][0]
            hight=R_rev[top]
            h.append(hight*scale)
        #相関関数の計算
        if r_bottom==0.0:
            r_bottom=min(h)
        h=np.array(h)-r_bottom
        cor_func=correlation_func(h)
        diff_cor_func=np.diff(cor_func)
        local_max=np.where((diff_cor_func[:-1]>0)&(diff_cor_func[1:]<0))[0]+1
        
        #画像出力
        # if n_frame==True:
        fig,ax=plt.subplots(1, 2, figsize=(13.5, 9))
        h_from_r0=np.array(h)+r0*scale
        ax[0].plot(Theta,h_from_r0)
        ax[0].set_title("Height Tracking "+r"$t=$"+str(n_frame/video.fps)+" sec")
        ax[0].set_xlabel(r"$\theta$ rad")
        ax[0].set_ylabel("Height from center "+r"$r$ cm")
        ax[0].set_ylim(r0*scale,(r0+50)*scale)
        ax[0].axhline(y=np.mean(h_from_r0),color='r',ls='--',lw=0.6)
        ax[0].text(Theta[0], np.mean(h_from_r0), rf"{np.mean(h_from_r0):.2f} cm", color='red', ha='center')
        
        ax[1].plot(delta_theta[:int(len(cor_func)/2)],cor_func[:int(len(cor_func)/2)])
        ax[1].set_title("Correlation Function "+r"$t=$"+str(n_frame/video.fps)+" sec",pad=30)
        # ax[1].set_xlabel(r"$\delta \theta$ rad")
        # ax[1].set_ylabel("Correlation")
        y_lim_min=0.5
        y_lim_max=1
        ax[1].set_ylim(y_lim_min,y_lim_max)
        ax[1].set_xlim(0,0.03)
        ax[1].scatter(delta_theta[local_max[:3]],cor_func[local_max[:3]],c="r")
        ax[1].tick_params(axis='x', labelsize=18)
        ax[1].tick_params(axis='y', labelsize=18)
        #指数表記
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)  # 科学的表記を有効にする
        formatter.set_powerlimits((-1, 1))
        ax[1].xaxis.set_major_formatter(formatter)
        # オフセットテキストを取得してフォントサイズを設定
        offset_text = ax[1].xaxis.get_offset_text()
        offset_text.set_fontsize(18)  # フォントサイズ設定

        
        for i,x0 in enumerate(delta_theta[local_max[:3]]):
            y0=cor_func[local_max[i]]
            y_max=float((y0-y_lim_min)/(y_lim_max-y_lim_min))
            y_min=float(i+1)/20
            ax[1].axvline(x=x0,c='r',ls='--',lw=0.6,ymax=y_max,ymin=y_min)
            # exp=int(np.log10(x0))
            exp=-2
            mantissa=x0/10**exp
            # ax[1].text(x0,0.5+y_min*0.5,rf"${mantissa:.2f} \times 10^{{\mathrm{{{exp}}}}}$", color='k', ha='left', fontsize=18)
            ax[1].text(x0,y_lim_min+y_min*(y_lim_max-y_lim_min),rf"${mantissa:.2f}$", color='k', ha='left', fontsize=18)
        plt.savefig(corr_dir_path + "/correlation_{0:04d}.png".format(n_frame))
        plt.close()

        h_mean.append(np.mean(h))
        h_std.append(np.std(h))
        pbar.update()

    #data output
    csv_file_path = file_path.replace(".avi", "_polar_data.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Height_mean (cm)", "Height_Std (cm)"])
        for frame_i in range(len(h_mean)):
            writer.writerow([frame_i*video.fps, h_mean[frame_i], h_std[frame_i]])
    cap.release()
    return None

# メイン処理
if __name__ == "__main__":
    # time
    start_time = time.time()

    # setting for video writer
    fourcc = cv2.VideoWriter.fourcc("M", "J", "P", "G")  # 動画保存時のfourcc設定、確認のためだけなので圧縮形式

    # Video Source
    file_path="D:/master_thesis_data/experment_data/movie_data/movie_data/data_for_test/surface_growth/20241105/20241105_NOTWEEN_0.00mM_5V_02/20241105_NOTWEEN_0.00mM_5V_02.avi"
    # file_path="D:/master_thesis_data/experment_data/movie_data/movie_data/data_for_test/surface_growth/20241105/20241105_TWEEN20_0.005mM_5V_01/20241105_TWEEN20_0.005mM_5V_01.avi"
    video = Video(file_path)
    video.show_info()
    cap = cv2.VideoCapture(file_path)
    binary_frames = dust_remove(file_path, video.fps, video.Lx, video.Ly, video.scale)#ここでbinary_frameを作って後で極座標に変換しているので２度手間感あるが，一旦このままにしておく(Done is better than perfect.)
    cx,cy,r0,polor_range = get_circle(file_path,binary_frames, video.Lx, video.Ly)
    decart_img2polar_img_analisys_plot(file_path, video, cx, cy, r0, polor_range, time_range=len(binary_frames))