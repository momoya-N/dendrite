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
import threading

class Video:
    def __init__(self, file_path_avi: str):
        cap = cv2.VideoCapture(file_path_avi)
        fname = os.path.basename(file_path_avi)
        ret, frame = cap.read()

        self.bool = cap.isOpened()
        self.path = file_path_avi
        self.fname = fname
        self.scale=5000/((2372.03+2368.01+2368.05)/3) #μm/pix
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
        print("Scale : ", self.scale, " μm/pix")
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

def dust_removed(frame):#チリの除去
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray scale に変換
    thresh, frame_binary = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#Otsuの二値化
    negative_frame_binary = cv2.bitwise_not(frame_binary)#ネガポジ反転
    black=np.zeros_like(frame_gray)

    contour,_=cv2.findContours(negative_frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contour:
        if cv2.contourArea(cont)>4000*500:#画面に対して1/6以上の領域を解析領域とする
            cv2.fillPoly(black, [cont], (255,255,255))

    return black

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

def get_circle(img,Lx,Ly):#最初の画像の極板の円の推定
    #estimating circle
    img_binary=dust_removed(img)
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
        cv2.circle(img,(X[i],int(Y_fit[i])),4,(0,255,0),-1)
    cv2.circle(img,(int(get_rth2XY(cx,cy,r_L_edge,theta_L_edge)[0]),int(get_rth2XY(cx,cy,r_L_edge,theta_L_edge)[1])),20,(0,0,255),-1)
    cv2.circle(img,(int(get_rth2XY(cx,cy,r_R_edge,theta_R_edge)[0]),int(get_rth2XY(cx,cy,r_R_edge,theta_R_edge)[1])),20,(0,0,255),-1)

    cv2.imwrite(file_path.replace(".avi", "_circle.png"), img)
    polar_range=range_value(r_min,theta_center,r_L_max,r_R_max,theta_L_lim,theta_R_lim,theta_L_0,theta_R_0)

    return cx,cy,r0,polar_range

def correlation_func(Y):
    Y=np.array(Y)-np.mean(Y)
    N=len(Y)
    cor=np.zeros(N//2)
    for i in range(N//2):
        cor[i]=np.sum(Y[:N-i]*Y[i:])/np.sum(Y[:N-i]**2)
    return cor

def decart_img2polar_img(img,video,cx,cy,polar_range):#動画の座標変換
    # set parameter
    Lx = video.Lx
    Ly = video.Ly
    theta_min = polar_range.theta_L_0
    theta_max = polar_range.theta_R_0
    r_min = polar_range.r_min
    r_max = max(polar_range.r_L_max, polar_range.r_R_max)
    black=np.zeros((Ly,Lx),dtype=np.uint8)

    # Create Theta and R arrays
    Theta = np.linspace(theta_min, theta_max, Lx)
    R = np.linspace(r_min, r_max, Ly)

    # Create meshgrid for Theta and R
    R_grid, Theta_grid = np.meshgrid(R, Theta, indexing='ij')

    # Convert polar coordinates (R, Theta) to Cartesian coordinates (x, y)
    X = cx + R_grid * np.cos(Theta_grid)
    Y = cy + R_grid * np.sin(Theta_grid)

    # Clip X and Y to valid image range
    X_clipped = np.clip(X, 0, Lx - 1).astype(int)
    Y_clipped = np.clip(Y, 0, Ly - 1).astype(int)

    # Create polar image
    polar_img = img[Y_clipped, X_clipped]
    polar_img=np.flipud(polar_img)

    return polar_img

def surface_tracking(img,r_bottom,video,polar_range):#界面高さトラッキング
    Lx=video.Lx
    Ly=video.Ly
    scale=video.scale
    theta_min=polar_range.theta_L_lim
    theta_max=polar_range.theta_R_lim
    r_min=polar_range.r_min
    r_max=max(polar_range.r_L_max,polar_range.r_R_max)
    Theta_lim=np.linspace(theta_min,theta_max,Lx)
    Theta=[theta for theta in Theta_lim if theta>polar_range.theta_L_0 and theta<polar_range.theta_R_0]
    R=np.linspace(r_min,r_max,Ly)
    h_t=[] #その時間の界面高さ
    for i,theta in enumerate(Theta):
        R_rev=np.flip(R)
        top=np.nonzero(img[...,i])[0][0]
        hight=R_rev[top]
        h_t.append(hight*scale)
    if r_bottom==0.0:
        r_bottom=min(h_t)
    #相関関数の計算
    h_t=np.array(h_t)-r_bottom
    cor_func=correlation_func(h_t)
    diff_cor_func=np.diff(cor_func)
    local_max=np.where((diff_cor_func[:-1]>0)&(diff_cor_func[1:]<0))[0]+1
    return Theta,h_t,cor_func,local_max,r_bottom

def surface_analisis(file_path,video):#動画の解析&データ出力
    surface_hight_data=pd.DataFrame()
    cor_func_data=pd.DataFrame()
    local_max_data=pd.DataFrame()
    cap=cv2.VideoCapture(file_path)
    # pbar = tqdm(total=video.total_frames, desc="Analyzing video")
    r_bottom=0.0

    while True:
        ret,img=cap.read()

        if not ret:
            break

        n_frame=int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if n_frame==1:
            cx,cy,r0,polar_range = get_circle(img, video.Lx, video.Ly)
    
        binary_img=dust_removed(img)
        polor_binary_img=decart_img2polar_img(binary_img,video,cx,cy,polar_range)
        Theta,h_t,cor_func,local_max,r_bottom=surface_tracking(polor_binary_img,r_bottom,video,polar_range)
        delta_theta=Theta-min(Theta)#相関関数用の角度の差
        if n_frame==1:#DateFrameの1行目に角度の行を追加
            cor_func_data=pd.concat([cor_func_data,pd.DataFrame({"delta_theta":list(delta_theta[:len(delta_theta)//2])})],axis=1)
            surface_hight_data=pd.concat([surface_hight_data,pd.DataFrame({"Theta":list(Theta)})],axis=1)
            # local_max_data=pd.concat([local_max_data,pd.DataFrame({"local_max":list(Theta)})],axis=1)#うまく取れていないので無し
        cor_func_data=pd.concat([cor_func_data,pd.DataFrame({str(n_frame/video.fps):list(cor_func)})],axis=1)
        surface_hight_data=pd.concat([surface_hight_data,pd.DataFrame({str(n_frame/video.fps):list(h_t)})],axis=1)
        local_max_data=pd.concat([local_max_data,pd.DataFrame({str(n_frame/video.fps):list(local_max[:3])})],axis=1)
        # pbar.update()

    # pbar.close()

    return surface_hight_data,cor_func_data,local_max_data

def worker(file_path):
    # file_path=file_pluronic+con+"/"+file+"/"+file+".avi"
    # video = Video(file_path)
    # video.show_info()
    # print("Start surface groth analisis")
    surface_hight_data,cor_func_data,local_max_data=surface_analisis(file_path,video)
    #save data
    surface_hight_data.to_csv(file_path.replace(".avi", "_surface_hight_data.csv"),float_format='%.5f')
    cor_func_data.to_csv(file_path.replace(".avi", "_cor_func_data.csv"),float_format='%.5f')
    local_max_data.to_csv(file_path.replace(".avi", "_local_max_data.csv"),float_format='%.5f')

# メイン処理
if __name__ == "__main__":
    file_pluronic="D:/master_thesis_data/experiment_data/movie_data/movie_data/data_for_surface_groth/Pluronic-F127/"
    con_list=os.listdir(file_pluronic)
    
    for con in con_list:
        start=time.time()
        file_list=os.listdir(file_pluronic+con)
        threads = []
        for i in range(len(file_list)):# Create 3 threads,mulitple thread processing
            file_path=file_pluronic+con+"/"+file_list[i]+"/"+file_list[i]+".avi"
            video = Video(file_path)
            video.show_info()
            print("Start surface groth analisis\n")
            t = threading.Thread(target=worker, args=(file_path,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
            
        # for file in file_list:
        #     file_path=file_pluronic+con+"/"+file+"/"+file+".avi"
        #     video = Video(file_path)
        #     video.show_info()
        #     print("Start surface groth analisis")
        #     surface_hight_data,cor_func_data,local_max_data=surface_analisis(file_path,video)
        #     #save data
        #     surface_hight_data.to_csv(file_path.replace(".avi", "_surface_hight_data.csv"),float_format='%.5f')
        #     cor_func_data.to_csv(file_path.replace(".avi", "_cor_func_data.csv"),float_format='%.5f')
        #     local_max_data.to_csv(file_path.replace(".avi", "_local_max_data.csv"),float_format='%.5f')
        print(f"{con} analyses are done. Time:{time.time()-start:.2f} s\n")