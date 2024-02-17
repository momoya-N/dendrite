# import numpy as np
# import math
# from matplotlib import pyplot as plt
# import cv2
# from PIL import Image

# # Bernstein多項式を計算する関数
# def bernstein(n, t):
#     B = []
#     for k in range(n + 1):
#         # 二項係数を計算してからBernstein多項式を計算
#         nCk = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
#         B.append(nCk * t ** k * (1 - t) ** (n - k))
#     return B

# # ベジェ曲線を描く関数
# def bezie_curve(Q):
#     n = len(Q) - 1
#     dt = 0.001
#     t = np.arange(0, 1 + dt, dt)
#     B = bernstein(n, t)
#     px = 0
#     py = 0
#     for i in range(len(Q)):
#         px += np.dot(B[i], Q[i][0])
#         py += np.dot(B[i], Q[i][1])
#     return px, py

# #液滴の底面算出
# def bottom_line(q1,q4,px):
#     delta=q4-q1
#     alpha=delta[1]/delta[0]

#     y_botom=alpha*(px-q1[0])+q1[1]

#     return y_botom

# #ベジエ曲線の面積の厳密解
# def bezie_area_exact(Q):
#     S_bottom=0.5*(Q[3][0]-Q[0][0])*(Q[0][1]+Q[3][1])
#     S_bezie=0.05*(-Q[3][1]*(Q[0][0]+3*Q[1][0]+6*Q[2][0]-10*Q[3][0])-3*Q[2][1]*(Q[0][0]+Q[1][0]-2*Q[3][0])+3*Q[1][1]*(-2*Q[0][0]+Q[2][0]+Q[3][0])+Q[0][1]*(-10*Q[0][0]+6*Q[1][0]+3*Q[2][0]+Q[3][0]))

#     S_exact=S_bottom-S_bezie

#     return S_exact

# #色のヒストグラム作成
# def color_hist(filename):
#     img = np.asarray(Image.open(filename).convert("L")).reshape(-1,1)
#     plt.hist(img, bins=128)
#     plt.show()

# #hyperbolic tangent での境界表現
# def surface(epsilnon,R,r): #epsilon:境界の幅(2*epsilon),R:境界の半径,r:動径半径
#     f=0.5*(1+math.tanh((R-r)/epsilnon))
    
#     return f

import math
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.collections as mc
import matplotlib.cm as cm
import time

def N_Frame_Image(frameIndex):  # N番目のフレーム画像を返す
    # インデックスがフレームの範囲内なら…
    if 0<= frameIndex < Total_Frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        ret, image = cap.read() #ret:bool値(画像が読めれば True) image:画像のnbarray
        return image
    else:
        return None

def CM(n): #n:n_0,calculate the center of mass
    cap.set(cv2.CAP_PROP_POS_FRAMES,n)
    ret,image_cm=cap.read()
    gray_cm=cv2.cvtColor(image_cm,cv2.COLOR_BGR2GRAY)
    threshold,binary_cm=cv2.threshold(gray_cm,cut,255,cv2.THRESH_BINARY)

    binary_cm=Remove_Dust(binary_cm)
    
    m=cv2.moments(binary_cm,True) #bool値はbinary画像かどうか
    #重心の計算、四捨五入
    x,y=round(m['m10']/m['m00']) , round(m['m01']/m['m00'])
    
    return x,y

def First_Frame():  # ピクセル値を持つfirst frameを計算
    n = 0
    while 1:
        # making binary array
        gray = cv2.cvtColor(N_Frame_Image(n), cv2.COLOR_BGR2GRAY)
        threshold,binary=cv2.threshold(gray,cut,255,cv2.THRESH_BINARY)#完全に二値化
        binary=Remove_Dust(binary)
        
        if np.count_nonzero(binary) > 0:
            break
        n += 1
    return n

def Correlation_Function(k,power_specturm): #波数空間での相関関数
    d_theta=2*math.asin(0.5/k)
    theta=0
    rho_k=0
    
    if k>=max(Lx/2,Ly/2):
        return 

    while theta<2*math.pi:
        k_x=k*math.cos(theta)
        k_y=k*math.sin(theta)
        rho_k += power_specturm[int(Ly/2+k_y),int(Lx/2+k_x)]
        theta += d_theta
        
    return rho_k/(2*math.pi)

def Remove_Dust(Matrix):#remove dust funvtion
    for i in range(Ly):
        for j in range(Lx):
            if Matrix[i,j]==255:
                c_temp = 0
                for k in range(5):  # 1粒子あたり周囲25マスの探索
                    for l in range(5):
                        if (2 <= (j - 2 + l) < Lx - 2) and (2 <= (i - 2 + k) < Ly - 2)and (Matrix[i - 2 + k, j - 2 + l]==255) :
                            c_temp += 1
                if c_temp <= dust:  # 周囲25マスの粒子数が(dust)個より少なければゴミと判定
                    Matrix[i,j]=0
    return Matrix

def search_branch(binary,r,x_c,y_c):  # binsry data,radius r, cm_x,cm_y
    dth = 1 / r  # delta theta,十分大きいrではokそう？→f(x)=2arcsin(1/2x)-1/xは、x=2で0.005360...
    t = 0  # 角度のステップ数
    th = 0  # 角度
    branch_th=[]

    r_x = int(r * math.cos(0) + x_c)
    r_y = int(r * math.sin(0) + y_c)

    while th < 2 * math.pi:
        th = dth * t
        r_x = int(r * math.cos(th) + x_c)
        r_y = int(r * math.sin(th) + y_c)

        th_tmp=[]

        if (1 <= r_x < Lx-1) and (1 <= r_y < Ly-1):
            while binary[r_y, r_x] == 255:#枝の重心計測
                th_tmp.append(th)
                t += 1
                th = dth * t
                
                if th > 2 * math.pi:
                    break
                
                r_x = int(r * math.cos(th) + x_c)  # dth分回転させる
                r_y = int(r * math.sin(th) + y_c)

        if np.size(th_tmp)!=0:
            th=np.average(th_tmp)
            branch_th.append(th)

        t += 1

    return branch_th

def next_point(position_now,position_next_list):#position[i][j],position[i+1]
    rnow=position_now[0]
    thnow=position_now[1]
    rnext=position_next_list[0][0]
    dist_tmp2=pow(r_max,2) #十分大きな数で
    for i in range(len(position_next_list)):
        thnext=position_next_list[i][1]
        dist2=pow(rnow,2)+pow(rnext,2)-2*rnow*rnext*math.cos(thnow-thnext)
        if dist_tmp2 > dist2:
            dist_tmp2=dist2
            index=i

    return index

def tree_search(i,j):
    if i < len(search_range)-1:
        i_tmp=i+1
        j_tmp=next_point(position[i][j],position[i_tmp])
        position[i][j][3]=position[i_tmp][j_tmp][2] #[i][j] in <- [i_tmp][j_tmp] position_id
        if position[i_tmp][j_tmp][4]!=0:
            position[i_tmp][j_tmp][5]=1 #node_flag=True
            return
        else:
            position[i_tmp][j_tmp][4]=position[i][j][2] #[i_tmp][j_tmp] out <- [i][j] position_id
        
        tree_search(i_tmp,j_tmp)
        
#data[i][j]=[radius,theta,position_id,in,out,node_flag,branch_id1,branch_id2,branch_id3]
def branch_search(i,j):
    if i < len(search_range)-1:
        i_tmp=i+1
        j_tmp=next_point(position[i][j],position[i_tmp])
        position[i][j][8]=branch_id[0]
        if position[i][j][5] == 1:
            # print("OK",branch_id[0])
            # position[i][j][6]=branch_id[0]
            del position[i][j][6]
            branch_id[0]+=1
            position[i][j].append(branch_id[0]) #node_flag=True
            
        # print(branch_id[0])
        branch_search(i_tmp,j_tmp)
        

#Main

#time
start=time.time()

#constants
dust = 4  # チリの大きさ判定用変数
cut = 30  # threshold value,輝度値は0が黒色、255が白色。
# K = 15  # distance of pick up frame,2 s/frame ->30 s毎に取得

#Video Source
Dir_name="/mnt/c/Users/PC/Desktop/"
f_name="20230205_nonsur_77.2mN_No.1.avi"
f_name2="20230222_0.05sur_73.2mN_No.3.avi"
f_name3="20230221_nonsur_76.8mN_No.1.avi"

file_path=Dir_name + f_name3
name_tag=file_path.replace(Dir_name,"")
name_tag=name_tag.replace(".avi","")
window_name = file_path[len(Dir_name):]
cap = cv2.VideoCapture(file_path)
print(window_name)
print(type(cap))

Lx=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Ly=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS=cap.get(cv2.CAP_PROP_FPS)
print(cap.isOpened())
Total_Frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("File Name : ",window_name)
print("Frame Width : ", Lx)
print("Frame Hight : ", Ly)
print("FPS : ", FPS)
print("Frame Count : ", Total_Frames)

n0 = First_Frame()  # origin frame number, time=0

#Last Frame
cap.set(cv2.CAP_PROP_POS_FRAMES,Total_Frames-1)
ret,image=cap.read()

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#輝度値情報のみの画像
threshold,binary=cv2.threshold(gray,cut,255,cv2.THRESH_BINARY)#完全に二値化(RGB -> 輝度値算出 ->二値化)
threshold,nongray_binary=cv2.threshold(image,cut,255,cv2.THRESH_BINARY)#RGBを残したまま二値化

binary=Remove_Dust(binary)

#位置データの作成
x,y=CM(n0) #重心計算
r_max=max(x,(Lx-x),y,(Ly-y))#最大半径＝重心からの距離の最大値
search_range=[r for r in reversed(range(2,r_max,int(r_max/40)))]
position=[[]for i in range(len(search_range))]
position_id=1
branch_id=[2] #Pythonには参照渡しは存在しない(objectのaddressを値渡しする)ため、関数内ではコピーして渡されたaddressにアクセスしてobjectを書き換えることで参照渡しのような挙動を実現するらしい。

for i , r in enumerate(search_range):
    branch_th=search_branch(binary,r,x,y)
    for j in range(len(branch_th)):
        position[i].append([r,2*math.pi-branch_th[j],position_id,0,0,0,0,0,0])#data[i][j]=[radius,theta,position_id,in,out,node_flag,branch_id1,branch_id2,branch_id3]
        position_id+=1

#接続の計算
for i in range(len(search_range)):
    for j in range(len(position[i])):
        if position[i][j][4]==0: #最外端の判定
            position[i][j][4]=-1
            tree_search(i,j)

for i in range(len(search_range)):
    for j in range(len(position[i])):
        if position[i][j][4]==-1: #最外端の判定
            branch_search(i,j)

print(position)

#枝のベクトル計算
vector=[]
for i in range(0,len(position)-1):
    for j in range(len(position[i])):
        thnow=position[i][j][1]
        rnow=position[i][0][0]
        thnext=position[i+1][next_point(position[i][j],position[i+1])][1]
        rnext=position[i+1][0][0]
        vector.append([[thnow,rnow],[thnext,rnext]])

vector_tmp=[]

node=[]
edge=[]

# color=np.linspace(0,brance_id)
# print(position,brance_id)

# 画像
img_origin=image #original image
img_n=N_Frame_Image(n0) #N frames image
scalebar=ScaleBar(11/681,"cm",length_fraction=0.5,location="lower right")

# 画像として可視化する
fig = plt.figure(figsize=(18,9))
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2,projection="polar")

# #元画像(gray)
cv2.line(img_origin, (x-5,y-5), (x+5,y+5), (255, 0, 0), 2)
cv2.line(img_origin, (x+5,y-5), (x-5,y+5), (255, 0, 0), 2)
ax1.imshow(img_origin,cmap='gray')
ax1.set_title('Input Image')
ax1.add_artist(scalebar)

#theta-半径グラフ
ax2.set_axis_off()
ax2.set_axis_on()
ax2.set_xticks(
    [0, np.pi/4, np.pi/2, np.pi*3/4, np.pi, np.pi*5/4, np.pi*3/2, np.pi*7/4], 
    ["0", "\u03c0/4", "\u03c0/2", "3\u03c0/4", "\u03c0", "5\u03c0/4", "3\u03c0/2", "7\u03c0/4"]
)

# search node & edge
for i in range(len(position)):
    for j in range(len(position[i])):
        if position[i][j][4]==-1:
            edge.append([position[i][j][1],position[i][j][0]]) #theta,rの順に格納
        elif position[i][j][5]==1:
            node.append([position[i][j][1],position[i][j][0]])

for i , r in enumerate(search_range):
    for j in range(len(position[i])):
        ax2.scatter(position[i][j][1],r,s=1,c="k")

for i in range(len(node)):
    ax2.scatter(node[i][0],node[i][1],s=10,c="r")

for i in range(len(edge)):
    ax2.scatter(edge[i][0],edge[i][1],s=10,c="b")

lc1 = mc.LineCollection(vector, colors="k", linewidths=1)

#r,theta,対応する枝の番号の分の情報をつけて再帰的に探索
ax2.add_collection(lc1)
# plt.savefig(str(name_tag)+".png")
# plt.savefig(str(name_tag)+"_lareg.png")
finish=time.time()
plt.show()
total_time=finish-start
print("total time:",total_time)
print("x:",x,",(Lx-x):",(Lx-x),",y:",y,",(Ly-y):",(Ly-y))

#時間計測メモ file3で計測
# 系サイズ最大、1ステップ毎：75.58797311782837 sec ->20倍になっても時間は10倍ほど
# 系サイズ最大、r_max/20毎：7.566540241241455 sec
# 重心から系の端まで、1ステップ毎：77.58949756622314 sec ->こっちの方がむしろ長い？
# 重心から系の端まで、1ステップ毎、探索関数にif文をかませない：83.54880547523499 sec ->ifをかませない方が長い？
# 重心から系の端まで、r_max/20毎：6.782961368560791 sec
# 点数が増えるとplot関数により時間がかかる印象,cpuの使用状況に多少よるかも



# step=100
# R=50
# epsilon=20
# dth=2*math.pi/step
# dr=2*R/step
# theta=[dth*i for i in range(step)]
# r=[dr*i for i in range(step)]
# f=[[surface(epsilon,R,r[i]) for j in range(step)] for i in range(step)]
# # fig , ax=plt.subplots( )
# fig , ax = plt.subplots(figsize=(9,9),subplot_kw={'projection': 'polar'})
# map=ax.pcolor(theta,r,f)
# plt.colorbar(map, ax=ax)
# plt.show()

# #main
# #Video Source
# Dir_name="/mnt/c/Users/PC/Desktop/data_for_contact_angle/"
# fname_list=[]
# for i in range(4):
#     fname="contact_angle_H2O_Plronic_TWEEN_00"+str(i+1)+".tif"
#     fname_list.append(fname)

# file_path=Dir_name + fname_list[0]
# name_tag=file_path.replace(Dir_name,"")
# name_tag=name_tag.replace(".tif","")
# window_name = file_path[len(Dir_name):]
# img=cv2.imread(file_path,1)
# img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# print(window_name)
# print(type(img))

# hight, width, channels=img.shape
# print("File Name : ",window_name)
# print("Frame Width : ", width)
# print("Frame Hight : ", hight)

# # # 大津の二値化
# ret,th = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# # スレッショルドを切る
# # ret,th = cv2.threshold(img_gray,40,255,cv2.THRESH_BINARY)


# # カーネルの設定
# kernel = np.ones((5,5),np.uint8)

# # モルフォロジー変換（膨張）
# th_dilation = cv2.dilate(th,kernel,iterations = 1)

# # モルフォロジー変換（収縮）
# th_elosion = cv2.erode(th,kernel,iterations = 1)

# #オープニング処理(収縮->膨張),極板の除去
# th_opening=cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel)

# #クロージング処理(膨張->収縮),黒い穴の除去
# kernel=np.ones((20,20),np.uint8) #カーネルの変更
# th_closing=cv2.morphologyEx(th_opening,cv2.MORPH_CLOSE,kernel)

# # 輪郭抽出
# contours, hierarchy = cv2.findContours(th_closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE) #RETR_LISTは白領域をCCWで探索

# # 輪郭を元画像に描画
# img_contour = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

# # # 重心と両端の点の取得,探索は黒領域のCCWなのでcontersには右->左->中の順番で格納
# color=["r","b","g"]
# Q=[]
# Area=[]
# px=[]
# py=[]
# for i , c in enumerate(contours):
#     area=cv2.contourArea(c)
    
#     index_L=np.where(c[...,0]==np.min(c[...,0]))
#     left=c[index_L[0][len(index_L[0])-1]][0] #各液滴の一番左下の検出(RETR_LISTで白領域をCCWで探索の場合)

#     index_R=np.where(c[...,0]==np.max(c[...,0]))
#     right=c[index_R[0][0]][0] #各液滴の一番右下の検出(RETR_LISTで白領域をCCWで探索の場合)

#     q1 = left
#     q2=np.round(left+(right-left)/3.0).astype(int)
#     q3=np.round(left+(right-left)*2.0/3.0).astype(int)
#     q4 = right
#     q2+=[0,-50]
#     q3+=[0,-50]

#     Q.append([q1,q2,q3,q4])
#     Area.append(area)
#     px.append(bezie_curve(Q[i])[0])
#     py.append(bezie_curve(Q[i])[1])

# y_bottom=bottom_line(q1,q4,px)
# print(Area)
# print(Q)
# print(bezie_area_exact(Q[0]),bezie_area_exact(Q[1]),bezie_area_exact(Q[2]))
# bezie_area=[]

# for i in range(len(contours)):
#     area_temp=0
#     for j in range(len(py[i])-1):
#         dx=px[i][j+1]-px[i][j]
#         area_temp+=(bottom_line(Q[i][0],Q[i][3],px[i][j])-py[i][j])*dx
#     print(area_temp)

# fig= plt.figure(figsize=(18,9))
# ax1=fig.add_subplot(1,2,1)
# ax2=fig.add_subplot(1,2,2)
# # ax3=fig.add_subplot(2,2,3)

# #元画像(gray)
# ax1.imshow(img_contour,cmap='gray')
# ax1.set_title('Input Image')

# #二値化画像(binary)
# ax2.imshow(th_closing,cmap='gray')
# ax2.set_title('Binary Image')

# #各液滴の重心を描画、探索は黒領域のCCWなので右->左->中の順番で格納
# color=["r","b","g"]
# for i , c in enumerate(contours):
#     M = cv2.moments(c)
#     x = int(M["m10"] / M["m00"])
#     y = int(M["m01"] / M["m00"])
    
#     ax2.plot(x,y,marker='.',c=color[i])
#     for j in range(len(Q[i])):
#         ax2.plot(Q[i][j][0],Q[i][j][1],marker="o",c=color[i])
#         ax2.plot(px[i],py[i],c=color[i])
#         ax2.plot(px[i],y_bottom[i],c=color[i])

#     for j in range(len(py[i])):
#         x1=[px[i][j],px[i][j]]
#         y1=[y_bottom[i][j],py[i][j]]
#         ax2.plot(x1,y1,c=color[i])

# # レイアウト設定
# fig.tight_layout()

# # グラフを表示する。
# # plt.savefig(str(name_tag)+".png")
# plt.show()


# # ---------------------------------------------------