import math
import sys
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.collections as mc
import matplotlib.cm as cm
import time
import copy

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

def search_branch(binary,r,x_c,y_c):  # binary data,radius r, cm_x,cm_y
    if r==1:
        dth=math.pi/2
    else:
        dth = 2*math.asin(1/(2*r))
        
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

def next_point(i:int,j:int):#position[i][j]
    rnow=position[i][j][0]
    thnow=position[i][j][1]
    dist2=pow(r_max,2) #十分大きな数で
    for k in range(max(i,0),min(i+5,len(position))):
        rnext=position[k][0][0]
        for l in range(len(position[k])):
            if (k==i and l==j) or ((k in skip_list_i) and (l in skip_list_j)):
                continue
            thnext=position[k][l][1]
            dist_tmp2=pow(rnow,2)+pow(rnext,2)-2*rnow*rnext*math.cos(thnow-thnext)
            if dist2 > dist_tmp2:
                dist2=dist_tmp2
                index_i=k
                index_j=l

    skip_list_i.append(index_i)
    skip_list_j.append(index_j)
    
    return index_i , index_j

def tree_search(i,j):
    if i < len(position)-1:
        i_tmp , j_tmp = next_point(i,j)
        position[i][j][3]=position[i_tmp][j_tmp][2] #[i][j] in <- [i_tmp][j_tmp] position_id
        if position[i_tmp][j_tmp][4]!=0:
            position[i_tmp][j_tmp][5]=1 #node_flag=True
            return
        else:
            position[i_tmp][j_tmp][4]=position[i][j][2] #[i_tmp][j_tmp] out <- [i][j] position_id

        tree_search(i_tmp,j_tmp)

def branch_search(i,j):
    if i < len(position):
        i_tmp , j_tmp = next_point(i,j)
        position[i][j].append(branch_id[0])
        if position[i][j][5] == 1:
            branch_id[0]+=1
            position[i][j].append(branch_id[0])

        if i_tmp==len(position)-1: #中心の例外処理
            position[i_tmp][0].append(branch_id[0])
            return

        branch.append([[position[i][j][1],position[i][j][0]],[position[i_tmp][j_tmp][1],position[i_tmp][j_tmp][0]]])
        
        branch_search(i_tmp,j_tmp)

def branch_angle(node:list,branch1:list,branch2:list): #各点の極座標(theta,r)をわたす
    r0=node[1]
    r1=branch1[1]
    r2=branch2[1]
    th0=node[0]
    th1=branch1[0]
    th2=branch2[0]

    r01_2=pow(r0,2)+pow(r1,2)-2*r0*r1*math.cos(th0-th1)
    r12_2=pow(r1,2)+pow(r2,2)-2*r1*r2*math.cos(th1-th2)
    r20_2=pow(r2,2)+pow(r0,2)-2*r2*r0*math.cos(th2-th0)

    value=(-r12_2+r01_2+r20_2)/(2*math.sqrt(r01_2*r20_2))
    
    if r01_2==0 or r12_2==0 or r20_2==0: #エラー処理
        print(node,branch1,branch2)
        print("Emergency Stop")
        sys.exit(1)
        
    #数値誤差で+-1を超えることがたまにあるのでその例外処理
    if value < -1.0:
        angle=math.acos(-1.0)
    elif value > 1.0:
        angle=math.acos(1.0)
    else:
        angle=math.acos(value)

    return angle

def Next_Long_Branch(i:int):
    for j in range(len(Long_branch_vector_tmp)):
        if j==i:
            continue
        if Long_branch_vector_tmp[i][1]==Long_branch_vector_tmp[j][0]:
            return j

    return len(Long_branch_vector_tmp)
        
def Make_Long_Branch(i:int):
    i_tmp=Next_Long_Branch(i)
    if i_tmp < len(Long_branch_vector_tmp):
        skip_list.append(i_tmp)
        tmp.append(Long_branch_vector_tmp[i_tmp])
        Make_Long_Branch(i_tmp)

def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

#Main
#time
start=time.time()

#constants
dust = 4  # チリの大きさ判定用変数
cut = 30  # threshold value,輝度値は0が黒色、255が白色。

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

n0 = First_Frame() # origin frame number, time=0

#Getting Last Frame
cap.set(cv2.CAP_PROP_POS_FRAMES,Total_Frames-1)
ret,image=cap.read()

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#輝度値情報のみの画像
threshold,binary=cv2.threshold(gray,cut,255,cv2.THRESH_BINARY)#完全に二値化(RGB -> 輝度値算出 ->二値化)
threshold,nongray_binary=cv2.threshold(image,cut,255,cv2.THRESH_BINARY)#RGBを残したまま二値化

binary=Remove_Dust(binary)

#Making Position Data
print("Start Calculation")
x,y=CM(n0) #重心計算
r_max=max(x,(Lx-x),y,(Ly-y))#最大半径＝重心からの距離の最大値
search_range=[r for r in reversed(range(1,r_max))]
position=[[]for i in range(len(search_range))]
position_id=1
branch_id=[1] #Pythonには参照渡しは存在しない(objectのaddressを値渡しする)ため、関数内ではコピーして渡されたaddressにアクセスしてobjectを書き換えることで参照渡しのような挙動を実現するらしい。

for i , r in enumerate(search_range):
    branch_th=search_branch(binary,r,x,y)
    for j in range(len(branch_th)):
        position[i].append([r,2*math.pi-branch_th[j],position_id,0,0,0]) #radius ,theta,position,in,out,node
        position_id+=1

#Adding origin point
position_id+=1
position.append([[0,0,position_id,0,0,0]]) #radius ,theta,position,in,out,node

#Tree search 
for i in range(len(position)):
    for j in range(len(position[i])):
        if position[i][j][4]==0: #最外端の判定
            skip_list_i=[]
            skip_list_j=[]
            skip_list_i.append(i)
            skip_list_j.append(j)
            position[i][j][4]=-1
            tree_search(i,j)

#Adding branch ID & Tracking branch
branch=[]
for i in range(len(position)):
    for j in range(len(position[i])):
        if position[i][j][4]==-1: 
            skip_list_i=[]
            skip_list_j=[]
            skip_list_i.append(i)
            skip_list_j.append(j)
            branch_search(i,j)
            branch_id[0]+=1

branch=get_unique_list(branch)

# Search node & edge
node=[]
edge=[]
branch_vector_edge=[]
for i in range(len(position)):
    for j in range(len(position[i])):
        if position[i][j][4]==-1:
            edge.append([position[i][j][1],position[i][j][0]]) #theta,rの順に格納,枝の端
            branch_vector_edge.append(position[i][j])
        elif position[i][j][5]==1:
            node.append([position[i][j][1],position[i][j][0]]) #枝の節
            branch_vector_edge.append(position[i][j])

# make branch vector
branch_vector=[]
branch_vector_length=[]
for i in range(len(branch_vector_edge)):
    for j in range(i+1,len(branch_vector_edge)):
        len1=len(branch_vector_edge[i])
        len2=len(branch_vector_edge[j])
        tmp1=[branch_vector_edge[i][6:len1][k] for k in range(len(branch_vector_edge[i][6:len1]))]
        tmp2=[branch_vector_edge[j][6:len2][k] for k in range(len(branch_vector_edge[j][6:len2]))]
        if not set(tmp1).isdisjoint(set(tmp2)) :
            rev1=list(reversed(branch_vector_edge[i][0:2]))
            rev2=list(reversed(branch_vector_edge[j][0:2]))
            branch_vector.append([rev1,rev2]) #vecter edge,[out,in],r:large -> small
            length=math.sqrt(pow(rev1[1],2)+pow(rev2[1],2)-2*rev1[1]*rev2[1]*math.cos(rev1[0]-rev2[0]))
            branch_vector_length.append(length)

# make branch vector angle
branch_vector_angle=[]
for i in reversed(range(len(branch_vector))): #内から外へ探索
    angle=math.pi
    for j in reversed(range(0,i)):
        if branch_vector[i][1]==branch_vector[j][1]:
            angle_tmp=branch_angle(branch_vector[i][1],branch_vector[i][0],branch_vector[j][0])
            if angle_tmp <= angle:
                angle=angle_tmp
    if angle!=math.pi: #これがないと、分岐しないベクトルの時に初期値πが入ってしまう
        branch_vector_angle.append(angle)

#Making Long & Short Branch Vector
threshold_angle=(1-1/10)*math.pi
print("Threshold Angle is " + str(threshold_angle/math.pi) +"\u03c0")
Long_branch_vector_tmp=[]
Short_branch_vector=copy.deepcopy(branch_vector)#copy methodを使わないと、Pythonの場合リストは"代入"ではなく、"同じリストオブジェクトの参照"の意味になる。
Long_branch_vector_length=[]
Short_branch_vector_length=[]
branch_vector_outer_anlge=[]
for i in reversed(range(len(branch_vector))):
    angle=0
    for j in reversed(range(0,i)):
        if branch_vector[i][0]==branch_vector[j][1]:
            angle_tmp = branch_angle(branch_vector[i][0],branch_vector[i][1],branch_vector[j][0])
            if angle_tmp >= angle:
                angle=angle_tmp
                i_tmp=i
                j_tmp=j
    if angle!=0: #これがないと端の枝(始めのif文にかからない時)にもカウントされてしまう
        branch_vector_outer_anlge.append(angle)

    if angle>= threshold_angle:
        rev1=list(reversed(branch_vector[i_tmp]))
        rev2=list(reversed(branch_vector[j_tmp]))
        Long_branch_vector_tmp.append([rev1,rev2]) #内側,外側ベクトルの順で格納,rについても内から外に格納
        if branch_vector[i_tmp] in Short_branch_vector:
            Short_branch_vector.remove(branch_vector[i_tmp])
        if branch_vector[j_tmp] in Short_branch_vector:
            Short_branch_vector.remove(branch_vector[j_tmp])

skip_list=[]
Long_branch_vector=[]
for i in range(len(Long_branch_vector_tmp)):
    if i in skip_list:
        continue
    skip_list.append(i)
    tmp=[Long_branch_vector_tmp[i]]
    Make_Long_Branch(i)
    Long_branch_vector.append([tmp[0][0][0],tmp[-1][-1][-1]])
    r1=tmp[0][0][0][1]
    r2=tmp[-1][-1][-1][1]
    th1=tmp[0][0][0][0]
    th2=tmp[-1][-1][-1][0]
    length_long=math.sqrt(pow(r1,2)+pow(r2,2)-2*r1*r2*math.cos(th1-th2))
    Long_branch_vector_length.append(length_long)

for i in range(len(Short_branch_vector)):
    r1=Short_branch_vector[i][0][1]
    r2=Short_branch_vector[i][1][1]
    th1=Short_branch_vector[i][0][0]
    th2=Short_branch_vector[i][1][0]
    length_short=math.sqrt(pow(r1,2)+pow(r2,2)-2*r1*r2*math.cos(th1-th2))
    Short_branch_vector_length.append(length_short)

Edited_vector_length=copy.deepcopy(Long_branch_vector_length)
Edited_vector_length.extend(Short_branch_vector_length)

print("Finish Calculation")
print("Start Making Figure")

# # Making Images
# img_origin=image #original image
# scalebar=ScaleBar(11/681,"cm",length_fraction=0.5,location="lower right")

# # Making figure
# fig = plt.figure(figsize=(18,9))
# ax1=fig.add_subplot(1,2,1)
# # ax2=fig.add_subplot(1,2,2,projection="polar")
# ax2=fig.add_subplot(1,2,2)

# #元画像(gray)
# cv2.line(img_origin, (x-5,y-5), (x+5,y+5), (255, 0, 0), 2)
# cv2.line(img_origin, (x+5,y-5), (x-5,y+5), (255, 0, 0), 2)
# ax1.imshow(img_origin,cmap='gray')
# ax1.set_title('Input Image')
# ax1.add_artist(scalebar)

# plt.savefig(str(name_tag)+".png")

fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(9,9))

#theta-半径グラフ
ax2.set_axis_off()
ax2.set_axis_on()
ax2.set_xticks(
    [0, np.pi/4, np.pi/2, np.pi*3/4, np.pi, np.pi*5/4, np.pi*3/2, np.pi*7/4], 
    ["0", "\u03c0/4", "\u03c0/2", "3\u03c0/4", "\u03c0", "5\u03c0/4", "3\u03c0/2", "7\u03c0/4"]
)

for i , r in enumerate(search_range):
    for j in range(len(position[i])):
        ax2.scatter(position[i][j][1],r,s=1,c="k")

for i in range(len(node)):
    ax2.scatter(node[i][0],node[i][1],s=10,c="r")

for i in range(len(edge)):
    ax2.scatter(edge[i][0],edge[i][1],s=10,c="b")

ax2.scatter(position[len(position)-1][0][1],position[len(position)-1][0][0],s=10,c="m") #Plot Center

color_list = ["y", "c" , "coral" , "orange" , "gold" , "olive" , "greenyellow" , "turquoise" , "lightseagreen" , "navy" , "plum" , "crimson"]

lc1 = mc.LineCollection(branch, colors="k", linewidths=1)
# lc2 = mc.LineCollection(branch_vector,colors=color_list,linewidth=1)
lc3 = mc.LineCollection(Short_branch_vector, colors="orchid", linewidths=1)
lc4 = mc.LineCollection(Long_branch_vector, colors="g", linewidths=1)

ax2.add_collection(lc1)
# ax2.add_collection(lc2)
ax2.add_collection(lc3)
ax2.add_collection(lc4)

plt.savefig(str(name_tag)+"_lareg.pdf")
# plt.savefig(str(name_tag)+"_lareg.eps")
# plt.savefig(str(name_tag)+".png")

# Making Date File
Dir_path="/mnt/c/Users/PC/Desktop/Master_Thesis/movie_analisis/movie_analisis_data"
fname=str(name_tag)+".dat"

data_dir_list=["/Vector_length/","/Vector_angle/","/Edited_vector_length/","/Vector_outer_angle/"]
data_list=[branch_vector_length,branch_vector_angle,Edited_vector_length,branch_vector_outer_anlge]
data_name=["#length (cm)","#angle (radian)"]
unit=11/681 #cm/pix

for i in range(len(data_dir_list)):
    path=Dir_path+data_dir_list[i]
    if i%2==0: #length
        os.makedirs(path,exist_ok=True)
        path=path+fname
        with open(path,mode="w") as f:
            f.write("#length (cm) \n")
            for j in range(len(data_list[i])):
                f.write(str(data_list[i][j])+"\n")
    else: #angle
        os.makedirs(path,exist_ok=True)
        path=path+fname
        with open(path,mode="w") as f:
            f.write("#angle (radian) \n")
            for j in range(len(data_list[i])):
                f.write(str(data_list[i][j])+"\n")

finish=time.time()
print("Finish Making Figure")
# plt.show()
total_time=finish-start
print("total time:",total_time)
print("x:",x,",(Lx-x):",(Lx-x),",y:",y,",(Ly-y):",(Ly-y))
