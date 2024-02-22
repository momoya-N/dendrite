import numpy as np
import math
from matplotlib import pyplot as plt
import cv2
from PIL import Image

# Bernstein多項式を計算する関数
def bernstein(n, t):
    B = []
    for k in range(n + 1):
        # 二項係数を計算してからBernstein多項式を計算
        nCk = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
        B.append(nCk * t ** k * (1 - t) ** (n - k))
    return B

# ベジェ曲線を描く関数
def bezie_curve(Q):
    n = len(Q) - 1
    dt = 0.001
    t = np.arange(0, 1 + dt, dt)
    B = bernstein(n, t)
    px = 0
    py = 0
    for i in range(len(Q)):
        px += np.dot(B[i], Q[i][0])
        py += np.dot(B[i], Q[i][1])
    return px, py

#液滴の底面算出
def bottom_line(q1,q4,px):
    delta=q4-q1
    alpha=delta[1]/delta[0]

    y_botom=alpha*(px-q1[0])+q1[1]

    return y_botom

#ベジエ曲線の面積の厳密解
def bezie_area_exact(Q):
    S_bottom=0.5*(Q[3][0]-Q[0][0])*(Q[0][1]+Q[3][1])
    S_bezie=0.05*(-Q[3][1]*(Q[0][0]+3*Q[1][0]+6*Q[2][0]-10*Q[3][0])-3*Q[2][1]*(Q[0][0]+Q[1][0]-2*Q[3][0])+3*Q[1][1]*(-2*Q[0][0]+Q[2][0]+Q[3][0])+Q[0][1]*(-10*Q[0][0]+6*Q[1][0]+3*Q[2][0]+Q[3][0]))

    S_exact=S_bottom-S_bezie

    return S_exact

#色のヒストグラム作成
def color_hist(filename):
    img = np.asarray(Image.open(filename).convert("L")).reshape(-1,1)
    plt.hist(img, bins=128)
    plt.show()

#hypabolic tangent での境界表現
def surface(epsilnon,R): #epsilon:境界の幅(2*epsilon),R:円の半径
    step=100
    dth=2*math.pi()/step
    dr=2*R/step

    theta=[dth*i for i in range(step)]
    r=[dr*i for i in range(step)]
    
    f=0.5*(1+math.tanh((R-r)/epsilnon))

    return 

#main
#Video Source
Dir_name="/mnt/c/Users/PC/Desktop/data_for_contact_angle/"
fname_list=[]
for i in range(4):
    fname="contact_angle_H2O_Plronic_TWEEN_00"+str(i+1)+".tif"
    fname_list.append(fname)

file_path=Dir_name + fname_list[0]
name_tag=file_path.replace(Dir_name,"")
name_tag=name_tag.replace(".tif","")
window_name = file_path[len(Dir_name):]
img=cv2.imread(file_path,1)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

print(window_name)
print(type(img))

hight, width, channels=img.shape
print("File Name : ",window_name)
print("Frame Width : ", width)
print("Frame Hight : ", hight)

# # 大津の二値化
ret,th = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# スレッショルドを切る
# ret,th = cv2.threshold(img_gray,40,255,cv2.THRESH_BINARY)


# カーネルの設定
kernel = np.ones((5,5),np.uint8)

# モルフォロジー変換（膨張）
th_dilation = cv2.dilate(th,kernel,iterations = 1)

# モルフォロジー変換（収縮）
th_elosion = cv2.erode(th,kernel,iterations = 1)

#オープニング処理(収縮->膨張),極板の除去
th_opening=cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel)

#クロージング処理(膨張->収縮),黒い穴の除去
kernel=np.ones((20,20),np.uint8) #カーネルの変更
th_closing=cv2.morphologyEx(th_opening,cv2.MORPH_CLOSE,kernel)

# 輪郭抽出
contours, hierarchy = cv2.findContours(th_closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE) #RETR_LISTは白領域をCCWで探索

# 輪郭を元画像に描画
img_contour = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

# # 重心と両端の点の取得,探索は黒領域のCCWなのでcontersには右->左->中の順番で格納
color=["r","b","g"]
Q=[]
Area=[]
px=[]
py=[]
for i , c in enumerate(contours):
    area=cv2.contourArea(c)
    
    index_L=np.where(c[...,0]==np.min(c[...,0]))
    left=c[index_L[0][len(index_L[0])-1]][0] #各液滴の一番左下の検出(RETR_LISTで白領域をCCWで探索の場合)

    index_R=np.where(c[...,0]==np.max(c[...,0]))
    right=c[index_R[0][0]][0] #各液滴の一番右下の検出(RETR_LISTで白領域をCCWで探索の場合)

    q1 = left
    q2=np.round(left+(right-left)/3.0).astype(int)
    q3=np.round(left+(right-left)*2.0/3.0).astype(int)
    q4 = right
    q2+=[0,-50]
    q3+=[0,-50]

    Q.append([q1,q2,q3,q4])
    Area.append(area)
    px.append(bezie_curve(Q[i])[0])
    py.append(bezie_curve(Q[i])[1])

y_bottom=bottom_line(q1,q4,px)
print(Area)
print(Q)
print(bezie_area_exact(Q[0]),bezie_area_exact(Q[1]),bezie_area_exact(Q[2]))
bezie_area=[]

for i in range(len(contours)):
    area_temp=0
    for j in range(len(py[i])-1):
        dx=px[i][j+1]-px[i][j]
        area_temp+=(bottom_line(Q[i][0],Q[i][3],px[i][j])-py[i][j])*dx
    print(area_temp)

fig= plt.figure(figsize=(18,9))
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
# ax3=fig.add_subplot(2,2,3)

#元画像(gray)
ax1.imshow(img_contour,cmap='gray')
ax1.set_title('Input Image')

#二値化画像(binary)
ax2.imshow(th_closing,cmap='gray')
ax2.set_title('Binary Image')

#各液滴の重心を描画、探索は黒領域のCCWなので右->左->中の順番で格納
color=["r","b","g"]
for i , c in enumerate(contours):
    M = cv2.moments(c)
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])
    
    ax2.plot(x,y,marker='.',c=color[i])
    for j in range(len(Q[i])):
        ax2.plot(Q[i][j][0],Q[i][j][1],marker="o",c=color[i])
        ax2.plot(px[i],py[i],c=color[i])
        ax2.plot(px[i],y_bottom[i],c=color[i])

    for j in range(len(py[i])):
        x1=[px[i][j],px[i][j]]
        y1=[y_bottom[i][j],py[i][j]]
        ax2.plot(x1,y1,c=color[i])

# レイアウト設定
fig.tight_layout()

# グラフを表示する。
# plt.savefig(str(name_tag)+".png")
plt.show()


# ---------------------------------------------------
