import cv2
import numpy as np

# Video
frameWidth = 640
frameHeight = 480

#Video Source
# Path="D:/master_thesis_data/experment_data/movie_data/movie_data/data_for_test/"
# aa=np.loadtxt("mnt/c/ユーザー/PC/デスクトップ/Master_Thesis/test/test1.txt")
# aa=np.loadtxt("/mnt/c/Users/PC/Desktop/Master_Thesis/test/test1.txt")
# print(aa)
bb=np.loadtxt("/mnt/d/dendrite_data/edited_data/analisys_data/20230204_0.05sur_67.3mN_No.1.dat")
print(bb)
# Path="/mnt/d/test"
# test=np.loadtxt(Path+"/test.txt")
# print(test)

# parm=0.00
# cap = cv2.VideoCapture("D:/dendrite_data/edited_data/edited_movie/20230221_nonsur_76.6mN_No.2.avi") #自分のaviのpathを入力
# #cap = cv2.VideoCapture(0)
# #test=np.loadtxt(Path+f"20231116_test_2ML_{parm}sur/test_data.dat")
# "D:\master_thesis_data\experment_data\movie_data\movie_data\data_for_test\20231116_test_2ML_0.00sur\test.txt"
# print(cap.isOpened())

# while cap.isOpened:
#     ret, img = cap.read()
        
#     cv2.imshow("img",img)
#     # img = cv2.resize(img, (frameWidth, frameHeight))
#     # cv2.imshow('Video', img)
#     # print('ret=', ret)

#     # qを押すと止まる。
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
