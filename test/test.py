import numpy as np
import matplotlib.pyplot as plt
# カラーマップを指定
# color_map_list = ["GnBu", "Set3" , "prism", "magma", "twilight", "gnuplot"]

x = np.linspace(0, np.pi*2)
y = np.sin(x)
# カラーマップごとに0〜1の値で色を変えて10本描画
# for color in color_map_list:
for n in range(41):
    # ↓↓↓ カラーマップインスタンスをplt.get_cmap("GnBu")で作成
    color_map = plt.get_cmap("tab20")

    # ↓↓↓ 指定したカラーマップの0〜1に対応する色で描画
    plt.plot(x, y+n, color=color_map(n/40))
    plt.text(0, n-0.5, n/40)
    n -= 2

plt.xticks([])
plt.yticks([])
plt.show()