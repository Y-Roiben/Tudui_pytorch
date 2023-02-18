# Designed by Roiben
# 开发时间： 2022-08-31 0:29

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)  # x-y 平面的网格
R = np.sqrt(X ** 2 + Y ** 2)
# height value
Z = np.sin(R)
plt.title("X-Y-Z")
# 绘制3D图
# 用取样点(x,y,z)去构建曲面,rstride和cstride表示行列隔多少个取样点建一个小面
surf = ax.plot_surface(X, Y, Z,
                       rstride=1,  # rstride（row）指定行的跨度
                       cstride=1,  # cstride(column)指定列的跨度
                       cmap=plt.get_cmap('rainbow')  # 设置颜色映射
                       )

fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_zlabel('z label')  # 给三个坐标轴注明

# 下面添加平面的等高线
# ax.contourf(X, Y, Z, zdir='y', offset=-4.5, cmap=plt.get_cmap('rainbow'))
plt.show()
