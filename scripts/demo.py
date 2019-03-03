#!/usr/bin/python
import pandas as pd
import numpy as np
from birl_baxter_dmp.dmp_train import train
from birl_baxter_dmp.dmp_generalize import dmp_imitate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pydmps
import pydmps.dmp_discrete
import ipdb
from birl_baxter_dmp.path import dmp_lib_path,data_path
import os 

data = os.path.join(data_path, "3D_demo_data.txt")

PointCount = len(open(data,'rU').readlines())
Colum0_Traj=[0.0]*PointCount
Colum1_Traj=[0.0]*PointCount
Colum2_Traj=[0.0]*PointCount
Colum0_Traj=[float(l.split()[0]) for l in open(data)]
Colum1_Traj=[float(l.split()[1]) for l in open(data)]
Colum2_Traj=[float(l.split()[2]) for l in open(data)]    
traj=[[0.0, 0.0, 0.0]]*PointCount
for i in range(PointCount):
    traj[i]=[Colum0_Traj[i], Colum1_Traj[i], Colum2_Traj[i]]
y_des = np.array([Colum0_Traj,Colum1_Traj,Colum2_Traj]) # 
y_des = y_des.T
'''
If you wanna put in a set data, do it like this  y_des = np.array([[Colum0_Traj,Colum1_Traj,Colum2_Traj]])
If you wanna put in many sets data, do it like this y_des = np.array([[Colum0_Traj,Colum1_Traj,Colum2_Traj],[Colum0_Traj,Colum1_Traj,Colum2_Traj]])
'''


train_set = [y_des] 

param, base_function = train(train_set)


start_point = traj[0]  # [0.01, -0.80985915, -0.79287305]
ending_point = traj[-1] # [[2.0, 0.50469484, -0.080178174]]
goal_list = [[1.9, 0.50, -0.07],
            [1.8, 0.48, -0.07],
            [1.7, 0.48, -0.07],
            [1.6, 0.48, -0.07],
            [1.5, 0.48, -0.07],
            [1.4, 0.48, -0.07],
]

y_track_replay = dmp_imitate(starting_pose=start_point, ending_pose=ending_point, weight_mat=param )

#creat fig
fig=plt.figure()
ax = Axes3D(fig)    
plt.xlabel('X')
plt.ylabel('Y')

#plot traj fig 
ax.plot(Colum0_Traj,Colum1_Traj,Colum2_Traj,linewidth=3,label="Demonstration",color="black",)         
for goal in goal_list:
    y_track = dmp_imitate(starting_pose=start_point, ending_pose=goal, weight_mat=param )
    ax.plot(y_track[:,0],y_track[:,1],y_track[:,2],linewidth=2)
    ax.scatter(goal[0],goal[1],goal[2],color="r",s=100)
    ax.legend()
plt.show()
