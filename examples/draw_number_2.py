'''
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import sys,os  
from dynamic_movement_primitives.path import dmp_lib_path,data_path
sys.path.append(dmp_lib_path) 
import numpy as np
import matplotlib.pyplot as plt

import dmp_discrete
import ipdb

y_des = np.load(os.path.join(data_path,'2.npz'))['arr_0'].T
y_des -= y_des[:, 0][:, None]

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# test normal run
colors = ["r","lime","b"]
start = [
    np.array([0.001, 0]),
    np.array([0.2, 0.2]),
    np.array([0.2, 0.2]),
]
goal = [
    np.array([1.2, -1.4]),
    np.array([0.83333333, -1.35472803]),
    np.array([1.2, -1.401]),
]

plt.figure(1,facecolor='w', edgecolor='w')
plt.plot(y_des.T[:,0], y_des.T[:, 1], lw=3 ,label="Demonstration",color="grey")
for idx in range(len(goal)):
    # ipdb.set_trace()
    y_track = []
    dy_track = []
    ddy_track = []
    dmp = dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=300, ay=np.ones(2)*25.0,dt=0.001  )
    dmp.imitate_path(y_des=y_des,num=2, plot=False)
    dmp.goal=goal[idx]
    dmp.y0=start[idx]
    y_track, dy_track, ddy_track = dmp.rollout()
    plt.figure(1,facecolor='w', edgecolor='w')
    plt.scatter(dmp.goal[0],dmp.goal[1],label="goal_"+str(idx),marker="*", color=colors[idx], s=300,)
    plt.scatter(dmp.y0[0],dmp.y0[1],label="start_"+str(idx),marker="*", color=colors[idx], s=300,)
    plt.plot(y_track[:,0], y_track[:, 1],"--", lw=3 ,label="DMP_generalization"+str(idx),color=colors[idx])
    plt.title('DMP system - draw number 2')
    # plt.axis('equal')

plt.legend()
plt.show()
