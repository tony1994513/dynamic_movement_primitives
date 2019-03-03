'''
Copyright (C) 2013 Travis DeWolf

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
import sys  
from dynamic_movement_primitives.path import dmp_lib_path
sys.path.append(dmp_lib_path)  
from dmp import DMPs
from dmp_discrete import DMPs_discrete
import numpy as np

if __name__ == "__main__":
    import matplotlib.pyplot as plt
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

    # test imitation of path run
    # plt.figure(1 ,facecolor='w', edgecolor='w'  )
    n_bfs = [10, 30, 50, 100, 10000]

    # a straight line to target
    path1 = np.cos(np.arange(0, 1, .01)*2)
    # a strange path to target
    path2 = np.zeros(path1.shape)
    path2[int(len(path2) / 2.):] = .5

    for ii, bfs in enumerate(n_bfs):
        dmp = DMPs_discrete(n_dmps=2, n_bfs=bfs)

        dmp.imitate_path(y_des=np.array([path1, path2]))
        # change the scale of the movement
        dmp.goal[0] = 0
        dmp.goal[1] = 1

        y_track, dy_track, ddy_track = dmp.rollout()

        plt.figure(1, facecolor='w', edgecolor='w')
        plt.subplot(211)
        plt.plot(y_track[:, 0], lw=2)
        plt.subplot(212)
        plt.plot(y_track[:, 1], lw=2)

    a = plt.plot(path1, 'r--', lw=2)
    plt.title('DMP imitate path')
    plt.xlabel('time (ms)')
    plt.ylabel('system trajectory')
    plt.legend([a[0]], ['desired path'], loc='lower right')
    plt.subplot(212)
    b = plt.plot(path2 / path2[-1] * dmp.goal[1], 'r--', lw=2)
    plt.title('DMP imitate path')
    plt.xlabel('time (ms)')
    plt.ylabel('system trajectory')
    plt.legend(['%i BFs' % i for i in n_bfs], loc='lower right')

    plt.tight_layout()
    plt.show()
