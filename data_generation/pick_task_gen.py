import sys ,os
from dynamic_movement_primitives.path import dmp_lib_path,data_path
sys.path.append(dmp_lib_path)  
from dmp_discrete import DMPs_discrete
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb

def plot(demo_list,num=0,color="grey"):
    fig = plt.figure(num,figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w')
    ax = fig.gca(projection='3d')
    for demo in demo_list:
        ax.plot(demo[:,0],demo[:,1],demo[:,2],color=color)
    # plt.show()

def get_demo_var(demo_list):
    start_list = []
    goal_list = []
    for demo in demo_list:
        start = demo[0]
        goal = demo[-1]
        start_list.append(start)
        goal_list.append(goal)
    mean_start = np.mean(start_list,axis=0)
    mean_goal = np.mean(goal_list,axis=0)
    std_start = np.std(start_list,axis=0,ddof=1)
    std_goal = np.std(goal_list,axis=0,ddof=1)
    return mean_start,mean_goal,std_start,std_goal

def dmp_generate_w_mean(test_demo,mean_start,mean_goal):
    dim = test_demo.shape[1]
    dmp = DMPs_discrete(n_dmps=dim, n_bfs=300, ay=np.ones(dim)*25.0,dt=0.001 )
    dmp.imitate_path(y_des=test_demo.T, plot=False)
    dmp.goal=mean_goal
    dmp.y0=mean_start
    y_track, dy_track, ddy_track = dmp.rollout()
    return np.array(y_track)


def generate_new_start_goal(mean_start,mean_goal,std_start,std_goal,num_traj=10):
    rand_start = []
    for i in range(num_traj*7):
        rand = np.random.uniform(-1,1)
        rand_start.append(rand)
    rand_start = np.array(rand_start).reshape(num_traj,7)

    rand_goal = []
    for i in range(num_traj*7):
        rand = np.random.uniform(-1.5,1.5)
        rand_goal.append(rand)
    rand_goal = np.array(rand_goal).reshape(num_traj,7)

    new_start_list = []
    new_goal_list = []
    for idx in range(num_traj):
        new_start_list.append(mean_start+ std_start*rand_start[idx])
        new_goal_list.append(mean_goal+ std_goal*rand_goal[idx])

    return new_start_list, new_goal_list

def generate_new_demo(mean_demo,new_start_list, new_goal_list):
    dim = mean_demo.shape[1]
    y_track_list =[]
    for idx in range(len(new_start_list)):
        dmp = DMPs_discrete(n_dmps=dim, n_bfs=300, ay=np.ones(dim)*25.0,dt=0.001 )
        dmp.imitate_path(y_des=mean_demo.T, plot=False)
        dmp.goal=new_goal_list[idx]
        dmp.y0=new_start_list[idx] 
        y_track, dy_track, ddy_track = dmp.rollout()
        y_track_list.append(y_track)
    return np.array(y_track_list)



norm_data_list = joblib.load(os.path.join(data_path,'datasets_norm.pkl'))
test_demo = norm_data_list[0]
mean_start,mean_goal,std_start,std_goal = get_demo_var(norm_data_list)
mean_demo = dmp_generate_w_mean(test_demo,mean_start,mean_goal)
new_start_list, new_goal_list = generate_new_start_goal(mean_start,mean_goal,std_start,std_goal,num_traj=20)
y_track_list = generate_new_demo(mean_demo,new_start_list, new_goal_list)
joblib.dump(y_track_list,os.path.join(data_path,"dmp_generated_demo"))
# plot(norm_data_list,num=0)
# plot(y_track_list,num=0,color="r")
# print std_start
# print "-"*6
# print std_goal
# plt.show()