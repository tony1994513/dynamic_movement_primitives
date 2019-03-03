import sys ,os
from dynamic_movement_primitives.path import dmp_lib_path,data_path
sys.path.append(dmp_lib_path)  
from dmp_discrete import DMPs_discrete
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb

def plot_old_demo(demo_list,num=0,color="grey"):
    fig = plt.figure(num,figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w')
    ax = fig.gca(projection='3d')
    for demo in demo_list:
        human = demo["left_hand"]
        robot = demo["left_joints"]
        ax.plot(human[:,0],human[:,1],human[:,2],color=color)
        ax.plot(robot[:,0],robot[:,1],robot[:,2],color=color)
    # plt.show()

def plot_new_demo(demo_list,num=0,color="grey",label=None):
    fig = plt.figure(num,figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w')
    ax = fig.gca(projection='3d')
    for demo in demo_list:
        ax.plot(demo[:,0],demo[:,1],demo[:,2],color=color,label=label)

def get_demo_var(demo_list):
    r_start_list = []
    r_goal_list = []
    h_start_list = []
    h_goal_list = []
    time_list = []
    for demo in demo_list:
        
        r_start = demo["left_joints"][0]
        h_start = demo["left_hand"][0]
        r_goal = demo["left_joints"][-1]
        h_goal = demo["left_hand"][-1]
        time = demo["alpha"]
        r_start_list.append(r_start)
        r_goal_list.append(r_goal)
        h_start_list.append(h_start)
        h_goal_list.append(h_goal)
        time_list.append(time )
    h_mean_start = np.mean(h_start_list,axis=0)
    h_mean_goal = np.mean(h_goal_list,axis=0)
    h_std_start = np.std(h_start_list,axis=0,ddof=1)
    h_std_goal = np.std(h_goal_list,axis=0,ddof=1)
    
    r_mean_start = np.mean(r_start_list,axis=0)
    r_mean_goal = np.mean(r_goal_list,axis=0)
    r_std_start = np.std(r_start_list,axis=0,ddof=1)
    r_std_goal = np.std(r_goal_list,axis=0,ddof=1)  

    time_mean = np.mean(time_list,axis=0)
    time_std = np.std(time_list,axis=0,ddof=1)
    return h_mean_start,h_mean_goal,h_std_start,h_std_goal,r_mean_start,r_mean_goal,r_std_start,r_std_goal,time_mean,time_std

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
    # ipdb.set_trace()
    dim = len(mean_start)
    for i in range(num_traj*dim):
        rand = np.random.uniform(-1,1)
        rand_start.append(rand)
    rand_start = np.array(rand_start).reshape(num_traj,dim)
    rand_goal = []
    for i in range(num_traj*dim):
        rand = np.random.uniform(-1.5,1.5)
        rand_goal.append(rand)
    rand_goal = np.array(rand_goal).reshape(num_traj,dim)

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
        dmp = DMPs_discrete(n_dmps=dim, n_bfs=300, ay=np.ones(dim)*25.0,dt=0.005 )
        dmp.imitate_path(y_des=mean_demo.T, plot=False)
        dmp.goal=new_goal_list[idx]
        dmp.y0=new_start_list[idx] 
        y_track, dy_track, ddy_track = dmp.rollout()
        y_track_list.append(y_track)
    return np.array(y_track_list)

def generate_traj_time(time_mean,time_std,traj_len,num_traj=10):
    time_list = []
    for idx in range(num_traj):
        rand = np.random.uniform(-1.2,1.2)
        alpha = time_mean + time_std*rand
        new_time = np.linspace(0,alpha,traj_len)
        time_list.append(new_time)
    return np.array(time_list)

def concate_func(time_list, h_track, r_track):
    complete_demo = []
    for idx in range(len(time_list)):
        time = time_list[idx].reshape(-1,1)
        human = h_track[idx]
        robot = r_track[idx]
        temp = np.column_stack((time,human,robot))
        complete_demo.append(complete_demo)
    return np.array(complete_demo)

norm_data_list = joblib.load(os.path.join(data_path,'ipromp_datasets_norm.pkl'))
demo = norm_data_list[0] #task 0
idx = 0
num_traj = 10
human_demo = demo[idx]["left_hand"]
robot_demo = demo[idx]["left_joints"]

h_mean_start,h_mean_goal,h_std_start,h_std_goal,r_mean_start,r_mean_goal,r_std_start,r_std_goal ,time_mean,time_std = get_demo_var(demo)
print "robot std"
print r_std_start[0:3]
print r_std_goal[0:3]
print "human std"
print h_std_start[0:3]
print h_std_goal[0:3]

r_std_start[0:3] = np.array([ 0.00136225,  0.00095215,  0.00051284])
r_std_goal[0:3] = np.array([ 0.00963589,  0.0518944,  0.0609559 ])
h_std_start[0:3] = np.array([ 0.00362726,  0.0138945,   0.01790308])
h_std_goal[0:3] = np.array([ 0.0191868,   0.0153769,  0.01904192])

# ipdb.set_trace()
h_mean_demo = dmp_generate_w_mean(human_demo, h_mean_start, h_mean_goal)
r_mean_demo = dmp_generate_w_mean(robot_demo, r_mean_start, r_mean_goal)

h_new_start_list, h_new_goal_list = generate_new_start_goal(h_mean_start,h_mean_goal,h_std_start,h_std_goal,num_traj=num_traj)
r_new_start_list, r_new_goal_list = generate_new_start_goal(r_mean_start,r_mean_goal,r_std_start,r_std_goal,num_traj=num_traj)

h_track = generate_new_demo(h_mean_demo,h_new_start_list, h_new_goal_list)
r_track = generate_new_demo(r_mean_demo,r_new_start_list, r_new_goal_list)
traj_len = r_track.shape[1]
time_list = generate_traj_time(time_mean,time_std,traj_len,num_traj=num_traj)
# final_data_list = concate_func(time_list, h_track, r_track)

plot_old_demo(demo,num=0,color="grey")
plot_new_demo(h_track,num=0,color="r",label="robot")
plot_new_demo(r_track,num=0,color="r",label="robot")
plt.show()
ipdb.set_trace()
joblib.dump(final_data_list,os.path.join(data_path,"handover_data_fake"))
