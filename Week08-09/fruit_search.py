# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
from rrt import *

from operate import Operate

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure

### New Merge SLAM and TARGET txt
def merge_aruco_fruit(slamtxt, targettxt):
    with open(slamtxt, "r") as file1:
        usr_dict = json.load(file1)

    aruco_dict = {}
    for (i, tag) in enumerate(usr_dict['taglist']):
        aruco_dict[tag] = np.reshape([usr_dict['map'][0][i], usr_dict['map'][1][i]], (2, 1))

    formatted_dict = {}
    for tag, values in aruco_dict.items():
        key = f"aruco{tag}_0"
        formatted_dict[key] = {"y": values[1][0], "x": values[0][0]}

    with open(targettxt, "r") as file2:
        data2 = json.load(file2)

    full_data = {**formatted_dict, **data2}
    with open("Full_Map.txt", "w") as output_file:
        json.dump(full_data, output_file, indent=4)

def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

    @param fname: filename of the map
    @return:
        1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
        2) locations of the targets, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1

def target_fruits_pos_order(search_list, fruit_list, fruit_true_pos):
    coords_order = []
    for fruit in search_list:
        for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                x = np.round(fruit_true_pos[i][0], 1)
                y = np.round(fruit_true_pos[i][1], 1)
                coords = (x,y)
                coords_order.append(coords)
                continue
    return coords_order

# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
# fully automatic navigation:
# try developing a path-finding algorithm that produces the waypoints automatically
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',') # m/tick
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',') # m
    
####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point
    wheel_vel = 30 # tick

    xpos = waypoint[0] - robot_pose[0]
    ypos = waypoint[1] - robot_pose[1]

    angle_orientation = np.arctan2(ypos, xpos)
    turning_angle = angle_orientation - robot_pose[2]
    print("+++++++++++++++++++++++++++++++++++++",turning_angle)

    # turn towards the waypoint
    turning_angle = (turning_angle + np.pi) % (2 * np.pi) - np.pi
    #print("turning angle",turning_angle)

    # may not need
    if turning_angle > np.pi:
        turning_angle -= 2 * np.pi
    elif turning_angle < -np.pi:
        turning_angle += 2 * np.pi

    linear_vel = wheel_vel*scale
    angular_vel = linear_vel*2 / baseline
    turn_time = abs(turning_angle) / angular_vel
    turn_time = float(turn_time)
    print("turn time is", turn_time)
    print("Turning for {:.2f} seconds".format(turn_time))
    if turning_angle > 0:
        #lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
        drive_update_slam([0,1],wheel_vel,turn_time)
    else:
        #lv, rv = ppi.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
        drive_update_slam([0,-1],wheel_vel,turn_time)
        
    # after turning, drive straight to the waypoint
    dist_to_point = np.sqrt((waypoint[0]-robot_pose[0])**2+(waypoint[1]-robot_pose[1])**2)
    drive_time = dist_to_point/(scale*wheel_vel) 
    drive_time = float(drive_time)
    print("Driving for {:.2f} seconds".format(drive_time))
    #lv, rv = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    drive_update_slam([1,0],wheel_vel,drive_time)


    # Correction step
    robot_pose = get_robot_pose()

    x_robot_pose = robot_pose[0]
    y_robot_pose = robot_pose[1]
    theta_robot_pose = robot_pose[2]

    x_waypoint = waypoint[0]
    y_waypoint = waypoint[1]
    theta_waypoint = angle_orientation[0]

    print("SLAMS thinks we arrived at [{}, {}, {}]".format(robot_pose[0], robot_pose[1], robot_pose[2]))
    print("Waypoint was at [{}, {}, {}] \n \n".format(x_waypoint, y_waypoint, theta_waypoint))

    x_diff = np.mod(x_robot_pose, x_waypoint)
    y_diff = np.mod(y_robot_pose, y_waypoint)

    #TODO Add thresholding/correction step ???

    #theta_diff = np.mod(theta_robot_pose, theta_waypoint)

    # threshold = 0.05
    # angle_threshold = 
    # if 


def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    robot_pose = operate.ekf.robot.state

    ####################################################

    return robot_pose

def drive_update_slam(command,wheel_vel,turn_time):
    lv,rv = 0.0,0.0
    if not (command[0] == 0 and command[1] == 0): 
        if command[0] == 0:
            lv,rv = ppi.set_velocity(command, turning_tick=wheel_vel, time=turn_time)
        else: # 
            lv,rv = ppi.set_velocity(command, tick=wheel_vel, time=turn_time)    
        operate.take_pic()
        drive_meas = measure.Drive(lv, -rv, turn_time)
        operate.update_slam(drive_meas)

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='Full_Map.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument('--slam', type=str, default='lab_output/slam.txt') # change
    parser.add_argument('--target', type=str, default='lab_output/targets.txt') # change
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)

    # For Operate
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='YOLO/model/yolov8_model_2.pt')
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")


    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)


    #TODO: Read in merge_estimations to generate targets.txt
    print('1')
    # read in the true map
    merge_aruco_fruit(args.slam, args.target)
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    coords_order = target_fruits_pos_order(search_list, fruits_list, fruits_true_pos) # order of coords to go to for each fruit

    print('2')
    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    clock = time.time()
    datadir = "calibration/param/"
    fileK = "{}intrinsic.txt".format(datadir)
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    fileD = "{}distCoeffs.txt".format(datadir)
    dist_coeffs = np.loadtxt(fileD, delimiter=',')
    fileS = "{}scale.txt".format(datadir)
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "{}baseline.txt".format(datadir)
    baseline = np.loadtxt(fileB, delimiter=',')

    operate = Operate(args)
    
    # run SLAM (from operate)
    n_observed_markers = len(operate.ekf.taglist)
    if n_observed_markers == 0:
        if not operate.ekf_on:
            print('SLAM is running')
            operate.ekf_on = True
        else:
            print('> 2 landmarks is required for pausing')
    elif n_observed_markers < 3:
        print('> 2 landmarks is required for pausing')
    else:
        if not operate.ekf_on:
            operate.request_recover_robot = True
        operate.ekf_on = not operate.ekf_on
        if operate.ekf_on:
            print('SLAM is running')
        else:
            print('SLAM is paused')



    operate.add_markers(aruco_true_pos)

    ### use one or the other

    #lms=[]
    # for i,lm in enumerate(aruco_true_pos):
    #     measure_lm = measure.Marker(np.array([[lm[0]],[lm[1]]]),i+1)
    #     lms.append(measure_lm)
    # operate.ekf.add_landmarks(lms) 


    startpos = (0., 0.)

    #TODO Add obstacle around arena

    obstacles = np.concatenate((fruits_true_pos, aruco_true_pos))  # merging list of obstacles together (Aruco markers and Fruits)

    
    #TODO Test and change these values as needed
    n_iter = 300
    radius = 0.18
    stepSize = 0.5

    for target in coords_order: #loop for every shopping list target


        # DONE: Change endpos +- depending on which quadrant target is in
        if target[0]>0:
            newx = target[0] - 0.2
        else:
            newx = target[0] + 0.2

        if target[1] > 0:
            newy = target[1] - 0.2
        else:
            newy = target[1] + 0.2

        endpos = (newx, newy)
        print('endpos: ', endpos)
        print('startpos: ', startpos)

        ## RRT star
        #print(obstacles)
        G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)

        # DONE: increase n_iter with num of failures
        # If path available # DONE: regenerate paths until success; ,maybe implement go to 0,0 if unsuccessful after 20 iterations
        iter_fail = 1
        while not G.success:
            print('startpos: ', startpos, 'endpos: ', endpos, 'obstacles: ', obstacles, 'n_iter: ', n_iter, 'radius: ',
                  radius, 'stepSize:', stepSize)
            G = RRT_star(startpos, endpos, obstacles, 100*iter_fail + n_iter, radius, stepSize)
            plot(G, obstacles, radius, path)
            iter_fail += 1

        path = dijkstra(G)
        plot(G, obstacles, radius, path)
        
        # drive robot
        for drive in path[2:]:
            robot_pose = get_robot_pose() # TODO: not updating correctly?
            print("Robot pose: ", robot_pose)
            waypoint = drive # setting each waypoint in the path
            drive_to_point(waypoint,robot_pose)

            lv,rv = ppi.set_velocity([0, 0], turning_tick=0.0, time=0.0)
            drive_meas = measure.Drive(lv, -rv, 0.0)
            operate.update_slam(drive_meas)

        # Check distance from fruit
        xpos = endpos[0] - robot_pose[0]
        ypos = endpos[1] - robot_pose[1]
        dist_to_point = np.sqrt((xpos)**2+(ypos)**2)
        
        print("Distance to waypoint is: ", dist_to_point)
        
        # Check if within 5 cm
        # if dist_to_point > 0.5:
        #     drive_to_point(endpos,robot_pose)
                #TODO Run RRT_STAR Again

        # Stop for 2 seconds
        print("Arrived at target")
        time.sleep(4)
        robot_pose = get_robot_pose()
        x, y = robot_pose[0], robot_pose[1]
        print(robot_pose)
        
        # Move startpos to endpos now
        startpos = (float(x), float(y))

    print("Finished auto fruit search!")
    time.sleep(5)
    exit()
