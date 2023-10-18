# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
from copy import deepcopy
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
        if tag <= 10:
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
                break
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
                break
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

    # Ensure angle is capped
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


    # Position
    robot_pose = get_robot_pose()

    x_waypoint = waypoint[0]
    y_waypoint = waypoint[1]
    theta_waypoint = angle_orientation[0]

    print("SLAMS thinks we arrived at [{}, {}, {}]".format(robot_pose[0], robot_pose[1], robot_pose[2]))
    print("Waypoint was at [{}, {}, {}] \n \n".format(x_waypoint, y_waypoint, theta_waypoint))

def get_robot_pose():
    # update the robot pose [x,y,theta]
    robot_pose = operate.ekf.robot.state
    return robot_pose

def drive_update_slam(command,wheel_vel,turn_time):
    lv,rv = 0.0,0.0
    # Make sure it is moving
    if not (command[0] == 0 and command[1] == 0): 
        if command[0] == 0:
            # Turning
            lv,rv = ppi.set_velocity(command, turning_tick=wheel_vel, time=turn_time)
        else: 
            # Driving Straight
            lv,rv = ppi.set_velocity(command, tick=wheel_vel, time=turn_time)  
        # Operate commands  
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

    backtrack_path = []

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

    # Calibration Values
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

    # Initialise Operate
    operate = Operate(args)
    
    # Run SLAM (from operate)
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


    # Add landmarks from SLAM run
    operate.add_markers(aruco_true_pos)

    # Parameters for RRT*
    startpos = (0., 0.)
    obstacles = np.concatenate((fruits_true_pos, aruco_true_pos))  # merging list of obstacles together (Aruco markers and Fruits)
    
    n_iter = 300
    radius = 0.22 
    stepSize = 0.5

    num_of_fruits = len(coords_order)
    fruits_found = 0
    
    # Planning and pathing to fruit search list
    while fruits_found < num_of_fruits: # loop for every shopping list target
        target = coords_order[fruits_found]
        need_to_backtrack = False
        
        # Making sure not driving straight onto fruit
        fruit_threshold = 0.1
        if target[0] > 0:
            newx = target[0] - fruit_threshold
        else:
            newx = target[0] + fruit_threshold

        if target[1] > 0:
            newy = target[1] - fruit_threshold
        else:
            newy = target[1] + fruit_threshold

        endpos = (newx, newy)
        print('endpos: ', endpos)
        print('startpos: ', startpos)

        ## RRT star
        G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)

        ## Backtracking to previous waypoints
        iter_fail = 0
        fail_tolerance = 4
        num_of_paths_to_check = 10

        # Checking if path planning is successful
        while not G.success:
            if iter_fail < fail_tolerance:
                G = RRT_star(startpos, endpos, obstacles, 100*iter_fail + n_iter, radius, stepSize)
                iter_fail += 1
            else:
                need_to_backtrack = True
                break
        
        # Backtracking
        if need_to_backtrack:
            path = [startpos]
            backtrack_path.reverse()
            for point in backtrack_path[1:]:
                G = RRT_star(point, endpos, obstacles, n_iter, radius, stepSize)
                path.append(point)
                if G.success:
                    break
        else:
            # Choosing best RRT path planned
            minpath = dijkstra(G)
            min_waypoints = len(minpath)
            for _ in range(num_of_paths_to_check):
                G = RRT_star(startpos, endpos, obstacles, 100*iter_fail + n_iter, radius, stepSize)
                if G.success:
                    new_path = dijkstra(G)
                    if len(new_path) < min_waypoints:
                        minpath = new_path
                        min_waypoints = len(new_path)
                print(f"Run:{_+1}, len: {len(new_path)} with iters {100*iter_fail + n_iter} ")

            path = minpath
            backtrack_path = deepcopy(path)
            fruits_found += 1
            plot(G, obstacles, radius, path)

        
        # driving robot to each point in path
        for drive in path[1:]:
            robot_pose = get_robot_pose() 
            print("Robot pose: ", robot_pose)
            waypoint = drive # setting each waypoint in the path
            drive_to_point(waypoint,robot_pose)

            lv,rv = ppi.set_velocity([0, 0], turning_tick=0.0, time=0.0)
            drive_meas = measure.Drive(lv, -rv, 0.0)
            operate.update_slam(drive_meas)

        # Stop at target
        print("Arrived at target")
        time.sleep(4) # stop for at least 2 seconds
        robot_pose = get_robot_pose()
        x, y = robot_pose[0], robot_pose[1]
        print(robot_pose)
        
        # Move startpos to endpos
        startpos = (float(x), float(y))

    # End Run
    print("Finished auto fruit search!")
    time.sleep(5)
    exit()
