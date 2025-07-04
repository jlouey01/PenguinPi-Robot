# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time

#from operate import *

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure


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

    # turn towards the waypoint
    turning_angle = (turning_angle + np.pi) % (2 * np.pi) - np.pi
    #print("turning angle",turning_angle)

    # may not need
    # if turning_angle > np.pi:
    #     turning_angle -= 2 * np.pi
    # elif turning_angle < -np.pi:
    #     turning_angle += 2 * np.pi

    linear_vel = wheel_vel*scale
    angular_vel = linear_vel*2 / baseline
    turn_time = abs(turning_angle) / angular_vel
    turn_time = float(turn_time)
    print("turn time is", turn_time)
    print("Turning for {:.2f} seconds".format(turn_time))
    if turning_angle > 0:
        ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    else:
        ppi.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
        
    # after turning, drive straight to the waypoint
    dist_to_point = np.sqrt((waypoint[0]-robot_pose[0])**2+(waypoint[1]-robot_pose[1])**2)
    drive_time = dist_to_point/(scale*wheel_vel) 
    drive_time = float(drive_time)
    print("Driving for {:.2f} seconds".format(drive_time))
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

    robot_pose = [waypoint[0], waypoint[1], angle_orientation]

    return robot_pose

# def get_robot_pose(drive_meas):
#     ####################################################
#     # TODO: replace with your codes to estimate the pose of the robot
#     # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

#     # aruco_det = aruco.aruco_detector(
#     #     ekf.robot, marker_length=0.07)

#     # dt = time.time() - clock
    
#     # drive_meas = measure.Drive(wheel_vel, -wheel_vel, dt)
#     # clock = time.time()

#     # lms, _ = aruco_det.detect_marker_positions(ppi.get_image())
#     # ekf.predict(drive_meas)
#     # ekf.add_landmarks(lms)
#     # ekf.update(lms)
#     # #state = ekf.get_state_vector()
#     # #x, y, theta = state
#     # # update the robot pose [x,y,theta]
#     # #robot_pose = [x, y, theta] # replace with your calculation
#     # ####################################################
#     # state = ekf.get_state_vector()
#     # x, y, theta = state[0], state[1], state[2]
#     # robot_pose = [x, y, theta]
#     # #print(robot_pose)
#     # return robot_pose, clock
#     img = ppi.get_image()

#     landmarks, _ = aruco_det.detect_marker_positions(img)
#     ekf.predict(drive_meas)
#     ekf.update(landmarks)

#     full_circle = 2*np.pi
#     robot_pose = ekf.robot.state

#     if robot_pose[2][0] < 0:
#         robot_pose[2][0] = robot_pose[2][0] + full_circle
#     elif robot_pose[2][0] > full_circle:
#         robot_pose[2][0] = robot_pose[2][0] - full_circle

#     return robot_pose

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # read in the true map
    #fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    #search_list = read_search_list()
    #print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

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

    wheel_vel = 30

    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    ekf = EKF(robot)
    aruco_det = aruco.aruco_detector(robot, marker_length = 0.06)


    #Move this section into main of auto_fruit_search
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    coords_order = target_fruits_pos_order(search_list, fruits_list, fruits_true_pos) # order of coords to go to for each fruit


    # The following is only a skeleton code for semi-auto navigation
    while True:
        # enter the waypoints
        # instead of manually enter waypoints, you can give coordinates by clicking on a map, see camera_calibration.py from M2
        
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        # estimate the robot's pose
        #get_robot_pose(drive_meas)
        print(robot_pose)
        

        # robot drives to the waypoint
        waypoint = [x,y]
        robot_pose = drive_to_point(waypoint,robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break
