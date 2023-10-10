import os
import sys
import time
import cv2
import numpy as np
import argparse
import json
import threading
from random import random
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from collections import deque

# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi    # access the robot
import util.DatasetHandler as dh    # save/load functions
import util.measure as measure      # measurements
import pygame                       # python package for GUI
import shutil                       # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import YOLO components 
from YOLO.detector import Detector

# Import Operate
from operate_m4 import Operate

# Import rrt
from rrt import *

# Import auto_fruit_search_lvl2_slam.py
from auto_fruit_search_lvl2_slam import *


def run_operate(operate, args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default='party_room.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument('--slam-est', type=str, default='lab_output/slam.txt') # change
    parser.add_argument('--target-est', type=str, default='lab_output/targets.txt') # change
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='YOLO/model/yolov8_model_2.pt')
    args, _ = parser.parse_known_args()
    
    while start:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()
        print(operate.get_robot_pose())
 
def run_fruit_search(operate, args, pibot):
    pibot = PenguinPi(args.ip,args.port)
    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]
    startpos = (0., 0.)
    obstacles = np.concatenate((fruits_true_pos, aruco_true_pos))
    n_iter = 100
    radius = 0.14
    stepSize = 0.6
    for target in coords_order: #loop for every shopping list target
        #print(target)
        endpos = (target[0]+0.2, target[1]+0.2)# need to add so it is not targeting exactly on fruit?
        #print(endpos)

        ## RRT star
        print(obstacles)
        G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)

        # If path available
        if G.success:
            path = dijkstra(G)
            print(path)
            #plot(G, obstacles, radius, path)
        else:
            print("No path available")
        
        # drive robot
        for drive in path:
            robot_pose = operate.get_robot_pose() 
            print("Robot pose: ", robot_pose)
            waypoint = drive # setting each waypoint in the path
            operate.drive_to_point(waypoint,robot_pose)
            lv, rv = operate.pibot.set_velocity([0, 0])

        #Stop for 2 seconds
        print("Arrived at target")
        time.sleep(5)
        
        # move startpos to endpos now
        startpos = endpos

    exit() # exit code once at the end


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default='party_room.txt') # CHANGE to Truemap.txt
    parser.add_argument('--slam-est', type=str, default='lab_output/slam.txt') # change
    parser.add_argument('--target-est', type=str, default='lab_output/targets.txt') # change
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='YOLO/model/yolov8_model_2.pt')
    args, _ = parser.parse_known_args()

    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2023 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                     pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter % 10 // 2], (x_, 565))
            pygame.display.update()
            counter += 2

    pibot = PenguinPi(args.ip,args.port)
    operate = Operate(args) # add args

    wheel_vel = 30

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

    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    ekf = EKF(robot)
    aruco_det = aruco.aruco_detector(robot, marker_length = 0.06)

    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2023 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                     pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()


    #Move this section into main of auto_fruit_search
    #merge_aruco_fruit(args.slam, args.target)
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map) # CHANGE to output of merge_aruco_fruit
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    coords_order = target_fruits_pos_order(search_list, fruits_list, fruits_true_pos) # order of coords to go to for each fruit

    operate.add_markers(aruco_true_pos)
    operate.run_slam() # initializes slam
    
    print("SLAM is on? ", operate.ekf_on)

    startpos = (0., 0.)
    obstacles = np.concatenate((fruits_true_pos, aruco_true_pos))  # merging list of obstacles together (Aruco markers and Fruits)

    
    # Change these values as needed
 
    # creating threads
    t1 = threading.Thread(target=run_operate, args = (operate,args, ), name='operate_slam')
    t2 = threading.Thread(target=run_fruit_search, args = (operate,args, pibot, ), name='auto_fruit_search')
 
    # starting threads
    t1.start()
    t2.start()
 