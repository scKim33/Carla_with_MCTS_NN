#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.
# This code is modified by MK.
# It has been tested in python 3.8
"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    Y            : toggle autopilot // change! (P -> Y)
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk
    T            : toggle vehicle's telemetry

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
from shutil import which
import sys
import struct
import ctypes
import pcl
import time

try:
    # sys.path.append("./carla-0.9.12-binary_seg.egg")
    sys.path.append("./carla-0.9.12-py3.8-linux-x86_64_with_full_seg.egg")
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ WORK!")
except IndexError:
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ NOTWORK!")
    pass
import carla


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
from PIL import ImageDraw
from PIL import Image as PILIMAGE
import cv2
import cv_bridge

# ROS
import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, PoseArray, PoseWithCovarianceStamped, PoseWithCovariance
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Bool, String, Header, Int32
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs.point_cloud2 import create_cloud
# from carla_scripts.msg import PoseWithCovarianceArray # custom msg
import rospkg
rospack = rospkg.RosPack()
# rospack.list() 

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_e
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_u
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_y
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Scenario Reader -----------------------------------------------------------
# ==============================================================================
# import pickle
import pickle5 as pickle

M_SCENARIO = '0'
M_CASE = 45
M_mini_scenario = None
M_Initial_X = None
M_Initial_Y = None
M_Initial_Angle = None
M_Obstacle_X = None
M_Obstacle_Y = None
M_Goal_X = None
M_Goal_Y = None
M_Goal_Angle = None
M_Obstacle_Exist = True
M_Space_Width = 5.0


# At below, the candidate poses for the empty target parking spot.. only for perpendicular parking
# m_candidate_pose_set = [(-59.40, 252.21, 0), (-59.40, 256.80, 0), (-59.40, 261.41, 0), (-59.40, 268.31, 0), (-59.40, 270.61, 0), (-59.40, 275.21, 0), (-59.40, 279.81, 0), (-59.40, 282.11, 0), (-59.40, 295.95, 0),
#                         (-48.5, 254.21, 180), (-48.50, 261.11, 180), (-48.50, 265.71, 180), (-48.50, 268.01, 180), (-48.50, 274.91, 180), (-48.50, 279.51, 180), (-48.50, 286.51, 180), (-48.50, 293.31, 180),
#                         (-59.40, 319.75, 0), (-59.40, 328.85, 0), (-48.5, 327.05, 180), (-48.5, 336.25, 180), (-58.60, 341.11, -90), (-63.30, 341.11, -90), (-67.80, 341.11, -90),
#                         (-77.70, 322.01, 0), (-77.70, 315.11, 0), (-66.7, 317.51, 180), 
#                         (-77.7, 243.11, 0), (-77.7, 249.91, 0), (-77.7, 252.21, 0), (-77.7, 266.05, 0), (-77.7, 268.35, 0), (-77.7, 275.25, 0), (-77.7, 284.45, 0), (-77.7, 286.75, 0), (-77.7, 298.25, 0),
#                         (-66.7, 254.55, 180), (-66.7, 263.65, 180), (-66.7, 277.55, 180), (-66.7, 279.86, 180), (-66.7, 293.66, 180), (-66.7, 295.96, 180), (-66.7, 302.86, 180),
#                         (-53.90, 360.56, -90), (-60.80, 360.56, -90), (-65.4, 360.56, -90), (-72.4, 360.56, -90), (-77.1, 360.56, -90)]
obs_candidate_idx = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
goal_candidate_idx = [18, 19, 20, 21, 22, 23, 24, 25, 122, 123, 124, 125, 126, 127, 128, 129]
m_candidate_pose_set = [(-77.8, 238.45, 0.0), # 1
                        (-77.8, 240.75, 0.0),
                        (-77.8, 243.05, 0.0),
                        (-77.8, 245.35, 0.0),
                        (-77.8, 247.65, 0.0),
                        (-77.8, 249.95, 0.0),
                        (-77.8, 252.25, 0.0),
                        (-77.8, 254.55, 0.0),
                        (-77.8, 256.85, 0.0),
                        (-77.8, 259.15, 0.0),
                        (-77.8, 261.45, 0.0),
                        (-77.8, 263.75, 0.0),
                        (-77.8, 266.05, 0.0),
                        (-77.8, 268.35, 0.0),
                        (-77.8, 270.65, 0.0),
                        (-77.8, 272.95, 0.0),
                        (-77.8, 275.25, 0.0),
                        (-77.8, 277.55, 0.0),
                        (-77.8, 279.85, 0.0),
                        (-77.8, 282.15, 0.0),
                        (-77.8, 284.45, 0.0),
                        (-77.8, 286.75, 0.0),
                        (-77.8, 289.05, 0.0),
                        (-77.8, 291.35, 0.0),
                        (-77.8, 293.65, 0.0),
                        (-77.8, 295.95, 0.0),
                        (-77.8, 298.25, 0.0),
                        (-77.8, 300.55, 0.0),
                        (-77.8, 302.85, 0.0), # fixed
                        (-77.8, 305.15, 0.0), # 30

                        (-77.8, 312.85, 0.0), # 31
                        (-77.8, 315.15, 0.0),
                        (-77.8, 317.45, 0.0),
                        (-77.8, 319.75, 0.0),
                        (-77.8, 322.05, 0.0),
                        (-77.8, 324.35, 0.0),
                        (-77.8, 326.65, 0.0),
                        (-77.8, 328.95, 0.0),
                        (-77.8, 331.25, 0.0),
                        (-77.8, 333.55, 0.0), # 40


                        (-54.1, 341.11, -90.0), # 49
                        (-56.4, 341.11, -90.0),
                        (-58.7, 341.11, -90.0),
                        (-61.0, 341.11, -90.0), # fixed
                        (-63.3, 341.11, -90.0),
                        (-65.6, 341.11, -90.0),
                        (-67.9, 341.11, -90.0),
                        (-70.2, 341.11, -90.0),
                        (-72.5, 341.11, -90.0), # 41


                        (-48.5, 254.22, 180.0), # 75
                        (-48.5, 256.52, 180.0),
                        (-48.5, 258.82, 180.0),
                        (-48.5, 261.12, 180.0),
                        (-48.5, 263.42, 180.0),
                        (-48.5, 265.72, 180.0),
                        (-48.5, 268.02, 180.0),
                        (-48.5, 270.32, 180.0),
                        (-48.5, 272.62, 180.0), # fixed
                        (-48.5, 274.92, 180.0),
                        (-48.5, 277.22, 180.0),
                        (-48.5, 279.52, 180.0),
                        (-48.5, 281.82, 180.0),
                        (-48.5, 284.12, 180.0),
                        (-48.5, 286.42, 180.0),
                        (-48.5, 288.72, 180.0),
                        (-48.5, 291.02, 180.0),
                        (-48.5, 293.32, 180.0), # 58

                        (-48.5, 320.12, 180.0), # 57 fixed
                        (-48.5, 322.42, 180.0),
                        (-48.5, 324.72, 180.0),
                        (-48.5, 327.02, 180.0),
                        (-48.5, 329.32, 180.0),
                        (-48.5, 331.62, 180.0),
                        (-48.5, 333.92, 180.0),
                        (-48.5, 336.22, 180.0), # 50


                        (-59.5, 247.65, 0.0), # 76
                        (-59.5, 249.95, 0.0),
                        (-59.5, 252.25, 0.0),
                        (-59.5, 254.55, 0.0),
                        (-59.5, 256.85, 0.0),
                        (-59.5, 259.15, 0.0),
                        (-59.5, 261.45, 0.0),
                        (-59.5, 263.75, 0.0),
                        (-59.5, 266.05, 0.0),
                        (-59.5, 268.35, 0.0),
                        (-59.5, 270.65, 0.0),
                        (-59.5, 272.95, 0.0),
                        (-59.5, 275.25, 0.0),
                        (-59.5, 277.55, 0.0),
                        (-59.5, 279.85, 0.0),
                        (-59.5, 282.15, 0.0),
                        (-59.5, 284.45, 0.0),
                        (-59.5, 286.75, 0.0),
                        (-59.5, 289.05, 0.0),
                        (-59.5, 291.35, 0.0),
                        (-59.5, 293.65, 0.0), # fixed
                        (-59.5, 295.95, 0.0),
                        (-59.5, 298.25, 0.0),
                        (-59.5, 300.55, 0.0),
                        (-59.5, 302.85, 0.0),
                        (-59.5, 305.15, 0.0), # 101

                        (-59.5, 312.85, 0.0), # 102
                        (-59.5, 315.15, 0.0),
                        (-59.5, 317.45, 0.0),
                        (-59.5, 319.75, 0.0),
                        (-59.5, 322.05, 0.0),
                        (-59.5, 324.35, 0.0),
                        (-59.5, 326.65, 0.0),
                        (-59.5, 328.95, 0.0), # 109


                        (-66.7, 247.65, 180.0), # 143
                        (-66.7, 249.95, 180.0),
                        (-66.7, 252.25, 180.0),
                        (-66.7, 254.55, 180.0),
                        (-66.7, 256.85, 180.0),
                        (-66.7, 259.15, 180.0),
                        (-66.7, 261.45, 180.0),
                        (-66.7, 263.75, 180.0),
                        (-66.7, 266.05, 180.0),
                        (-66.7, 268.35, 180.0),
                        (-66.7, 270.65, 180.0), # fixed
                        (-66.7, 272.95, 180.0),
                        (-66.7, 275.25, 180.0),
                        (-66.7, 277.55, 180.0),
                        (-66.7, 279.85, 180.0),
                        (-66.7, 282.15, 180.0),
                        (-66.7, 284.45, 180.0),
                        (-66.7, 286.75, 180.0),
                        (-66.7, 289.05, 180.0),
                        (-66.7, 291.35, 180.0),
                        (-66.7, 293.65, 180.0),
                        (-66.7, 295.95, 180.0),
                        (-66.7, 298.25, 180.0),
                        (-66.7, 300.55, 180.0),
                        (-66.7, 302.85, 180.0),
                        (-66.7, 305.15, 180.0), # 118 

                        (-66.7, 312.85, 180.0), # 117
                        (-66.7, 315.15, 180.0),
                        (-66.7, 317.45, 180.0),
                        (-66.7, 319.75, 180.0),
                        (-66.7, 322.05, 180.0),
                        (-66.7, 324.35, 180.0),
                        (-66.7, 326.65, 180.0),
                        (-66.7, 328.95, 180.0), # 110


                        (-47.0, 360.56, -90.0), # 144
                        (-49.3, 360.56, -90.0),
                        (-51.6, 360.56, -90.0),
                        (-53.9, 360.56, -90.0),
                        (-56.2, 360.56, -90.0),
                        (-58.5, 360.56, -90.0),
                        (-60.8, 360.56, -90.0), # fixed
                        (-63.1, 360.56, -90.0),
                        (-65.4, 360.56, -90.0),
                        (-67.7, 360.56, -90.0),
                        (-70.0, 360.56, -90.0),
                        (-72.3, 360.56, -90.0),
                        (-74.6, 360.56, -90.0),
                        (-76.9, 360.56, -90.0),
                        (-79.2, 360.56, -90.0) # 158
                        ]

test_mcts_driving_pose_set = [(-100.0, 360.0, 90),
                              (-100.0, 380.0, 90),
                              (-120.0, 400.0, 180),
                              (-140.0, 420.0, 90)
                              ]
                        
# At below, this is for the empty target 'parallel' parking spot
m_candidate_pose_set2 = [(-36.25, 296.4, 90.0), (-36.25, 290.2, 90.0), (-36.25, 284.6, 90.0), (-36.25, 278.8, 90.0), (-36.25, 273.0, 90.0), (-36.25, 267.4, 90.0), (-36.25, 261.9, 90.0),
                         (-36.25, 255.4, 90.0), (-36.25, 249.7, 90.0), (-36.25, 243.8, 90.0)] # near the building
m_candidate_pose_set3 = [(-42.45, 296.4, 90.0), (-42.45, 290.2, 90.0), (-42.45, 284.6, 90.0), (-42.45, 278.8, 90.0), (-42.45, 273.0, 90.0), (-42.45, 267.4, 90.0), (-42.45, 261.9, 90.0),
                         (-42.45, 255.4, 90.0), (-42.45, 249.7, 90.0), (-42.45, 243.8, 90.0)] # near the parking lot

# At belsow, this is for the empty target 'circular' parking lots
m_candidate_pose_set4 = [(-86.9, 212.1, 150.0), (-89.81, 209.33, 120.0), (-93.65, 208.0, 90), (-97.45, 209.1, 60), (-100.3, 212.0, 30), (-101.5, 215.9, 0.0), 
                         (-100.5, 219.9, -30), (-97.5, 222.9, -60), (-93.5, 223.8, -90), (-89.75, 222.58, -120), (-86.65, 219.78, -150)]
m_candidate_start_pose4 = [(-86.75, 216.11, 0.0), (-93.8, 216.2, 0.0)] # the second one can have fully random orientation

# At below, arena min, max dimensions
m_arena_min_x = -107
m_arena_max_x = -86.4
m_arena_min_y = 243.0
m_arena_max_y = 299.4

# At below, cluterred road
m_candidate_pose_set5 = [(-77.1, 200.9, 0.0), (-64.0, 201.0, 0.0)] 
m_candidate_pose_set6 = [(-62.1, 201.0, 0.0), (-61.0, 214.0, 180.0)] 

bbox_extent = {0:[1.852685, 0.894339],
               2:[2.395890, 1.081725],
               3:[2.427854, 1.016378],
               11:[2.336819, 1.001146],
               14:[2.450842, 1.064162],
               15:[2.678740, 1.016601],
               24:[2.256761, 1.003407],
               28:[1.902900, 0.985138],
               29:[2.513388, 1.075773],
               32:[2.305503, 1.120857]
               }

def remainder(x: float, y: float) -> float:
    low = - y / 2.0
    high = y / 2.0
    if x <= low:
        return remainder(x + y, y)
    elif x > high:
        return remainder(x - y, y)
    else: return x
               
def InitializeScenario(scenario_num, case_num):
    global M_SCENARIO, M_CASE, M_Initial_X, M_Initial_Y, M_Initial_Angle, M_Obstacle_X, M_Obstacle_Y, M_Goal_X, M_Goal_Y, M_Goal_Angle, M_Obstacle_Exist
    M_SCENARIO = str(scenario_num)
    M_CASE = str(case_num)
    # dir_path = rospack.get_path('carla_scripts')
    # with open(dir_path + '/script/parking_lot_1.pckl', 'rb') as handle:
    #     scenario_case_table = pickle.load(handle)
    #     try:
    #         M_Initial_X = scenario_case_table['Scenario {}'.format(M_SCENARIO)]['case {}'.format(M_CASE)][0]
    #         M_Initial_Y = scenario_case_table['Scenario {}'.format(M_SCENARIO)]['case {}'.format(M_CASE)][1]
    #         M_Initial_Angle = scenario_case_table['Scenario {}'.format(M_SCENARIO)]['case {}'.format(M_CASE)][2]
    #         M_Obstacle_X = scenario_case_table['Scenario {}'.format(M_SCENARIO)]['case {}'.format(M_CASE)][3]
    #         M_Obstacle_Y = scenario_case_table['Scenario {}'.format(M_SCENARIO)]['case {}'.format(M_CASE)][4]
    #         M_Goal_X = scenario_case_table['Scenario {}'.format(M_SCENARIO)]['case {}'.format(M_CASE)][5]
    #         M_Goal_Y = scenario_case_table['Scenario {}'.format(M_SCENARIO)]['case {}'.format(M_CASE)][6]
    #         M_Goal_Angle = scenario_case_table['Scenario {}'.format(M_SCENARIO)]['case {}'.format(M_CASE)][7]          
    #     except:
    #         print("!!!!!!! Please, write down a valid scenario and case !!!!!!!")
    #         sys.exit(1)
    #     if (M_Obstacle_X == 0 and M_Obstacle_Y == 0):
    #         M_Obstacle_Exist = False
    print("\n @@@@@@@ Currenet scenario-case: {}--{}  ========================".format(M_SCENARIO, M_CASE))

def SpawnPoseAt(world, x, y, th):
    if not world.get_map().get_spawn_points():
        print('There are no spawn points available in your map/town.')
        print('Please add some Vehicle Spawn Point to your UE4 scene.')
        sys.exit(1)
    spawn_point = carla.Transform()
    spawn_point.location.x = x
    spawn_point.location.y = y
    spawn_point.rotation.yaw = th
    spawn_point.location.z += 1.0
    spawn_point.rotation.roll = 0.0
    spawn_point.rotation.pitch = 0.0

    return spawn_point 

def ScenarioSpawnPose(world):
    global M_Initial_X, M_Initial_Y, M_Initial_Angle, M_Goal_X, M_Goal_Y, M_Goal_Angle
    if not world.get_map().get_spawn_points():
        print('There are no spawn points available in your map/town.')
        print('Please add some Vehicle Spawn Point to your UE4 scene.')
        sys.exit(1)
    spawn_point = carla.Transform()
    spawn_point.location.x = M_Initial_X
    spawn_point.location.y = M_Initial_Y
    spawn_point.rotation.yaw = M_Initial_Angle
    spawn_point.location.z += 2.0
    spawn_point.rotation.roll = 0.0
    spawn_point.rotation.pitch = 0.0

    return spawn_point 

def SpawnObstacle(world, color = '200, 0, 0'):  
    global M_Obstacle_X, M_Obstacle_Y, M_Goal_Angle
    # vehicle_idx = 0
    # player_blueprint = world.get_blueprint_library().filter('vehicle.*')[vehicle_idx] #ms: get Tesla, pick a vehicle[19]
    player_blueprint = world.get_blueprint_library().find('vehicle.mercedes.coupe_2020')
    if player_blueprint.has_attribute('color'):
        player_blueprint.set_attribute('color', color)
    if player_blueprint.has_attribute('is_invincible'):
        player_blueprint.set_attribute('is_invincible', 'true')
    if not world.get_map().get_spawn_points():
        print('There are no spawn points available in your map/town.')
        print('Please add some Vehicle Spawn Point to your UE4 scene.')
        sys.exit(1)
    spawn_point = carla.Transform()
    spawn_point.location.x = M_Obstacle_X
    spawn_point.location.y = M_Obstacle_Y
    spawn_point.rotation.yaw = M_Goal_Angle + random.choice([90.0, -90.0])
    spawn_point.location.z += 2.0
    spawn_point.rotation.roll = 0.0
    spawn_point.rotation.pitch = 0.0
    obstacle = world.try_spawn_actor(player_blueprint, spawn_point)
    return obstacle

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
DEG2RAD = math.pi / 180.0
KMpH2MpS = 0.277778
M_VEHICLE_PHYSICS = {
    'Length': 4.263,
    'Width': 1.975
}


M_DIMENSION = 32 # [m]
M_RESOL = 0.2 # grid resolution. i.e., meter / grid
M_ACCURATE_GOAL = True

# forked from tf/transformations.py
# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# forked from tf/transformations.py
# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# forked from tf/transformations.py
# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


# forked from tf/transformations.py
def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)


# forked from tf/transformations.py
def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


# forked from tf/transformations.py
def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)

# def euler_from_quaternion(x, y, z, w):
#         """
#         Convert a quaternion into euler angles (roll, pitch, yaw)
#         roll is rotation around x in radians (counterclockwise)
#         pitch is rotation around y in radians (counterclockwise)
#         yaw is rotation around z in radians (counterclockwise)
#         """
#         t0 = +2.0 * (w * x + y * z)
#         t1 = +1.0 - 2.0 * (x * x + y * y)
#         roll_x = math.atan2(t0, t1)
     
#         t2 = +2.0 * (w * y - z * x)
#         t2 = +1.0 if t2 > +1.0 else t2
#         t2 = -1.0 if t2 < -1.0 else t2
#         pitch_y = math.asin(t2)
     
#         t3 = +2.0 * (w * z + x * y)
#         t4 = +1.0 - 2.0 * (y * y + z * z)
#         yaw_z = math.atan2(t3, t4)
     
#         return roll_x, pitch_y, yaw_z # in radians

def euler_to_quaternion(yaw: float, pitch: float, roll: float)->list:
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return qx, qy, qz, qw

def goal_std(dist_to_goal: float, mmax=10, mmin=4.5, stdmin=0.01, stdmax=1.0, k=2.0)->float:
    if dist_to_goal < mmin:
        return stdmin
    elif mmin<= dist_to_goal < mmax:
        return ((stdmax-stdmin) / (mmax-mmin)**k) * (dist_to_goal - mmin)**k + stdmin
    else:
        return stdmax     


def poseArray(x: float, y: float, theta: float)->Pose():
    local_pose = Pose()
    local_pose.position.x = x   # x [m]
    local_pose.position.y = y   # y [m]
    local_pose.position.z = +1  # It decides the extend direction of parking goal. Currently, only backward parking is considered.
    local_pose.orientation.x, local_pose.orientation.y, local_pose.orientation.z, local_pose.orientation.w =  euler_to_quaternion(theta, 0.0, 0.0)
    return local_pose

def poseWithCovArray(x: float, y: float, theta: float, std_x: float, std_y: float, std_th: float)->PoseWithCovariance():
    local_pose = PoseWithCovariance()
    local_pose.pose.position.x = x   # x [m]
    local_pose.pose.position.y = y   # y [m]
    local_pose.pose.position.z = +1  # It decides the extend direction of parking goal. Currently, only backward parking is considered.
    local_pose.pose.orientation.x, local_pose.pose.orientation.y, local_pose.pose.orientation.z, local_pose.pose.orientation.w =  euler_to_quaternion(theta, 0.0, 0.0)
    local_pose.covariance = [std_x**2, 0, 0, 0, 0, 0,
                             0, std_y**2, 0, 0, 0, 0,
                             0, 0,        0, 0, 0, 0,
                             0, 0, 0,        0, 0, 0,
                             0, 0, 0, 0,        0, 0,
                             0, 0, 0, 0, 0, std_th**2]
    return local_pose

def global2local(Gx, Gy, Gth, refX, refY, refTH):
    tmpX = Gx - refX
    tmpY = Gy - refY
    tmpth = Gth 
    Lx = -(tmpX * math.cos(tmpth) + tmpY * math.sin(tmpth))
    Ly = (tmpX * -math.sin(tmpth) + tmpY * math.cos(tmpth))
    # Lth = Gth - refTH if 0.0 <= Gth - refTH < 1.5 * math.pi else Gth - refTH + 2 * math.pi
    Lth = remainder(Gth - refTH, 2 * math.pi)
    return Lx, Ly, Lth

def coord(dx,dy, xx, yy, thh):
    yw_resol = (13/18)*(35/416)
    modx = math.cos(math.pi/2+thh)*dx - math.sin(math.pi/2+thh)*dy + xx
    mody = math.sin(math.pi/2+thh)*dx + math.cos(math.pi/2+thh)*dy + yy
    modx = 208+int(modx/yw_resol)#math.ceil
    mody = 208+int(mody/yw_resol)
    return (mody, modx)
    
def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)
    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# ==============================================================================
# -- AVMManager -------------------------------------------------------------
# ==============================================================================
class SegmentationAVMCamera(object):
    def __init__(self, parent_actor, hud, gamma_correction, camera_type, camera_loc):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.image = None
        self.image_l = None
        Attachment = carla.AttachmentType
        (lx, ly, lz, lroll, lpitch, lyaw) = camera_loc
        self._camera_transforms = (carla.Transform(carla.Location(x=lx, y=ly, z=lz), 
                                                   carla.Rotation(roll=lroll, pitch=lpitch, yaw=lyaw)), 
                                                   Attachment.Rigid)# default AVM setting

        self.seg_option = None
        if camera_type == 'rgb':
            self.seg_option = ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {'fov': str(160)}, {'sensor_tick': str(0.05)}, {'lens_flare_intensity': str(0.5)}]
        else:
            self.seg_option = ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                    'Camera Semantic Segmentation (CityScapes Palette)', {'fov': str(160)}, {'sensor_tick': str(0.05)}]

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        
        bp = bp_library.find(self.seg_option[0])
        if self.seg_option[0].startswith('sensor.camera'):
            if bp.has_attribute('gamma'):
                bp.set_attribute('gamma', str(gamma_correction))
            for attr_name, attr_value in self.seg_option[3].items():
                bp.set_attribute(attr_name, attr_value)
            for attr_name, attr_value in self.seg_option[4].items():
                bp.set_attribute(attr_name, attr_value)

            self.seg_option.append(bp)

    def set_sensor(self):
        if self.sensor is not None:
            self.sensor.destroy()
        self.sensor = self._parent.get_world().spawn_actor(
            self.seg_option[-1],
            self._camera_transforms[0],
            attach_to=self._parent,
            attachment_type=self._camera_transforms[1])

        # We need to pass the lambda a weak reference to self to avoid
        # circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: SegmentationAVMCamera._parse_image(weak_self, image))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        ind = 0        
        if not self:
            return
        
        image.convert(self.seg_option[1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        
        self.image = np.copy(array)
    
    def _destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================
class World(object):
    def __init__(self, client, carla_world, hud, args):
        rospy.init_node('carla_simulator')       
        self.client = client
        self.world = carla_world
        self.sync = args.sync
        self.args = args
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.obs1 = None
        self.lidar_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self.seg_avm_front_camera = None
        self.seg_avm_back_camera = None
        self.seg_avm_right_camera = None
        self.seg_avm_left_camera = None

        self.seg_avm_front_camera2 = None
        self.seg_avm_back_camera2 = None
        self.seg_avm_right_camera2 = None
        self.seg_avm_left_camera2 = None


        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.scenario_count = args.s
        self.case_count = args.c
        self.autonomous_mode = False
        self.scenario_ratio = [1.0, 0.0, 0.0, 0.0, 0.0] # perpendicular parking, parallel parking, circular parking, clutterred navigation, arena
        # self.scenario_ratio = [0.0, 0.25, 0.25, 0.25, 0.25] # perpendicular parking, parallel parking, circular parking, clutterred navigation, arena

        self.load_scenario_file = args.load
        self.cam_on = args.cam_on
        self.avm_on = args.avm_on
        print("::::: ", args.cam_on, args.log)

        self.rl_train_mode = args.rl
        
        self.gear_state = 'P'
        self.finish_call = False
        self.change_map = True
        self.current_pose_x = 0
        self.current_pose_y = 0
        self.current_pose_th = 0

        self.control_command_updated = 1

        # ============================= ROS related ============================= 
        # Publisher
        self.pubLocalizationData = rospy.Publisher('/LocalizationData', Float32MultiArray, queue_size = 1)
        self.pubSteerData = rospy.Publisher('/SteerData', Float32MultiArray, queue_size = 1)
        self.pubGearData = rospy.Publisher('/GearData', Int32, queue_size = 1)
        self.pubVelData = rospy.Publisher('CanVelData2', Float32MultiArray, queue_size = 1)    
        self.pubOccupancyGridMap = rospy.Publisher('true_occ_map', OccupancyGrid, queue_size=1)
        self.pubAVM = rospy.Publisher("/avm_usb_cam/image_raw", Image, queue_size = 1)
        self.pubParkingCand = rospy.Publisher('/parking_cands', PoseArray, queue_size=1)
        self.pub_carla_image = rospy.Publisher("/raw_carla_img", Image, queue_size=1)
        self.pub_carla_AVM = rospy.Publisher("/raw_carla_avm", Image, queue_size=1)
        self.pub_carla_AVMseg = rospy.Publisher("/seg_carla_img", Image, queue_size=1)
        self.pub_situation_info = rospy.Publisher("situation_info", Float32MultiArray, queue_size=1)
        self.pub_collision = rospy.Publisher("/collision", Int32, queue_size=1)
        self.pub_terminal_case = rospy.Publisher("/terminal_case", Int32, queue_size=1)
        self.pub_start_new_scenario = rospy.Publisher("/start_scenario", Int32, queue_size=1)
        self.pub_obs_poses = rospy.Publisher("/obs_poses", PoseArray, queue_size=1)
        self.pub_obs_info = rospy.Publisher("/obs_info", Float32MultiArray, queue_size=1)
        self.pub_goal_poses = rospy.Publisher("/goal_poses", PoseArray, queue_size=1)

        self.pubPose = rospy.Publisher('/vehicle_pose', Marker, queue_size=1)
        self.pubReplanning = rospy.Publisher('/replanning', Float32MultiArray, queue_size=1)
        self.pubPointClouds = rospy.Publisher('velodyne_points', PointCloud2, queue_size=1)

        # Subscriber
        self.subControlCommand = rospy.Subscriber('Control_Command', Float32MultiArray, self.control_callback)
        self.subFinishFlag = rospy.Subscriber('is_finish', Int32, self.finish_flag_call)
        self.subTrajLogFlag = rospy.Subscriber('traj_log_finish', Int32, self.finish_traj_log_call)
        self.subRestartFlag = rospy.Subscriber('restart_flag', Int32, self.restart_flag_call)
        self.subManualGoal = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.Callback_ManualGoal)
        
        self.obs_poses = PoseArray()
        self.obs_pose_lists = []
        self.obs_idx_lists = []
        self.obs_info = Float32MultiArray()
        self.obs_info = Float32MultiArray()

        self.steer = 0
        self.acc = 0
        self.target_velocity = 0
        self.dir_mode = 1

        self.init_flag = True
        self.x0 = 0# Be a ref. frame
        self.y0 = 0
        self.th0 = 0
        
        # based on the ref. frame
        self.local_x = 0
        self.local_y = 0
        self.local_th = 0
        self.vel = 0

        # PARKING
        self.parking_flag0 = False
        self.parking_flag1 = False
        self.restart_flag = False
        self.parking_goal = Pose()

        # image for screenShot
        self.rgbImage = None
        self.segImage = None

        InitializeScenario(self.scenario_count, self.case_count)# #MSK FIXME        
        global M_Initial_X, M_Initial_Y, M_Initial_Angle, M_Goal_X, M_Goal_Y, M_Goal_Angle
        if len(self.args.init_pose) >= 3: # add for spawning the vehicle in custom
            M_Initial_X = float(self.args.init_pose[0])
            M_Initial_Y = float(self.args.init_pose[1])
            M_Initial_Angle = float(self.args.init_pose[2])
        
        if len(self.args.goal_pose) >= 3:
            M_Goal_X = float(self.args.goal_pose[0])
            M_Goal_Y = float(self.args.goal_pose[1])
            M_Goal_Angle = float(self.args.goal_pose[2])
        
        self.obs_actors = list()
        self.obs_actor_ids = list()

        self.saver_obs = []
        self.saver_ego_pose = []
        self.saver_goal_pose = []

        self.init_world_flag = True
        self.restart_random()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def agent_blueprint(self):
        blueprint = random.choice(get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)) # Get a random blueprint.
        # blueprint = self.world.get_blueprint_library().filter('vehicle.*')[11] #0: small vehicle (audi), 14: small-medium,  8: medium vehicle (sedane), 20: large vehicle 
        blueprint = self.world.get_blueprint_library().find('vehicle.mercedes.coupe_2020')
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            # color = random.choice(blueprint.get_attribute('color').recommended_values)
            color = '150, 150, 150'
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        return blueprint


    def perpen_spawn_player(self, blueprint):
        global M_Goal_X, M_Goal_Y, M_Goal_Angle
        # Spawn the player.
        random_pose_x = 0
        random_pose_y = 0
        random_pose_th = 0
        while self.player is None:
            random_x = 7
            random_y = 5
            # random_x = 5
            # random_y = 5
            angle_dist = 30
            # angle_dist = 0
            move_x = np.random.uniform(low=5, high=random_x)
            move_y = np.random.uniform(low=-random_y, high=random_y)
            # move_x = random_x
            # move_y = random_y
            random_pose_x = M_Goal_X + move_x * math.cos(M_Goal_Angle * DEG2RAD) + move_y * math.cos((M_Goal_Angle+90.0) * DEG2RAD)
            random_pose_y = M_Goal_Y + move_x * math.sin(M_Goal_Angle * DEG2RAD) + move_y * math.sin((M_Goal_Angle+90.0) * DEG2RAD)
            random_pose_th = M_Goal_Angle + random.choice([-90, 90]) + angle_dist * np.random.uniform(low=-1, high=1)
            # random_pose_x = M_Goal_X + move_x * math.cos(M_Goal_Angle * DEG2RAD) + move_y * math.cos((M_Goal_Angle+90.0) * DEG2RAD)
            # random_pose_y = M_Goal_Y + move_x * math.sin(M_Goal_Angle * DEG2RAD) + move_y * math.sin((M_Goal_Angle+90.0) * DEG2RAD)
            # random_pose_th = M_Goal_Angle + random.choice([90]) + angle_dist * np.random.uniform(low=-1, high=1)

            # spawn_point = SpawnPoseAt(self.world, test_mcts_driving_pose_set[0][0],
            #                           test_mcts_driving_pose_set[0][1],
            #                           test_mcts_driving_pose_set[0][2])
            spawn_point = SpawnPoseAt(self.world, random_pose_x, random_pose_y, random_pose_th)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            
            self.modify_vehicle_physics(self.player)
            if self.player is None:
                continue
            if abs(self.player.get_transform().rotation.pitch) >= 0.001 or abs(self.player.get_transform().rotation.roll) >= 0.001:
                self.player = None

            # M_Goal_X = random_pose_x
            # M_Goal_Y = random_pose_y + 10.0
            # M_Goal_Angle = random_pose_th

        return random_pose_x, random_pose_y, random_pose_th

 
    def perpendicular_parking_scenario(self):
        global m_candidate_pose_set, M_Goal_X, M_Goal_Y, M_Goal_Angle
        # obs_poses = PoseArray()
        self.obs_poses = PoseArray()
        self.obs_pose_lists = []
        self.obs_idx_lists = []
        self.obs_poses.header.frame_id = "map"
        self.obs_poses.header.stamp = rospy.Time.now()
        ##### -------------------- Goal Setting
        which_goal = random.sample(goal_candidate_idx, 1)[0]
        M_Goal_X = m_candidate_pose_set[which_goal][0]
        M_Goal_Y = m_candidate_pose_set[which_goal][1]
        M_Goal_Angle = m_candidate_pose_set[which_goal][2]
        # M_Goal_X = test_mcts_driving_pose_set[0][0] + 10.0
        # M_Goal_Y = test_mcts_driving_pose_set[0][1] + 10.0
        # M_Goal_Angle = test_mcts_driving_pose_set[0][2]

        self.saver_goal_pose = [M_Goal_X, M_Goal_Y, M_Goal_Angle]
        self.saver_obs.clear()
        
        ################################################################################
        ##### -------------------- Obstacle Setting for each parking spot
        temp = obs_candidate_idx.copy()
        if which_goal in temp:
            temp.remove(which_goal)
        selected_obs_idx = random.sample(temp, 20)
        parked_car_angle_dev = 5 #[deg]
        for idx in selected_obs_idx:
            pose = m_candidate_pose_set[idx]
            vehicle_idx = random.choice([0, 2, 3, 11, 14, 15, 24, 28, 29, 32])
            obs_blueprint = self.world.get_blueprint_library().filter('vehicle.*')[vehicle_idx] #ms: get Tesla, pick a vehicle[19]
            color = '150, 150, 150'
            if obs_blueprint.has_attribute('color'):
                obs_blueprint.set_attribute('color', color)
            if obs_blueprint.has_attribute('is_invincible'):
                obs_blueprint.set_attribute('is_invincible', 'true')
                
            obs_x = pose[0]
            obs_y = pose[1]
            obs_angle = pose[2] + parked_car_angle_dev * np.random.uniform()
            spawn_point = SpawnPoseAt(self.world, obs_x, obs_y, obs_angle)
            self.obs_actors.append(carla.command.SpawnActor(obs_blueprint, spawn_point))

            self.saver_obs.append([obs_x, obs_y, obs_angle, vehicle_idx])

            # obs_x_, obs_y_, obs_angle_ = global2local(obs_x, obs_y, obs_angle * DEG2RAD, M_Goal_X, M_Goal_Y, M_Goal_Angle * DEG2RAD)
            # self.obs_pose_lists.append([obs_x_, obs_y_, obs_angle_])
            self.obs_pose_lists.append([obs_x, obs_y, obs_angle])
            self.obs_idx_lists.append(vehicle_idx)
            # self.obs_poses.poses.append(poseArray(obs_x_, obs_y_, obs_angle_))
            self.obs_poses.poses.append(poseArray(obs_x, obs_y, obs_angle))
        
        self.get_obstacle_info()
        # print(len(self.obs_info.data))

            # self.obs_actors.append(self.world.try_spawn_actor(obs_blueprint, spawn_point))

        # self.pub_obs_poses.publish(obs_poses)

        # angle_dev = 10 #[deg]
        # ################################################################################
        # ##### -------------------- Obstacle Setting for irregularly-placed obstacle
        # irr_obs_threshold = 0.5 #0.73
        # if np.random.uniform() < irr_obs_threshold:
        #     vehicle_idx = random.choice([0, 11])
        #     obs_blueprint = self.world.get_blueprint_library().filter('vehicle.*')[vehicle_idx] #ms: get Tesla, pick a vehicle[19]
        #     color = '150, 150, 150'
        #     if obs_blueprint.has_attribute('color'):
        #         obs_blueprint.set_attribute('color', color)
        #     if obs_blueprint.has_attribute('is_invincible'):
        #         obs_blueprint.set_attribute('is_invincible', 'true')
        #     random_pts = [(3.5, np.random.uniform(low=4, high=6)), (3.5, -np.random.uniform(low=4, high=6)), (7.25, -np.random.uniform(low=4, high=6)), (7.25, np.random.uniform(low=4, high=6))]
        #     indd = random.randint(0, len(random_pts)-1)
        #     obs_x = M_Goal_X + random_pts[indd][0] * math.cos(M_Goal_Angle * DEG2RAD) + random_pts[indd][1] * math.cos((M_Goal_Angle+90.0) * DEG2RAD)
        #     obs_y = M_Goal_Y + random_pts[indd][0] * math.sin(M_Goal_Angle * DEG2RAD) + random_pts[indd][1] * math.sin((M_Goal_Angle+90.0) * DEG2RAD)
        #     obs_angle = M_Goal_Angle + random.choice([-90 + angle_dev*np.random.uniform(), 90 + angle_dev*np.random.uniform()])
        #     spawn_point = SpawnPoseAt(self.world, obs_x, obs_y, obs_angle)
        #     self.obs_actors.append(carla.command.SpawnActor(obs_blueprint, spawn_point))
        #     # self.obs_actors.append(self.world.try_spawn_actor(obs_blueprint, spawn_point))
        #     self.saver_obs.append([obs_x, obs_y, obs_angle, vehicle_idx])

        # ################################################################################
        # ##### -------------------- Obstacle Setting for irregularly-placed obstacle2
        # irr_obs_threshold2 = 0.5
        # if np.random.uniform() < irr_obs_threshold2:
        #     vehicle_idx = random.choice([0, 11])
        #     obs_blueprint = self.world.get_blueprint_library().filter('vehicle.*')[vehicle_idx] #ms: get Tesla, pick a vehicle[19]
        #     color = '150, 150, 150'
        #     if obs_blueprint.has_attribute('color'):
        #         obs_blueprint.set_attribute('color', color)
        #     if obs_blueprint.has_attribute('is_invincible'):
        #         obs_blueprint.set_attribute('is_invincible', 'true')

        #     # x-axis is relatec to the orientation of the vehicle's parking goal 
        #     random_pts = [(3.5, np.random.uniform(low=7, high=9)), (3.5, -np.random.uniform(low=7, high=9)), (7.25, -np.random.uniform(low=7, high=9)), (7.25, np.random.uniform(low=7, high=9))]
        #     indd = random.randint(0, len(random_pts)-1)
        #     obs_x = M_Goal_X + random_pts[indd][0] * math.cos(M_Goal_Angle * DEG2RAD) + random_pts[indd][1] * math.cos((M_Goal_Angle+90.0) * DEG2RAD)
        #     obs_y = M_Goal_Y + random_pts[indd][0] * math.sin(M_Goal_Angle * DEG2RAD) + random_pts[indd][1] * math.sin((M_Goal_Angle+90.0) * DEG2RAD)
        #     obs_angle = M_Goal_Angle + random.choice([-90 + angle_dev*np.random.uniform(), 90 + angle_dev*np.random.uniform()])
        #     spawn_point = SpawnPoseAt(self.world, obs_x, obs_y, obs_angle)
        #     self.obs_actors.append(carla.command.SpawnActor(obs_blueprint, spawn_point))
        #     self.saver_obs.append([obs_x, obs_y, obs_angle, vehicle_idx])

        for response in self.client.apply_batch_sync(self.obs_actors, self.sync):
            if response.error:
                logging.error(response.error)
            else:
                self.obs_actor_ids.append(response.actor_id)

    def get_obstacle_info(self):
        # vehicles = self.world.get_actors().filter('vehicle.*')
        # obs_info = Float32MultiArray()
        # print(self.obs_idx_lists)
        # print(self.obs_pose_lists)
        self.obs_info.data = []
        for idx, pose in zip(self.obs_idx_lists, self.obs_pose_lists):
            obs_x = pose[0]
            obs_y = pose[1]
            obs_th = pose[2]
            obs_w = bbox_extent[idx][1]
            obs_h = bbox_extent[idx][0]
            self.obs_info.data.append(obs_x)
            self.obs_info.data.append(obs_y)
            self.obs_info.data.append(obs_th * DEG2RAD)
            self.obs_info.data.append((obs_h - 1.852685) / (2.678740 - 1.852685))
            self.obs_info.data.append((obs_w - 0.894339) / (1.120857 - 0.894339))
        self.pub_obs_info.publish(self.obs_info)
        

    def parallel_spawn_player(self, blueprint, near_building):
        global M_Goal_X, M_Goal_Y, M_Goal_Angle
        # Spawn the player.
        random_pose_x = 0
        random_pose_y = 0
        random_pose_th = 0

        to_right = True
        if ((near_building and M_Goal_Angle>0.0) or ((not near_building) and M_Goal_Angle < 0.0)): # near building and 90.0 degree
            to_right = False        

        while self.player is None:
            random_x = 4
            random_y = 7
            angle_dist = 20
            move_x = np.random.uniform(low=2, high=random_x)
            move_y = np.random.uniform(low=-random_y, high=random_y)
            if (to_right):
                random_pose_x = M_Goal_X + move_x * math.cos((M_Goal_Angle - 90.0) * DEG2RAD) + move_y * math.cos((M_Goal_Angle) * DEG2RAD)
                random_pose_y = M_Goal_Y + move_x * math.sin((M_Goal_Angle - 90.0) * DEG2RAD) + move_y * math.sin((M_Goal_Angle) * DEG2RAD)
                random_pose_th = M_Goal_Angle + angle_dist * np.random.uniform(low=-1, high=1)
            else: # to_left
                random_pose_x = M_Goal_X + move_x * math.cos((M_Goal_Angle + 90.0) * DEG2RAD) + move_y * math.cos((M_Goal_Angle) * DEG2RAD)
                random_pose_y = M_Goal_Y + move_x * math.sin((M_Goal_Angle + 90.0) * DEG2RAD) + move_y * math.sin((M_Goal_Angle) * DEG2RAD)
                random_pose_th = M_Goal_Angle + angle_dist * np.random.uniform(low=-1, high=1)

            spawn_point = SpawnPoseAt(self.world, random_pose_x, random_pose_y, random_pose_th)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            
            self.modify_vehicle_physics(self.player)
            if self.player is None:
                continue
            if abs(self.player.get_transform().rotation.pitch) >= 0.001 or abs(self.player.get_transform().rotation.roll) >= 0.001:
                self.player = None
        return random_pose_x, random_pose_y, random_pose_th

    def parallel_parking_scenario(self):
        global m_candidate_pose_set2, m_candidate_pose_set3, M_Goal_X, M_Goal_Y, M_Goal_Angle
        near_building = True
        if np.random.uniform() > 0.5:
            pose_set = m_candidate_pose_set2
            opp_pose_set = m_candidate_pose_set3
        else:
            pose_set = m_candidate_pose_set3
            opp_pose_set = m_candidate_pose_set2
            near_building = False

        which_goal = random.randint(0, len(pose_set)-1)
        M_Goal_X = pose_set[which_goal][0]
        M_Goal_Y = pose_set[which_goal][1]
        dir = -1 if np.random.uniform() > 0.5 else +1
        M_Goal_Angle = dir*pose_set[which_goal][2]
        self.saver_goal_pose = [M_Goal_X, M_Goal_Y, M_Goal_Angle]
        self.saver_obs.clear()

        obs_threshold = 0.7
        parked_car_angle_dev = 5
        for i, pose in enumerate(pose_set):
            if i == which_goal:
                continue
            
            if np.random.uniform() < obs_threshold:
                vehicle_idx = 8
                obs_blueprint = self.world.get_blueprint_library().filter('vehicle.*')[vehicle_idx] #ms: get Tesla, pick a vehicle[19]
                color = '150, 150, 150'
                if obs_blueprint.has_attribute('color'):
                    obs_blueprint.set_attribute('color', color)
                if obs_blueprint.has_attribute('is_invincible'):
                    obs_blueprint.set_attribute('is_invincible', 'true')
                    
                obs_x = pose[0]
                obs_y = pose[1]
                dir = -1 if np.random.uniform() > 0.5 else +1
                obs_angle = dir*pose[2] + parked_car_angle_dev * np.random.uniform()
                spawn_point = SpawnPoseAt(self.world, obs_x, obs_y, obs_angle)
                self.obs_actors.append(carla.command.SpawnActor(obs_blueprint, spawn_point))
                self.saver_obs.append([obs_x, obs_y, obs_angle, vehicle_idx])
        
        for i, pose in enumerate(opp_pose_set):
            if np.random.uniform() < obs_threshold:
                vehicle_idx = 8
                obs_blueprint = self.world.get_blueprint_library().filter('vehicle.*')[vehicle_idx] #ms: get Tesla, pick a vehicle[19]
                color = '150, 150, 150'
                if obs_blueprint.has_attribute('color'):
                    obs_blueprint.set_attribute('color', color)
                if obs_blueprint.has_attribute('is_invincible'):
                    obs_blueprint.set_attribute('is_invincible', 'true')
                    
                obs_x = pose[0]
                obs_y = pose[1]
                dir = -1 if np.random.uniform() > 0.5 else +1
                obs_angle = dir*pose[2] + parked_car_angle_dev * np.random.uniform()
                spawn_point = SpawnPoseAt(self.world, obs_x, obs_y, obs_angle)
                self.obs_actors.append(carla.command.SpawnActor(obs_blueprint, spawn_point))
                self.saver_obs.append([obs_x, obs_y, obs_angle, vehicle_idx])

        for response in self.client.apply_batch_sync(self.obs_actors, self.sync):
            if response.error:
                logging.error(response.error)
            else:
                self.obs_actor_ids.append(response.actor_id)

        return near_building

    def circular_spawn_player(self, blueprint):
        global m_candidate_start_pose4
        # Spawn the player.
        random_pose_x = 0
        random_pose_y = 0
        random_pose_th = 0

        ind = 0 if np.random.uniform() > 0.5 else 1
        random_pose_x = m_candidate_start_pose4[ind][0]
        random_pose_y = m_candidate_start_pose4[ind][1]
        random_pose_th = 0.0 if np.random.uniform() > 0.5 else 180.0
        
        if ind == 0:
            random_pose_th += 15 * np.random.uniform(low=-1, high=1)
            spawn_point = SpawnPoseAt(self.world, random_pose_x, random_pose_y, random_pose_th)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

            return random_pose_x, random_pose_y, random_pose_th

        else: # the vehicle is located on the center of the parking spot
            any_angle = 180 * np.random.uniform(low=-1, high=1)
            any_distance = 1.0 * np.random.uniform(low=-1, high=1)
            random_pose_x += any_distance*math.cos(any_angle * DEG2RAD)
            random_pose_y += any_distance*math.sin(any_angle * DEG2RAD)
            random_pose_th = any_angle
            spawn_point = SpawnPoseAt(self.world, random_pose_x, random_pose_y, random_pose_th)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
            return random_pose_x, random_pose_y, random_pose_th


    def circular_parking_scenario(self): 
        global m_candidate_pose_set4, M_Goal_X, M_Goal_Y, M_Goal_Angle

        which_goal = random.randint(0, len(m_candidate_pose_set4)-1)
        M_Goal_X = m_candidate_pose_set4[which_goal][0]
        M_Goal_Y = m_candidate_pose_set4[which_goal][1]
        dir = 180 if np.random.uniform() > 0.5 else 0
        M_Goal_Angle = m_candidate_pose_set4[which_goal][2] + dir
        self.saver_goal_pose = [M_Goal_X, M_Goal_Y, M_Goal_Angle]
        self.saver_obs.clear()
        
        obs_threshold = 0.8
        parked_car_angle_dev = 5
        for i, pose in enumerate(m_candidate_pose_set4):
            if i == which_goal:
                continue
            
            if np.random.uniform() < obs_threshold:
                vehicle_idx = 8
                obs_blueprint = self.world.get_blueprint_library().filter('vehicle.*')[vehicle_idx] #ms: get Tesla, pick a vehicle[19]
                color = '150, 150, 150'
                if obs_blueprint.has_attribute('color'):
                    obs_blueprint.set_attribute('color', color)
                if obs_blueprint.has_attribute('is_invincible'):
                    obs_blueprint.set_attribute('is_invincible', 'true')
                    
                obs_x = pose[0]
                obs_y = pose[1]
                dir = 180 if np.random.uniform() > 0.5 else 0.0
                obs_angle = dir + pose[2] + parked_car_angle_dev * np.random.uniform()
                spawn_point = SpawnPoseAt(self.world, obs_x, obs_y, obs_angle)
                self.obs_actors.append(carla.command.SpawnActor(obs_blueprint, spawn_point))
                self.saver_obs.append([obs_x, obs_y, obs_angle, vehicle_idx])
        
        for response in self.client.apply_batch_sync(self.obs_actors, self.sync):
            if response.error:
                logging.error(response.error)
            else:
                self.obs_actor_ids.append(response.actor_id)
        
        if np.random.uniform() < 0.05:
            ind = 0 if np.random.uniform() > 0.5 else 1
            M_Goal_X = m_candidate_start_pose4[ind][0]
            M_Goal_Y = m_candidate_start_pose4[ind][1]
            dir = 180 if np.random.uniform() > 0.5 else 0
            M_Goal_Angle = m_candidate_pose_set4[which_goal][2] + dir
            self.saver_goal_pose = [M_Goal_X, M_Goal_Y, M_Goal_Angle]
            self.saver_obs.clear()
            
    def arena_spawn(self, blueprint):
        global M_Goal_X, M_Goal_Y, M_Goal_Angle
        while self.player is None: # Generate a start pose
            random_pose_x = random.uniform(m_arena_min_x, m_arena_max_x)
            random_pose_y = random.uniform(m_arena_min_y, m_arena_max_y)
            random_pose_th = 180 * np.random.uniform(low=-1, high=1)
            spawn_point = SpawnPoseAt(self.world, random_pose_x, random_pose_y, random_pose_th)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            rospy.sleep(0.1)

            self.modify_vehicle_physics(self.player)
            if self.player is None:
                continue
            # print("ddd ", self.player.get_transform().rotation)
            if abs(self.player.get_transform().rotation.pitch) >= 0.001 or abs(self.player.get_transform().rotation.roll) >= 0.001:
                self.player = None
                continue
        
        temp_player = None
        while temp_player is None:
            M_Goal_X = random_pose_x + random.uniform(-14, 14)
            M_Goal_Y = random_pose_y + random.uniform(-14, 14)

            if ((m_arena_min_x > M_Goal_X) or (m_arena_max_x < M_Goal_X) or (m_arena_max_y < M_Goal_Y) or (m_arena_min_y > M_Goal_Y)):
                continue
            M_Goal_Angle = 180 * np.random.uniform(low=-1, high=1)
            spawn_point = SpawnPoseAt(self.world, M_Goal_X, M_Goal_Y, M_Goal_Angle)
            temp_player = self.world.try_spawn_actor(blueprint, spawn_point)
            rospy.sleep(0.1)

            self.modify_vehicle_physics(temp_player)
            if temp_player is None:
                continue
            # print("ddd ", temp_player.get_transform().rotation)
            if abs(temp_player.get_transform().rotation.pitch) >= 0.01 or abs(temp_player.get_transform().rotation.roll) >= 0.01:
                temp_player = None

        temp_player.destroy()
        return random_pose_x, random_pose_y, random_pose_th

    def arena_scenario(self): 
        num_obs = random.choice([10, 15, 20, 25, 30, 35, 40])
        obs_count = 0
        global m_arena_min_x, m_arena_max_x, m_arena_min_y, m_arena_max_y

        while obs_count < num_obs:
            vehicle_idx = random.choice([0, 8, 19])
            obs_blueprint = self.world.get_blueprint_library().filter('vehicle.*')[vehicle_idx] #ms: get Tesla, pick a vehicle[19]
            color = '150, 150, 150'
            if obs_blueprint.has_attribute('color'):
                obs_blueprint.set_attribute('color', color)
            if obs_blueprint.has_attribute('is_invincible'):
                obs_blueprint.set_attribute('is_invincible', 'true')
                
            obs_x = random.uniform(m_arena_min_x, m_arena_max_x)
            obs_y = random.uniform(m_arena_min_y, m_arena_max_y)
            obs_angle = 180 * np.random.uniform(low=-1, high=1)
            spawn_point = SpawnPoseAt(self.world, obs_x, obs_y, obs_angle)
            self.obs_actors.append(carla.command.SpawnActor(obs_blueprint, spawn_point))
            self.saver_obs.append([obs_x, obs_y, obs_angle, vehicle_idx])
            obs_count += 1

        for response in self.client.apply_batch_sync(self.obs_actors, self.sync):
            if response.error:
                logging.error(response.error)
            else:
                self.obs_actor_ids.append(response.actor_id)

    def narrow_spawn(self, blueprint):
        global M_Goal_X, M_Goal_Y, M_Goal_Angle, m_candidate_pose_set5, m_candidate_pose_set6
        pose_set = m_candidate_pose_set5 if random.random() > 0.5 else m_candidate_pose_set6

        random_pose_x, random_pose_y, random_pose_th = 0, 0, 0
        if random.random() > 0.5: 
            M_Goal_X, M_Goal_Y, M_Goal_Angle = pose_set[0]
            random_pose_x, random_pose_y, random_pose_th = pose_set[1]
            if random.random() > 0.5:
                M_Goal_Angle += 0.0
                random_pose_th += 0.0
            else: 
                M_Goal_Angle += 180.0
                random_pose_th += 180.0
        else: 
            M_Goal_X, M_Goal_Y, M_Goal_Angle = pose_set[1]
            random_pose_x, random_pose_y, random_pose_th = pose_set[0]
            if random.random() > 0.5:
                M_Goal_Angle += 0.0
                random_pose_th += 0.0
            else: 
                M_Goal_Angle += 180.0
                random_pose_th += 180.0

        while self.player is None: # Generate a start pose
            random_pose_x += np.random.uniform(low=-0.25, high=0.25)
            random_pose_y += np.random.uniform(low=-0.25, high=0.25)
            random_pose_th += 2.5 * np.random.uniform(low=-1, high=1)
            spawn_point = SpawnPoseAt(self.world, random_pose_x, random_pose_y, random_pose_th)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            rospy.sleep(0.1)

            self.modify_vehicle_physics(self.player)
            if self.player is None:
                continue
            # print("ddd ", self.player.get_transform().rotation)
            if abs(self.player.get_transform().rotation.pitch) >= 0.001 or abs(self.player.get_transform().rotation.roll) >= 0.001:
                self.player = None
                continue
        return random_pose_x, random_pose_y, random_pose_th


    def narrow_scenario(self):
        # The obstacles are set already in the environments
        return


    def restart_random(self): ################################################################ for logging
        self.finish_call = False
        global M_Goal_X, M_Goal_Y, M_Goal_Angle
        if self.load_scenario_file != '': # if a specific scenario is going to run!
            print("Load... ", self.load_scenario_file)
            if not self.init_world_flag:
                self.destroy()
            
            self.player = None

            # Use regular expression to extract the number
            match = re.search(r'\d+', self.load_scenario_file)

            if match:
                scenario_number = str(match.group())
            else:
                print("No number found in the input string.")
                raise ValueError("Invalid scenario name")

            file_path = "/home/dyros-mk/catkin_ws/src/data_logging/data/" + scenario_number + "/" + self.load_scenario_file
            print(file_path)
            scenario_file = open(file_path, 'rb')
            loaded_scenario = pickle.load(scenario_file)

            ego_pose = loaded_scenario['ego_pose']
            goal_pose = loaded_scenario['goal_pose']
            obs_poses = loaded_scenario['obs_poses']

            ##### -------------------- Goal Setting
            M_Goal_X = goal_pose[0]
            M_Goal_Y = goal_pose[1]
            M_Goal_Angle = goal_pose[2]

            ##### -------------------- Obstacle Setting
            for obs_pose in obs_poses:
                vehicle_idx = obs_pose[3]
                obs_blueprint = self.world.get_blueprint_library().filter('vehicle.*')[vehicle_idx] #ms: get Tesla, pick a vehicle[19]
                color = '150, 150, 150'
                if obs_blueprint.has_attribute('color'):
                    obs_blueprint.set_attribute('color', color)
                if obs_blueprint.has_attribute('is_invincible'):
                    obs_blueprint.set_attribute('is_invincible', 'true')
                    
                obs_x = obs_pose[0]
                obs_y = obs_pose[1]
                obs_angle = obs_pose[2]
                spawn_point = SpawnPoseAt(self.world, obs_x, obs_y, obs_angle)
                self.obs_actors.append(carla.command.SpawnActor(obs_blueprint, spawn_point))

            for response in self.client.apply_batch_sync(self.obs_actors, self.sync):
                if response.error:
                    logging.error(response.error)
                else:
                    self.obs_actor_ids.append(response.actor_id)

            self.player_max_speed = 1.589
            self.player_max_speed_fast = 3.713
            

            # Get a random blueprint.
            blueprint = self.agent_blueprint()

            # Spawn the player.
            spawn_point = SpawnPoseAt(self.world, ego_pose[0], ego_pose[1], ego_pose[2])
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
            
            self.current_pose_x = ego_pose[0]
            self.current_pose_y = ego_pose[1]
            self.current_pose_th = ego_pose[2]
        
            # Keep same camera config if the camera manager exists.
            if self.cam_on:
                cam_index = self.camera_manager.index if self.camera_manager is not None else 0
                cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
                self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
                self.camera_manager.transform_index = cam_pos_index
                self.camera_manager.set_sensor(cam_index, notify=False)
            
            # Set up the sensors.
            self.lidar_sensor = LidarSensor(self.player, self, semantic=True)
            self.collision_sensor = CollisionSensor(self.player, self.hud)
            # self.imu_sensor = IMUSensor(self.player)
            self.seg_camera = SegmentationTopViewCamera(self.player, self.hud, self._gamma, xx=0, yy=0, zz = M_DIMENSION / 2)
            self.seg_camera.set_sensor()

            if self.avm_on:
                self.seg_avm_front_camera = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'rgb', (2.4, 0.0, 1, 0.0, -10.0, 0.0))
                self.seg_avm_front_camera.set_sensor()
                self.seg_avm_back_camera = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'rgb', (-2.4, 0.0, 1, 0.0, -10.0, 180.0))
                self.seg_avm_back_camera.set_sensor()
                self.seg_avm_right_camera = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'rgb', (0.0, 0.9, 1, 0.0, -10.0, 90.0))
                self.seg_avm_right_camera.set_sensor()
                self.seg_avm_left_camera = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'rgb', (0.0, -0.9, 1, 0.0, -10.0, -90.0))
                self.seg_avm_left_camera.set_sensor()

                self.seg_avm_front_camera2 = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'seg', (2.4, 0.0, 1, 0.0, -10.0, 0.0))
                self.seg_avm_front_camera2.set_sensor()
                self.seg_avm_back_camera2 = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'seg', (-2.4, 0.0, 1, 0.0, -10.0, 180.0))
                self.seg_avm_back_camera2.set_sensor()
                self.seg_avm_right_camera2 = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'seg', (0.0, 0.9, 1, 0.0, -10.0, 90.0))
                self.seg_avm_right_camera2.set_sensor()
                self.seg_avm_left_camera2 = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'seg', (0.0, -0.9, 1, 0.0, -10.0, -90.0))
                self.seg_avm_left_camera2.set_sensor()
            
            # self.gnss_sensor = GnssSensor(self.player)

            self.target_velocity = 0
            self.parking_flag0 = False
            self.parking_flag1 = False
            self.x0, self.y0, self.th0 = 0, 0, 0
            self.restart_flag = False       

            if self.sync:
                self.world.tick()
            else:
                self.world.wait_for_tick()
            
            msg_ = Float32MultiArray()
            msg_.data = [ego_pose[0], ego_pose[1], ego_pose[2], M_Goal_X, M_Goal_Y, M_Goal_Angle]
            self.pub_situation_info.publish(msg_)
            self.init_world_flag = False

            return

        if self.change_map: # Every
            if not self.init_world_flag:
                self.destroy()
            
            self.player = None

            self.player_max_speed = 1.589
            self.player_max_speed_fast = 3.713
            
            randomElement = np.random.choice(range(len(self.scenario_ratio)), p=self.scenario_ratio)
            blueprint = self.agent_blueprint()
            if randomElement == 0:
                self.perpendicular_parking_scenario()
                self.current_pose_x, self.current_pose_y, self.current_pose_th = self.perpen_spawn_player(blueprint)

            elif randomElement == 1: 
                near_building = self.parallel_parking_scenario()
                self.current_pose_x, self.current_pose_y, self.current_pose_th = self.parallel_spawn_player(blueprint, near_building)
            elif randomElement == 2:
                self.circular_parking_scenario()
                self.current_pose_x, self.current_pose_y, self.current_pose_th = self.circular_spawn_player(blueprint)
            elif randomElement == 3: 
                self.arena_scenario()
                self.current_pose_x, self.current_pose_y, self.current_pose_th = self.arena_spawn(blueprint)
            elif randomElement == 4: 
                self.narrow_scenario()
                self.current_pose_x, self.current_pose_y, self.current_pose_th = self.narrow_spawn(blueprint)

            self.saver_ego_pose = [self.current_pose_x, self.current_pose_y, self.current_pose_th]          
            # Set up the sensors.
            self.lidar_sensor = LidarSensor(self.player, self, semantic=True)
            self.collision_sensor = CollisionSensor(self.player, self.hud)
            # self.imu_sensor = IMUSensor(self.player)
            if self.cam_on:
                # Keep same camera config if the camera manager exists.
                cam_index = self.camera_manager.index if self.camera_manager is not None else 0
                cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
                self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
                self.camera_manager.transform_index = cam_pos_index
                self.camera_manager.set_sensor(cam_index, notify=False)
            self.seg_camera = SegmentationTopViewCamera(self.player, self.hud, self._gamma, xx=0, yy=0, zz = M_DIMENSION / 2)
            self.seg_camera.set_sensor()

            if self.avm_on:
                self.seg_avm_front_camera = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'rgb', (2.4, 0.0, 1, 0.0, -10.0, 0.0))
                self.seg_avm_front_camera.set_sensor()
                self.seg_avm_back_camera = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'rgb', (-2.4, 0.0, 1, 0.0, -10.0, 180.0))
                self.seg_avm_back_camera.set_sensor()
                self.seg_avm_right_camera = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'rgb', (0.0, 0.9, 1, 0.0, -10.0, 90.0))
                self.seg_avm_right_camera.set_sensor()
                self.seg_avm_left_camera = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'rgb', (0.0, -0.9, 1, 0.0, -10.0, -90.0))
                self.seg_avm_left_camera.set_sensor()

                self.seg_avm_front_camera2 = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'seg', (2.4, 0.0, 1, 0.0, -10.0, 0.0))
                self.seg_avm_front_camera2.set_sensor()
                self.seg_avm_back_camera2 = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'seg', (-2.4, 0.0, 1, 0.0, -10.0, 180.0))
                self.seg_avm_back_camera2.set_sensor()
                self.seg_avm_right_camera2 = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'seg', (0.0, 0.9, 1, 0.0, -10.0, 90.0))
                self.seg_avm_right_camera2.set_sensor()
                self.seg_avm_left_camera2 = SegmentationAVMCamera(self.player, self.hud, self._gamma, 'seg', (0.0, -0.9, 1, 0.0, -10.0, -90.0))
                self.seg_avm_left_camera2.set_sensor()
            
            # self.gnss_sensor = GnssSensor(self.player)

            self.target_velocity = 0
            self.parking_flag0 = False
            self.parking_flag1 = False
            self.x0, self.y0, self.th0 = 0, 0, 0
            self.restart_flag = False       

            if self.sync:
                self.world.tick()
            else:
                self.world.wait_for_tick()
            
            msg_ = Float32MultiArray()
            msg_.data = [self.current_pose_x, self.current_pose_y, self.current_pose_th, M_Goal_X, M_Goal_Y, M_Goal_Angle]
            self.pub_situation_info.publish(msg_)

        else:            
            ego_pose = self.player.get_transform()
            ego_pose.location.x = self.current_pose_x
            ego_pose.location.y = self.current_pose_y
            ego_pose.rotation.yaw = self.current_pose_th
            ego_pose.location.z += 2.0
            self.player.set_transform(ego_pose)

            self.target_velocity = 0
            self.parking_flag0 = False
            self.parking_flag1 = False
            self.x0, self.y0, self.th0 = 0, 0, 0
            self.restart_flag = False       

            if self.sync:
                self.world.tick()
            else:
                self.world.wait_for_tick()
        self.init_world_flag = False



    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # # Keep same camera config if the camera manager exists.
        # cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        # cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        # blueprint = random.choice(get_actor_blueprints(self.world, self._actor_filter, self._actor_generation))
        blueprint = self.world.get_blueprint_library().filter('vehicle.*')[8] #0: small vehicle (audi), 14: small-medium,  8: medium vehicle (sedane), 20: large vehicle 
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            # color = random.choice(blueprint.get_attribute('color').recommended_values)
            color = '150, 150, 150'
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        global m_candidate_pose_set, M_Goal_X, M_Goal_Y, M_Goal_Angle
        ##### -------------------- Goal Setting
        which_goal = random.randint(0, len(m_candidate_pose_set)-1)
        M_Goal_X = m_candidate_pose_set[which_goal][0]
        M_Goal_Y = m_candidate_pose_set[which_goal][1]
        M_Goal_Angle = m_candidate_pose_set[which_goal][2]
        print("which goal: ", M_Goal_X, M_Goal_Y, M_Goal_Angle)

        ##### -------------------- Obstacle Setting
        obs_threshold = 0.7
        for i, pose in enumerate(m_candidate_pose_set):
            if i == which_goal:
                continue
            
            if np.random.uniform() < obs_threshold:
                vehicle_idx = random.choice([0, 8, 8])
                obs_blueprint = self.world.get_blueprint_library().filter('vehicle.*')[vehicle_idx] #ms: get Tesla, pick a vehicle[19]
                color = '150, 150, 150'
                if obs_blueprint.has_attribute('color'):
                    obs_blueprint.set_attribute('color', color)
                if obs_blueprint.has_attribute('is_invincible'):
                    obs_blueprint.set_attribute('is_invincible', 'true')
                    
                obs_x = pose[0]
                obs_y = pose[1]
                obs_angle = pose[2]
                spawn_point = SpawnPoseAt(self.world, obs_x, obs_y, obs_angle)
                self.obs_actors.append(carla.command.SpawnActor(obs_blueprint, spawn_point))
                # self.obs_actors.append(self.world.try_spawn_actor(obs_blueprint, spawn_point))
        
        irr_obs_threshold = 0.73
        if np.random.uniform() < irr_obs_threshold:
            vehicle_idx = random.choice([0, 8])
            obs_blueprint = self.world.get_blueprint_library().filter('vehicle.*')[vehicle_idx] #ms: get Tesla, pick a vehicle[19]
            color = '150, 150, 150'
            if obs_blueprint.has_attribute('color'):
                obs_blueprint.set_attribute('color', color)
            if obs_blueprint.has_attribute('is_invincible'):
                obs_blueprint.set_attribute('is_invincible', 'true')
            random_pts = [(3.5, np.random.uniform(low=4, high=6)), (3.5, -np.random.uniform(low=4, high=6)), (7.25, -np.random.uniform(low=4, high=6)), (7.25, np.random.uniform(low=4, high=6))]
            indd = random.randint(0, len(random_pts)-1)
            obs_x = M_Goal_X + random_pts[indd][0] * math.cos(M_Goal_Angle * DEG2RAD) + random_pts[indd][1] * math.cos((M_Goal_Angle+90.0) * DEG2RAD)
            obs_y = M_Goal_Y + random_pts[indd][0] * math.sin(M_Goal_Angle * DEG2RAD) + random_pts[indd][1] * math.sin((M_Goal_Angle+90.0) * DEG2RAD)
            obs_angle = M_Goal_Angle + random.choice([-90 + 2*np.random.uniform(), 90 + 2*np.random.uniform()])
            spawn_point = SpawnPoseAt(self.world, obs_x, obs_y, obs_angle)
            self.obs_actors.append(carla.command.SpawnActor(obs_blueprint, spawn_point))
            # self.obs_actors.append(self.world.try_spawn_actor(obs_blueprint, spawn_point))
        
        ##### -------------------- Obstacle Setting for irregularly-placed obstacle2
        irr_obs_threshold2 = 1.0
        if np.random.uniform() < irr_obs_threshold2:
            vehicle_idx = random.choice([0, 8])
            obs_blueprint = self.world.get_blueprint_library().filter('vehicle.*')[vehicle_idx] #ms: get Tesla, pick a vehicle[19]
            color = '150, 150, 150'
            if obs_blueprint.has_attribute('color'):
                obs_blueprint.set_attribute('color', color)
            if obs_blueprint.has_attribute('is_invincible'):
                obs_blueprint.set_attribute('is_invincible', 'true')
            random_pts = [(3.5, np.random.uniform(low=7, high=9)), (3.5, -np.random.uniform(low=7, high=9)), (7.25, -np.random.uniform(low=7, high=9)), (7.25, np.random.uniform(low=7, high=9))]
            indd = random.randint(0, len(random_pts)-1)
            obs_x = M_Goal_X + random_pts[indd][0] * math.cos(M_Goal_Angle * DEG2RAD) + random_pts[indd][1] * math.cos((M_Goal_Angle+90.0) * DEG2RAD)
            obs_y = M_Goal_Y + random_pts[indd][0] * math.sin(M_Goal_Angle * DEG2RAD) + random_pts[indd][1] * math.sin((M_Goal_Angle+90.0) * DEG2RAD)
            obs_angle = M_Goal_Angle + random.choice([-90 + 2*np.random.uniform(), 90 + 2*np.random.uniform()])
            spawn_point = SpawnPoseAt(self.world, obs_x, obs_y, obs_angle)
            self.obs_actors.append(carla.command.SpawnActor(obs_blueprint, spawn_point))

        for response in self.client.apply_batch_sync(self.obs_actors, self.sync):
            if response.error:
                logging.error(response.error)
            else:
                self.obs_actor_ids.append(response.actor_id)

        # Spawn the player.
        if self.player is not None:
            spawn_point = ScenarioSpawnPose(self.world)
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        # Spawn the player.
        while self.player is None:
            random_x = 7.5
            random_y = 8
            angle_dist = 40
            move_x = np.random.uniform(low=5, high=random_x)
            move_y = np.random.uniform(low=-random_y, high=random_y)
            random_pose_x = M_Goal_X + move_x * math.cos(M_Goal_Angle * DEG2RAD) + move_y * math.cos((M_Goal_Angle+90.0) * DEG2RAD)
            random_pose_y = M_Goal_Y + move_x * math.sin(M_Goal_Angle * DEG2RAD) + move_y * math.sin((M_Goal_Angle+90.0) * DEG2RAD)
            random_pose_th = M_Goal_Angle + random.choice([-90, 90]) + angle_dist * np.random.uniform(low=-1, high=1)

            spawn_point = SpawnPoseAt(self.world, random_pose_x, random_pose_y, random_pose_th)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            # print("self.player: ", self.player)
            self.modify_vehicle_physics(self.player)
            if self.player is None:
                continue
            if (abs(self.player.get_transform().rotation.pitch) >= 0.001 or abs(self.player.get_transform().rotation.roll) >= 0.001):
                self.player = None

        # while self.player is None:
        #     if not self.map.get_spawn_points():
        #         print('There are no spawn points available in your map/town.')
        #         print('Please add some Vehicle Spawn Point to your UE4 scene.')
        #         sys.exit(1)
        #     # spawn_points = self.map.get_spawn_points()
        #     # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        #     spawn_point = ScenarioSpawnPose(self.world)
        #     self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        #     self.modify_vehicle_physics(self.player)
        
        # Spawn obstacles
        # global M_Obstacle_Exist
        # if M_Obstacle_Exist:
            # if (self.obs1 is not None):
            #     self.obs1.destroy()
            # self.obs1 = SpawnObstacle(self.world)
        
        # Set up the sensors.
        self.lidar_sensor = LidarSensor(self.player, self, semantic=True)
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        # self.imu_sensor = IMUSensor(self.player)

        if self.cam_on:
            # Keep same camera config if the camera manager exists.
            cam_index = self.camera_manager.index if self.camera_manager is not None else 0
            cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
            self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
            self.camera_manager.transform_index = cam_pos_index
            self.camera_manager.set_sensor(cam_index, notify=False)

        self.seg_camera = SegmentationTopViewCamera(self.player, self.hud, self._gamma, xx=0, yy=0, zz = M_DIMENSION / 2)
        self.seg_camera.set_sensor()

        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        self.target_velocity = 0
        self.parking_flag0 = False
        self.parking_flag1 = False
        self.x0, self.y0, self.th0 = 0, 0, 0
        self.restart_flag = False

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
                
        msg_ = Float32MultiArray()
        msg_.data = [random_pose_x, random_pose_y, random_pose_th, M_Goal_X, M_Goal_Y, M_Goal_Angle]
        self.pub_situation_info.publish(msg_)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
            # # check max_steering in rad. in case of benz, 70 deg at front wheel
            # print(physics_control.wheels[0].max_steer_angle, physics_control.wheels[1].max_steer_angle, physics_control.wheels[2].max_steer_angle, physics_control.wheels[3].max_steer_angle)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        if self.cam_on:
            self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        if self.cam_on:
            self.camera_manager.sensor.destroy()
            self.camera_manager.sensor = None
            self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.lidar_sensor.sensor,
            self.collision_sensor.sensor,
            self.seg_camera.sensor,
            # self.camera_manager.sensor,
            # self.imu_sensor.sensor,
            ]
        if self.avm_on:
            sensors += [
                self.seg_avm_front_camera.sensor,
                self.seg_avm_back_camera.sensor,
                self.seg_avm_right_camera.sensor,
                self.seg_avm_left_camera.sensor,
                self.seg_avm_front_camera2.sensor,
                self.seg_avm_back_camera2.sensor,
                self.seg_avm_right_camera2.sensor,
                self.seg_avm_left_camera2.sensor,
            ]
        if self.cam_on and self.camera_manager.sensor is not None:
            self.camera_manager.sensor.stop()
            self.camera_manager.sensor.destroy()
            
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
        
        if self.obs1 is not None:
            self.obs1.destroy()
        
        print('\ndestroying %d vehicles' % len(self.obs_actor_ids))
        for i in range(len(self.obs_actors)):
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.obs_actor_ids])
            # if self.obs_actors[i] is not None:
            #     self.obs_actors[i].destroy()
        self.obs_actors.clear()
        self.obs_actor_ids.clear()

            
    def control_callback(self, msg): # take (deg, km/h)
        self.steer = msg.data[0]
        # print("MSG: ", msg.data)
        self.target_velocity = msg.data[1]*KMpH2MpS # [m/s]
        self.control_command_updated = 1

    def finish_traj_log_call(self, msg):
        self.autonomous_mode = False
        spawn_point = carla.Transform()
        spawn_point.location.x = self.saver_ego_pose[0]
        spawn_point.location.y = self.saver_ego_pose[1]
        spawn_point.rotation.yaw = self.saver_ego_pose[2]
        spawn_point.location.z += 0.5
        spawn_point.rotation.roll = 0.0
        spawn_point.rotation.pitch = 0.0
        self.player.set_transform(spawn_point)
        
        self.autonomous_mode = True

    def finish_flag_call(self, msg):
        if self.args.log:
            scenario_data = {
                            'ego_pose': self.saver_ego_pose,
                            'goal_pose': self.saver_goal_pose,
                            'obs_poses': self.saver_obs
                        }

            path = "/home/dyros-mk/catkin_ws/src/data_logging/data/"
            dir_list = sorted(os.listdir(path), key=int) # return list of the folder of the 'path' with 'string' form
            last_folder = dir_list[-1]
            pickle_file_addr = path + last_folder + "/scenario" + last_folder + ".pickle"
            print("dir_list: ", pickle_file_addr)
            # save
            with open(pickle_file_addr, 'wb') as f:
                pickle.dump(scenario_data, f, pickle.HIGHEST_PROTOCOL)
            
            self.restart_flag = True
        else:
            self.finish_call = True # Only for TEST

    def restart_flag_call(self, msg):
        print('Restart the scenario!!!')
        self.restart_flag = True
    
    def Callback_ManualGoal(self, msg):
        global M_Goal_X, M_Goal_Y, M_Goal_Angle

        quaternion = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )
        _, _, yaw = euler_from_quaternion(quaternion)
        # _, _, yaw = euler_from_quaternion(
        #     msg.pose.orientation.x,
        #     msg.pose.orientation.y,
        #     msg.pose.orientation.z,
        #     msg.pose.orientation.w)

        print(msg, yaw)

        M_Goal_X = msg.pose.position.x + self.x0
        M_Goal_Y = msg.pose.position.y + self.y0
        M_Goal_Angle = (yaw + self.th0) / DEG2RAD
# ===================================================================================

# ===================================================================================
    def ImageToOccupancyGrid(self, src):
        src = np.rot90(src, 3)
        # Assume Square
        Width = int(M_DIMENSION * int(1.0 / M_RESOL)) # --> grid dimension width
        Height = int(M_DIMENSION * int(1.0 / M_RESOL))
        dst = cv2.resize(src, dsize=(Width, Height), interpolation=cv2.INTER_LINEAR) # the pixel image to the grid map or INTER_AREA
        array = np.zeros((Width,Height))
        self.segImage = dst

        array[:,:] = dst[:,:,0] / 2.55
        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id = "map"
        grid_msg.info.resolution = M_RESOL
        grid_msg.info.width = Width
        grid_msg.info.height = Height
        grid_msg.info.origin = Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1))
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.data = np.reshape(array.swapaxes(1, 0).astype(np.uint8), (Width*Height,)).tolist()
        self.pubOccupancyGridMap.publish(grid_msg)

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================
class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        self.autonomous_mode = world.autonomous_mode

    def parse_events(self, client, world, clock, sync_mode):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart_random()
                        # world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_x:
                    scenario_data = {
                        'ego_pose': world.saver_ego_pose,
                        'goal_pose': world.saver_goal_pose,
                        'obs_poses': world.saver_obs
                    }
                    # save
                    with open('./scenario.pickle', 'wb') as f:
                        pickle.dump(scenario_data, f, pickle.HIGHEST_PROTOCOL)

                elif event.key == K_e and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_e:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB and world.cam_on:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE and world.cam_on:
                    world.camera_manager.next_sensor()
                elif event.key == K_n and world.cam_on:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception:
                            pass
                elif event.key > K_0 and event.key <= K_9 and world.cam_on:
                    index_ctrl = 0
                    if pygame.key.get_mods() & KMOD_CTRL:
                        index_ctrl = 9
                    world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                
                elif event.key == K_p: #publish current velocity, and set the current location as the new origin
                    # initialize
                    world.parking_flag0 = False
                    world.parking_flag1 = False
                    world.x0, world.y0, world.th0 = 0, 0, 0
                    world.parking_flag0 = True
                    print("@@@@@@@@@@@@@@@@@@ Path Planning!")

                    velData = Float32MultiArray()
                    velData.data = [round(world.vel, 2)]
                    world.pubVelData.publish(velData)
                    print("Velocity Pub! Now, DO-RRT starts!")
                
                elif event.key == K_v:# Switching mode (driver/autonomous)
                    self.autonomous_mode = not self.autonomous_mode
                    world.autonomous_mode = self.autonomous_mode
                    world.hud.notification('Autonomous Mode: {}'.format(world.autonomous_mode))

                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL) and world.cam_on:
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_y and (pygame.key.get_mods() & KMOD_CTRL) and world.cam_on:
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_y and not pygame.key.get_mods() & KMOD_CTRL:
                        if not self._autopilot_enabled and not sync_mode:
                            print("WARNING: You are currently in asynchronous mode and could "
                                  "experience some issues with the traffic simulation")
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    # elif event.key == K_z:
                    #     current_lights ^= carla.VehicleLightState.LeftBlinker
                    # elif event.key == K_x:
                    #     current_lights ^= carla.VehicleLightState.RightBlinker
                    elif event.key == K_u:
                        world.change_map = not world.change_map
                        world.hud.notification('Change Next Scenario: {}'.format(world.change_map))

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
            if world.autonomous_mode == False:
                world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1.00)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        # compass = world.imu_sensor.compass
        # heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        # heading += 'S' if 90.5 < compass < 269.5 else ''
        # heading += 'E' if 0.5 < compass < 179.5 else ''
        # heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        gear_state = world.gear_state if not world.finish_call else 'P'

        self._info_text = [
            # 'Server:  % 16.0f FPS' % self.server_fps,
            # 'Client:  % 16.0f FPS' % clock.get_fps(),
            # '',
            # 'Scenarion-Case: % 10s' % M_SCENARIO + '-' + M_CASE,
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            # 'Map:     % 20s' % world.map.name.split('/')[-1],
            # 'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            # '',
            # self.steer = msg.data[0]
            # self.target_velocity = msg.data[1]*KMpH2MpS
            'Gear State:   % 15s' % (gear_state),
            'Steering:   % 11.00f [deg]' % (world.steer),
            # 'Speed:   % 15.00f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            'Speed:   % 13.2f [km/h]' % (abs(world.target_velocity) / KMpH2MpS),
            # u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            # 'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            # 'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            # 'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            # 'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            # 'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl) and False:
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl) and False:
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        # self._info_text += [
        #     '',
        #     'Collision:',
        #     collision,
        #     '',
        #     'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1 and False:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((self.dim[0] * 0.44, self.dim[1] * 0.16)) #info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(200)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================
class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================
class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- LidarSensor -----------------------------------------------------------
# ==============================================================================
class LidarSensor(object):
    def __init__(self, parent_actor, world, semantic = False):
        self.sensor = None
        self.world = world
        self._parent = parent_actor
        simworld = self._parent.get_world()
        if semantic:
            lidar_bp = simworld.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        else:
            lidar_bp = simworld.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',str(32))
        lidar_bp.set_attribute('points_per_second',str(300000))
        lidar_bp.set_attribute('rotation_frequency',str(60))
        lidar_bp.set_attribute('lower_fov',str(-35.0))
        
        lidar_bp.set_attribute('range',str(45))
        lidar_location = carla.Location(0, 0, 2.7)
        lidar_rotation = carla.Rotation(0, 0, 0)
        lidar_transform = carla.Transform(lidar_location,lidar_rotation)
        self.sensor = simworld.spawn_actor(lidar_bp,lidar_transform, attach_to = self._parent)
        # weak_self = weakref.ref(self)

        if semantic:
            self.sensor.listen(lambda pointcloud: LidarSensor._LiDAR_callback_semantic(pointcloud, self.world))
        else:
            self.sensor.listen(lambda pointcloud: LidarSensor._LiDAR_callback(pointcloud, self.world))
        # self.sensor.listen(lambda point_cloud: point_cloud.save_to_disk('tutorial/new_lidar_output/%.6d.ply' % point_cloud.frame))

    def test(self):
        print("TEST")
    
    @staticmethod
    def _LiDAR_callback(point_cloud, world):
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        lidar_data = np.fromstring(bytes(point_cloud.raw_data), dtype=np.float32) # fromstring
        lidar_data = np.reshape(
            lidar_data, (int(lidar_data.shape[0] / 4), 4))
        # we take the opposite of y axis
        # (as lidar point are express in left handed coordinate system, and ros need right handed)
        lidar_data[:, 1] *= -1

        header = Header()
        header.frame_id = "map"
        header.stamp = rospy.Time.now()
        # lidar_data = lidar_data[:, :3]
        # print("lidara dara: ", lidar_data.shape)
        msg = create_cloud(header, fields, lidar_data)
        world.pubPointClouds.publish(msg)
        
    @staticmethod
    def _LiDAR_callback_semantic(point_cloud, world):
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        lidar_data = np.fromstring(bytes(point_cloud.raw_data), dtype=np.float32) # fromstring
        lidar_data = np.reshape(
            lidar_data, (int(lidar_data.shape[0] / 6), 6))
        # we take the opposite of y axis
        # (as lidar point are express in left handed coordinate system, and ros need right handed)
        lidar_data[:, 1] *= -1
        # lidar_data[:, 2] += 2.35
        lidar_data = lidar_data[:, [0, 1, 2, 5]]

        header = Header()
        header.frame_id = "map"
        header.stamp = rospy.Time.now()
        # lidar_data = lidar_data[:, :3]
        # print("lidara dara: ", lidar_data[:, -1])
        msg = create_cloud(header, fields, lidar_data)
        world.pubPointClouds.publish(msg)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================
class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))
        self.is_collision = False

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        # self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        self.is_collision = True
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================
class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================
class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================
class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================
class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================
class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.sensor2 = None # weird... With including sensor2, it decreases the server fps, monotonically.
        self.parking_cand = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.image = None
        self.image2 = None
        self.recording = False
        Attachment = carla.AttachmentType
        self.bridge = cv_bridge.CvBridge()

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=0.0, z=16.0), carla.Rotation(roll = 90.0, pitch = -90.0, yaw = 90.0)), Attachment.Rigid),# default AVM setting
                (carla.Transform(carla.Location(x = 0.12, y=0, z=1.8), carla.Rotation(pitch = -10.0)), Attachment.Rigid),# front view
                ]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)
                ]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}], 
            # ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {'fov': str(160)}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            # ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            # ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            
            # if self.sensor2 is not None:# #MSK: You should change that index of 'self.sensors' for parshing image
            #     self.sensor2.destroy()
            # self.sensor2 = self._parent.get_world().spawn_actor(
            #     self.sensors[1][-1],
            #     self._camera_transforms[self.transform_index][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))# Get true RGB
            # self.sensor2.listen(lambda image: CameraManager._parse_image(weak_self, image, raw_seg=True))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image, raw_seg=False):
        self = weak_self()
        ind = self.index if not raw_seg else 1
        
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])# (x,y)
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[ind][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            if raw_seg: #segmentation image
                seg = np.copy(array)
                seg[seg[:,:,2]<=100] = 0
                # seg[159:256,186:231,:] = 0# for removing the smaller vehicle (vertical x horizontal)
                # seg[183:233,196:220,:] = 0# for removing the smaller vehicle
                seg[169:247,189:227,:] = 0  # for removing the larger TESLA vehicle (vertical x horizontal) when AVM's z is '13,0'    
                self.image2 = np.copy(seg)
            else:
                array = np.rot90(array, 1) if self.transform_index == 0 else array# np.rot90(image, iteration)
                self.image = np.copy(array)
                self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================
class SegmentationTopViewCamera(object):
    def __init__(self, parent_actor, hud, gamma_correction, img_x=480, img_y=480, xx=0.0, yy=0.0, zz=16.0):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.image = None
        Attachment = carla.AttachmentType
        self.bridge = cv_bridge.CvBridge()
        self._camera_transforms = (carla.Transform(carla.Location(x=xx, y=yy, z=zz), carla.Rotation(roll = 90.0, pitch = -90.0, yaw = 90.0)), Attachment.Rigid)# default AVM setting
        self.seg_option = ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {'fov': str(90)}]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        
        bp = bp_library.find(self.seg_option[0])
        if self.seg_option[0].startswith('sensor.camera'):
            bp.set_attribute('image_size_x', str(img_x))
            bp.set_attribute('image_size_y', str(img_y))
            if bp.has_attribute('gamma'):
                bp.set_attribute('gamma', str(gamma_correction))
            for attr_name, attr_value in self.seg_option[3].items():
                bp.set_attribute(attr_name, attr_value)

            self.seg_option.append(bp)
        self.resolution = zz / img_x # assume that the image has a square-shape, then it is going to have a meter/pixel
        # print("Resolution: ", self.resolution)

    def set_sensor(self):
        if self.sensor is not None:
            self.sensor.destroy()
        self.sensor = self._parent.get_world().spawn_actor(
            self.seg_option[-1],
            self._camera_transforms[0],
            attach_to=self._parent,
            attachment_type=self._camera_transforms[1])

        # We need to pass the lambda a weak reference to self to avoid
        # circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: SegmentationTopViewCamera._parse_image(weak_self, image))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        ind = 0        
        if not self:
            return
        
        image.convert(self.seg_option[1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = np.rot90(array, 1)
        seg = np.copy(array)

        # seg[seg[:,:,2]<=100] = 0
        
        # change the pixels // If you want
        # __unlabeled = (seg == [0., 0., 0.]).all(axis=2)
        # __curb = (seg == [255., 0., 0.]).all(axis=2)
        # __vehicle = (seg == [0., 0., 142.]).all(axis=2)
        __road_line = (seg == [157., 234., 50.]).all(axis=2)
        __road = (seg == [128., 64., 128.]).all(axis=2)

        # seg[__unlabeled] = [251, 0, 251]
        # seg[__curb] = [251, 0, 255]
        # seg[__vehicle] = [255, 255, 255]
        seg[...] = 255
        seg[__road_line] = [0, 0, 0]
        seg[__road] = [0, 0, 0]
        
        seg[222:258,203:276,:] = 0  # for removing the larger TESLA vehicle (vertical x horizontal) when AVM's z is '13,0'    
        self.image = np.copy(seg)
    
    def _destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()

# def newAVM_maker(front_img, rear_img, right_img, left_img):
#     if front_img is not None and rear_img is not None and right_img is not None and left_img is not None:
#         # print(front_img.shape)
#         # input coordinates in pixel resolution.
#         p1 = [273, 202]
#         p2 = [527, 202]
#         p3 = [-921, 600]
#         p4 = [1721, 600]

#         # p1 = [222, 293]
#         # p2 = [578, 293]
#         # p3 = [-699, 600]
#         # p4 = [1499, 600]
        

#         background_image = np.zeros(shape=[200, 200, 3], dtype=np.uint8)
#         avm_img_h, avm_img_w, _ = background_image.shape

#         src_points = np.array([p1,p2,p3,p4], dtype = np.float32)
#         f_dst_points = np.array([[0, 0],[avm_img_w, 0],[0, avm_img_h],[avm_img_w, avm_img_h]], dtype=np.float32)
#         f_perspective = cv2.getPerspectiveTransform(src_points, f_dst_points)
#         front_warpped_image = cv2.warpPerspective(front_img, f_perspective, (avm_img_w, avm_img_h))
#         cv2.imshow("front_warpped_image", front_img)
#         cv2.waitKey(1)

def AVM_maker(front_img, rear_img, right_img, left_img): # YW AVM maker
    if front_img is not None and rear_img is not None and right_img is not None and left_img is not None:
        background_image = np.zeros(shape=[480, 480, 3], dtype=np.uint8)

        # input coordinates in pixel resolution.
        p1 = [346.25, 297] # [Left-Top]
        p2 = [453.75, 297] # [Right-Top]
        p3 = [2093.2, 600] # [Right-Bottom]
        p4 = [-1293.2, 600] # [Left-Bottom]

        avm_img_h, avm_img_w, avm_img_c = background_image.shape

        # input point coordinates
        src_points = np.array([p1,p2,p3,p4], dtype = np.float32)

        # output point coordinates, // Should do it manually. Currently, our target-resolution is 0.05 [m/px]. 
        # Therefore, from the center, the vehicle looks upto 12 m forward. 8 m right.
        f_dst_points = np.array([[int(avm_img_w/2-90),int(avm_img_h/2-198)],
                                 [int(avm_img_w/2+90),int(avm_img_h/2-198)],
                                 [int(avm_img_w/2+90),int(avm_img_h/2-48)],
                                 [int(avm_img_w/2-90),int(avm_img_h/2-48)]], dtype=np.float32)
        b_dst_points = np.array([[int(avm_img_w/2+90),int(avm_img_h/2+199)],
                                 [int(avm_img_w/2-90),int(avm_img_h/2+199)],
                                 [int(avm_img_w/2-90),int(avm_img_h/2+48)],
                                 [int(avm_img_w/2+90),int(avm_img_h/2+48)]], dtype=np.float32)
        r_dst_points = np.array([[int(avm_img_w/2+133),int(avm_img_h/2-112)],
                                 [int(avm_img_w/2+133),int(avm_img_h/2+112)],
                                 [int(avm_img_w/2+16), int(avm_img_h/2+112)],
                                 [int(avm_img_w/2+16), int(avm_img_h/2-112)]], dtype=np.float32)
        l_dst_points = np.array([[int(avm_img_w/2-133),int(avm_img_h/2+112)],
                                 [int(avm_img_w/2-133),int(avm_img_h/2-112)],
                                 [int(avm_img_w/2-16), int(avm_img_h/2-112)],
                                 [int(avm_img_w/2-16), int(avm_img_h/2+112)]], dtype=np.float32)

        f_perspective = cv2.getPerspectiveTransform(src_points, f_dst_points)
        b_perspective = cv2.getPerspectiveTransform(src_points, b_dst_points)
        r_perspective = cv2.getPerspectiveTransform(src_points, r_dst_points)
        l_perspective = cv2.getPerspectiveTransform(src_points, l_dst_points)

        front_warpped_image = cv2.warpPerspective(front_img, f_perspective, (avm_img_w, avm_img_h))
        back_warpped_image = cv2.warpPerspective(rear_img, b_perspective, (avm_img_w, avm_img_h))
        right_warpped_image = cv2.warpPerspective(right_img, r_perspective, (avm_img_w, avm_img_h))
        left_warpped_image = cv2.warpPerspective(left_img, l_perspective, (avm_img_w, avm_img_h))

        front_warpped_image[195:, ...] = 0.0
        back_warpped_image[:288, ...] = 0.0

        # Create a mask with the same dimensions as Image A
        maskF = np.zeros(front_warpped_image.shape[:2], dtype=np.uint8)
        maskB = np.zeros(back_warpped_image.shape[:2], dtype=np.uint8)
        maskR = np.zeros(right_warpped_image.shape[:2], dtype=np.uint8)
        maskL = np.zeros(left_warpped_image.shape[:2], dtype=np.uint8)

        # Define the coordinates of the ROI polygon in Image A
        roi_points_F = np.array([[79, 0], [401, 0], [258, 206], [222, 206]])
        roi_points_B = np.array([[222, 274], [258, 274], [401, 479], [79, 479]])
        roi_points_R = np.array([[258, 206], [401, 0], [479, 0], [479, 479], [401, 479], [258, 274]])
        roi_points_L = np.array([[0, 0], [79, 0], [222, 206], [222, 274], [79, 479], [0, 479]])

        # Fill the ROI polygon in the mask with white color (255)
        cv2.fillPoly(maskF, [roi_points_F], 255)
        cv2.fillPoly(maskB, [roi_points_B], 255)
        cv2.fillPoly(maskR, [roi_points_R], 255)
        cv2.fillPoly(maskL, [roi_points_L], 255)

        # Extract the pixels within the ROI from Image A using the mask
        front_warpped_image = cv2.bitwise_and(front_warpped_image, front_warpped_image, mask=maskF)
        back_warpped_image = cv2.bitwise_and(back_warpped_image, back_warpped_image, mask=maskB)
        right_warpped_image = cv2.bitwise_and(right_warpped_image, right_warpped_image, mask=maskR)
        left_warpped_image = cv2.bitwise_and(left_warpped_image, left_warpped_image, mask=maskL)

        avm_image_bgr = front_warpped_image+back_warpped_image+right_warpped_image+left_warpped_image
        # cv2.imshow("Front Image", avm_image_bgr)
        # cv2.imshow("Right Image", right_warpped_image)
        cv2.waitKey(1)

        return avm_image_bgr
    else:
        return None


# def AVM_maker(front_img, rear_img, right_img, left_img): # YW AVM maker
#     param = -18 # slope
#     param1 = 14 # trans rl
#     param2 = 10 # trans fr

#     param3 = 24
#     param4 = 30

#     if front_img is not None and rear_img is not None and right_img is not None and left_img is not None:

#         background_image = np.zeros(shape=[344+2*param2, 266+2*param1, 3], dtype=np.uint8)

#         img_height, img_width, _ = front_img.shape

#         # input coordinates in pixel resolution.
#         p1 = [330, 300]
#         p2 = [470, 300]
#         p3 = [2093.2, 600]
#         p4 = [-1293.2, 600]

#         avm_img_h, avm_img_w, avm_img_c = background_image.shape
#         # favm_img_h, favm_img_w, favn_img_c = fbackground_image.shape

#         # cv2.imshow("front_img", front_img)

#         # input point coordinates
#         src_points = np.array([p1,p2,p3,p4], dtype = np.float32)

#         # output point coordinates
#         f_dst_points = np.array([[int(avm_img_w/2-110-18),int(avm_img_h/2-172+28)],[int(avm_img_w/2+110+18),int(avm_img_h/2-172+28)],[int(avm_img_w/2+110+18),int(avm_img_h/2-20)],[int(avm_img_w/2-110-18),int(avm_img_h/2-20)]], dtype=np.float32)
#         b_dst_points = np.array([[int(avm_img_w/2+110+18),int(avm_img_h/2+172-28)],[int(avm_img_w/2-110-18),int(avm_img_h/2+172-28)],[int(avm_img_w/2-110-18),int(avm_img_h/2+20)],[int(avm_img_w/2+110+18),int(avm_img_h/2+20)]], dtype=np.float32)
#         r_dst_points = np.array([[int(avm_img_w/2+130),int(avm_img_h/2-110-18)],[int(avm_img_w/2+130),int(avm_img_h/2+110+18)],[int(avm_img_w/2+6),int(avm_img_h/2+110+18)],[int(avm_img_w/2+6),int(avm_img_h/2-110-18)]], dtype=np.float32)
#         l_dst_points = np.array([[int(avm_img_w/2-130),int(avm_img_h/2+110+18)],[int(avm_img_w/2-130),int(avm_img_h/2-110-18)],[int(avm_img_w/2-6),int(avm_img_h/2-110-18)],[int(avm_img_w/2-6),int(avm_img_h/2+110+18)]], dtype=np.float32)

#         f_perspective = cv2.getPerspectiveTransform(src_points, f_dst_points)
#         b_perspective = cv2.getPerspectiveTransform(src_points, b_dst_points)
#         r_perspective = cv2.getPerspectiveTransform(src_points, r_dst_points)
#         l_perspective = cv2.getPerspectiveTransform(src_points, l_dst_points)

#         front_warpped_image = cv2.warpPerspective(front_img, f_perspective, (avm_img_w, avm_img_h))
#         back_warpped_image = cv2.warpPerspective(rear_img, b_perspective, (avm_img_w, avm_img_h))
#         right_warpped_image = cv2.warpPerspective(right_img, r_perspective, (avm_img_w, avm_img_h))
#         left_warpped_image = cv2.warpPerspective(left_img, l_perspective, (avm_img_w, avm_img_h))

#         front_cropped_image1 = front_warpped_image[0:79, 0:avm_img_w]
#         front_cropped_image1_h, front_cropped_image1_w, front_cropped_image1_c = front_cropped_image1.shape
#         resize_front_cropped_image1 = cv2.resize(front_cropped_image1, (front_cropped_image1_w, int(79.0*12.0/11.0)))
#         resize_front_cropped_image1_h, resize_front_cropped_image1_w, resize_front_cropped_image1_c = resize_front_cropped_image1.shape
#         front_warpped_image[0:79, 0:avm_img_w] = resize_front_cropped_image1[resize_front_cropped_image1_h-79:resize_front_cropped_image1_h,0:avm_img_w]
        
#         back_cropped_image1 = back_warpped_image[avm_img_h-79:avm_img_h, 0:avm_img_w]
#         back_cropped_image1_h, back_cropped_image1_w, back_cropped_image1_c = back_cropped_image1.shape
#         resize_back_cropped_image1 = cv2.resize(back_cropped_image1, (back_cropped_image1_w, int(79.0*12.0/11.0)))
#         back_warpped_image[avm_img_h-79:avm_img_h, 0:avm_img_w] = resize_back_cropped_image1[0:79,0:avm_img_w]

#         left_cropped_image1 = left_warpped_image[0:avm_img_h, 0:45]
#         left_cropped_image1_h, left_cropped_image1_w, left_cropped_image1_c = left_cropped_image1.shape
#         resize_left_cropped_image1 = cv2.resize(left_cropped_image1, (int(45.0*12.0/11.0), left_cropped_image1_h))
#         resize_left_cropped_image1_h, resize_left_cropped_image1_w, resize_left_cropped_image1_c = resize_left_cropped_image1.shape
#         left_warpped_image[0:avm_img_h, 0:45] = resize_left_cropped_image1[0:avm_img_h,resize_left_cropped_image1_w-45:resize_left_cropped_image1_w]

#         right_cropped_image1 = right_warpped_image[0:avm_img_h, avm_img_w-45:avm_img_w]
#         right_cropped_image1_h, right_cropped_image1_w, right_cropped_image1_c = right_cropped_image1.shape
#         resize_right_cropped_image1 = cv2.resize(right_cropped_image1, (int(45.0*12.0/11.0), right_cropped_image1_h))
#         resize_left_cropped_image1_h, resize_left_cropped_image1_w, resize_left_cropped_image1_c = resize_left_cropped_image1.shape
#         right_warpped_image[0:avm_img_h, avm_img_w-45:avm_img_w] = resize_right_cropped_image1[0:avm_img_h,0:45]

#         front_warpped_image[int(avm_img_h/2-20):avm_img_h, 0:avm_img_w, :] = [0,0,0]
#         back_warpped_image[0:int(avm_img_h/2+20), 0:avm_img_w, :] = [0,0,0]
#         right_warpped_image[0:avm_img_h, 0:int(avm_img_w/2+6), :] = [0,0,0]
#         left_warpped_image[0:avm_img_h, int(avm_img_w/2-6):avm_img_w, :] = [0,0,0]

#         f_stencil = np.zeros(background_image.shape).astype(background_image.dtype)
#         b_stencil = np.zeros(background_image.shape).astype(background_image.dtype)
#         r_stencil = np.zeros(background_image.shape).astype(background_image.dtype)
#         l_stencil = np.zeros(background_image.shape).astype(background_image.dtype)
        
#         # f_roi = np.array([[[int(avm_img_w/2-133-param1),int(avm_img_h/2-172-param2)],[int(avm_img_w/2+133+param1),int(avm_img_h/2-172-param2)],[int(avm_img_w/2),int(avm_img_h/2)]]],dtype=np.int32)
#         # b_roi = np.array([[[int(avm_img_w/2+133+param1),int(avm_img_h/2+172+param2)],[int(avm_img_w/2-133-param1),int(avm_img_h/2+172+param2)],[int(avm_img_w/2),int(avm_img_h/2)]]],dtype=np.int32)
#         # r_roi = np.array([[[int(avm_img_w/2+133+param1),int(avm_img_h/2-172-param2)],[int(avm_img_w/2+133+param1),int(avm_img_h/2+172+param2)],[int(avm_img_w/2),int(avm_img_h/2)]]],dtype=np.int32)
#         # l_roi = np.array([[[int(avm_img_w/2-133-param1),int(avm_img_h/2+172+param2)],[int(avm_img_w/2-133-param1),int(avm_img_h/2-172-param2)],[int(avm_img_w/2),int(avm_img_h/2)]]],dtype=np.int32)

#         f_roi = np.array([[[int(avm_img_w/2-133-param1+param3),int(avm_img_h/2-172-param2+param4)],[int(avm_img_w/2+133+param1-param3),int(avm_img_h/2-172-param2+param4)],[int(avm_img_w/2),int(avm_img_h/2)]]],dtype=np.int32)
#         b_roi = np.array([[[int(avm_img_w/2+133+param1-param3),int(avm_img_h/2+172+param2-param4)],[int(avm_img_w/2-133-param1+param3),int(avm_img_h/2+172+param2-param4)],[int(avm_img_w/2),int(avm_img_h/2)]]],dtype=np.int32)
#         r_roi = np.array([[[int(avm_img_w/2+133+param1-param3),int(avm_img_h/2-172-param2+param4)],[int(avm_img_w/2+133+param1-param3),int(avm_img_h/2+172+param2-param4)],[int(avm_img_w/2),int(avm_img_h/2)]]],dtype=np.int32)
#         l_roi = np.array([[[int(avm_img_w/2-133-param1+param3),int(avm_img_h/2+172+param2-param4)],[int(avm_img_w/2-133-param1+param3),int(avm_img_h/2-172-param2+param4)],[int(avm_img_w/2),int(avm_img_h/2)]]],dtype=np.int32)

#         cv2.fillPoly(f_stencil,f_roi,[255, 255, 255])
#         cv2.fillPoly(b_stencil,b_roi,[255, 255, 255])
#         cv2.fillPoly(r_stencil,r_roi,[255, 255, 255])
#         cv2.fillPoly(l_stencil,l_roi,[255, 255, 255])

#         f_selected = f_stencil != 255
#         b_selected = b_stencil != 255
#         r_selected = r_stencil != 255
#         l_selected = l_stencil != 255

#         front_warpped_image[f_selected] = 0
#         back_warpped_image[b_selected] = 0
#         right_warpped_image[r_selected] = 0
#         left_warpped_image[l_selected] = 0

#         avm_image_bgr = front_warpped_image+back_warpped_image+right_warpped_image+left_warpped_image

#         return avm_image_bgr

# def prevAVM_maker(front_img, rear_img, right_img, left_img): # YW AVM maker
#     param = -18 # slope
#     param1 = 14 # trans rl; 14
#     param2 = 10 # trans fr; 10

#     param3 = 24
#     param4 = 30

#     if front_img is not None and rear_img is not None and right_img is not None and left_img is not None:

#         background_image = np.zeros(shape=[344+2*param2, 266+2*param1, 3], dtype=np.uint8)

#         # input coordinates in pixel resolution.
#         # p1 = [330, 299]
#         # p2 = [470, 299]
#         # p3 = [2098.6, 600]
#         # p4 = [-1298.6, 600]

#         # # input coordinates in pixel resolution.
#         p1 = [330, 300]
#         p2 = [470, 300]
#         p3 = [2093.2, 600]
#         p4 = [-1293.2, 600]

#         avm_img_h, avm_img_w, avm_img_c = background_image.shape
#         # favm_img_h, favm_img_w, favn_img_c = fbackground_image.shape

#         # cv2.imshow("front_img", front_img)

#         # input point coordinates
#         src_points = np.array([p1,p2,p3,p4], dtype = np.float32)

#         # output point coordinates
#         f_dst_points = np.array([[int(avm_img_w/2-110+param),int(avm_img_h/2-172-param2)],[int(avm_img_w/2+110-param),int(avm_img_h/2-172-param2)],[int(avm_img_w/2+110-param),int(avm_img_h/2-48-param2)],[int(avm_img_w/2-110+param),int(avm_img_h/2-48-param2)]], dtype=np.float32)
#         b_dst_points = np.array([[int(avm_img_w/2+110-param),int(avm_img_h/2+172+param2)],[int(avm_img_w/2-110+param),int(avm_img_h/2+172+param2)],[int(avm_img_w/2-110+param),int(avm_img_h/2+48+param2)],[int(avm_img_w/2+110-param),int(avm_img_h/2+48+param2)]], dtype=np.float32)
#         r_dst_points = np.array([[int(avm_img_w/2+133+param1),int(avm_img_h/2-110+param)],[int(avm_img_w/2+133+param1),int(avm_img_h/2+110-param)],[int(avm_img_w/2+9+param1),int(avm_img_h/2+110-param)],[int(avm_img_w/2+9+param1),int(avm_img_h/2-110+param)]], dtype=np.float32)
#         l_dst_points = np.array([[int(avm_img_w/2-133-param1),int(avm_img_h/2+110-param)],[int(avm_img_w/2-133-param1),int(avm_img_h/2-110+param)],[int(avm_img_w/2-9-param1),int(avm_img_h/2-110+param)],[int(avm_img_w/2-9-param1),int(avm_img_h/2+110-param)]], dtype=np.float32)

#         f_perspective = cv2.getPerspectiveTransform(src_points, f_dst_points)
#         b_perspective = cv2.getPerspectiveTransform(src_points, b_dst_points)
#         r_perspective = cv2.getPerspectiveTransform(src_points, r_dst_points)
#         l_perspective = cv2.getPerspectiveTransform(src_points, l_dst_points)

#         front_warpped_image = cv2.warpPerspective(front_img, f_perspective, (avm_img_w, avm_img_h))
#         back_warpped_image = cv2.warpPerspective(rear_img, b_perspective, (avm_img_w, avm_img_h))
#         right_warpped_image = cv2.warpPerspective(right_img, r_perspective, (avm_img_w, avm_img_h))
#         left_warpped_image = cv2.warpPerspective(left_img, l_perspective, (avm_img_w, avm_img_h))

#         front_cropped_image1 = front_warpped_image[0:79, 0:avm_img_w]
#         front_cropped_image1_h, front_cropped_image1_w, front_cropped_image1_c = front_cropped_image1.shape
#         resize_front_cropped_image1 = cv2.resize(front_cropped_image1, (front_cropped_image1_w, int(79.0*12.0/11.0)))
#         resize_front_cropped_image1_h, resize_front_cropped_image1_w, resize_front_cropped_image1_c = resize_front_cropped_image1.shape
#         front_warpped_image[0:79, 0:avm_img_w] = resize_front_cropped_image1[resize_front_cropped_image1_h-79:resize_front_cropped_image1_h,0:avm_img_w]
        
#         back_cropped_image1 = back_warpped_image[avm_img_h-79:avm_img_h, 0:avm_img_w]
#         back_cropped_image1_h, back_cropped_image1_w, back_cropped_image1_c = back_cropped_image1.shape
#         resize_back_cropped_image1 = cv2.resize(back_cropped_image1, (back_cropped_image1_w, int(79.0*12.0/11.0)))
#         back_warpped_image[avm_img_h-79:avm_img_h, 0:avm_img_w] = resize_back_cropped_image1[0:79,0:avm_img_w]

#         left_cropped_image1 = left_warpped_image[0:avm_img_h, 0:45]
#         left_cropped_image1_h, left_cropped_image1_w, left_cropped_image1_c = left_cropped_image1.shape
#         resize_left_cropped_image1 = cv2.resize(left_cropped_image1, (int(45.0*12.0/11.0), left_cropped_image1_h))
#         resize_left_cropped_image1_h, resize_left_cropped_image1_w, resize_left_cropped_image1_c = resize_left_cropped_image1.shape
#         left_warpped_image[0:avm_img_h, 0:45] = resize_left_cropped_image1[0:avm_img_h,resize_left_cropped_image1_w-45:resize_left_cropped_image1_w]

#         right_cropped_image1 = right_warpped_image[0:avm_img_h, avm_img_w-45:avm_img_w]
#         right_cropped_image1_h, right_cropped_image1_w, right_cropped_image1_c = right_cropped_image1.shape
#         resize_right_cropped_image1 = cv2.resize(right_cropped_image1, (int(45.0*12.0/11.0), right_cropped_image1_h))
#         resize_left_cropped_image1_h, resize_left_cropped_image1_w, resize_left_cropped_image1_c = resize_left_cropped_image1.shape
#         right_warpped_image[0:avm_img_h, avm_img_w-45:avm_img_w] = resize_right_cropped_image1[0:avm_img_h,0:45]

#         front_warpped_image[int(avm_img_h/2-48-param2):avm_img_h, 0:avm_img_w, :] = [0,0,0]
#         back_warpped_image[0:int(avm_img_h/2+48+param2), 0:avm_img_w, :] = [0,0,0]
#         right_warpped_image[0:avm_img_h, 0:int(avm_img_w/2+9+param1), :] = [0,0,0]
#         left_warpped_image[0:avm_img_h, int(avm_img_w/2-9-param1):avm_img_w, :] = [0,0,0]

#         f_stencil = np.zeros(background_image.shape).astype(background_image.dtype)
#         b_stencil = np.zeros(background_image.shape).astype(background_image.dtype)
#         r_stencil = np.zeros(background_image.shape).astype(background_image.dtype)
#         l_stencil = np.zeros(background_image.shape).astype(background_image.dtype)
        
#         f_roi = np.array([[[int(avm_img_w/2-133-param1+param3),int(avm_img_h/2-172-param2+param4)],[int(avm_img_w/2+133+param1-param3),int(avm_img_h/2-172-param2+param4)],[int(avm_img_w/2),int(avm_img_h/2)]]],dtype=np.int32)
#         b_roi = np.array([[[int(avm_img_w/2+133+param1-param3),int(avm_img_h/2+172+param2-param4)],[int(avm_img_w/2-133-param1+param3),int(avm_img_h/2+172+param2-param4)],[int(avm_img_w/2),int(avm_img_h/2)]]],dtype=np.int32)
#         r_roi = np.array([[[int(avm_img_w/2+133+param1-param3),int(avm_img_h/2-172-param2+param4)],[int(avm_img_w/2+133+param1-param3),int(avm_img_h/2+172+param2-param4)],[int(avm_img_w/2),int(avm_img_h/2)]]],dtype=np.int32)
#         l_roi = np.array([[[int(avm_img_w/2-133-param1+param3),int(avm_img_h/2+172+param2-param4)],[int(avm_img_w/2-133-param1+param3),int(avm_img_h/2-172-param2+param4)],[int(avm_img_w/2),int(avm_img_h/2)]]],dtype=np.int32)

#         cv2.fillPoly(f_stencil,f_roi,[255, 255, 255])
#         cv2.fillPoly(b_stencil,b_roi,[255, 255, 255])
#         cv2.fillPoly(r_stencil,r_roi,[255, 255, 255])
#         cv2.fillPoly(l_stencil,l_roi,[255, 255, 255])

#         f_selected = f_stencil != 255
#         b_selected = b_stencil != 255
#         r_selected = r_stencil != 255
#         l_selected = l_stencil != 255

#         front_warpped_image[f_selected] = 0
#         back_warpped_image[b_selected] = 0
#         right_warpped_image[r_selected] = 0
#         left_warpped_image[l_selected] = 0

#         avm_image_bgr = front_warpped_image+back_warpped_image+right_warpped_image+left_warpped_image

#         return avm_image_bgr
#     else:
#         return None


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        client.load_world('Town10HD')
        client.reload_world

        sim_world = client.get_world()
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                settings.no_rendering_mode = True # add...
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(client, sim_world, hud, args)
        controller = KeyboardControl(world, args.autopilot)


        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()
        clock = pygame.time.Clock()

        vel_pub_flag = False
        loop_counter = 0
        scenario_start_time = rospy.get_time()
        while True:
            if args.sync:
                sim_world.tick() # When this command is not enabled even if --sync, display will not reflect the custom command (just freeze)
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync):
                return
            
            if world.restart_flag:
                world.restart_flag = False
                # prev_case_count = world.case_count
                world.autonomous_mode = False
                world.restart_random()
                world.steer = 0
                # re-planning msg to Infromed RRT* node 
                msg_ = Float32MultiArray()
                msg_.data = [1]
                world.pubReplanning.publish(msg_)
                print("sleeping for 1 secs...")
                time.sleep(1)

                vel_pub_flag = True
                loop_counter = 0
                
                # initialize
                world.parking_flag0 = False
                world.parking_flag1 = False
                world.x0, world.y0, world.th0 = 0, 0, 0
                world.parking_flag0 = True
                scenario_start_time = rospy.get_time()
                # if world.collision_sensor.is_collision:
                #     print("Collision Occur...Reset!")
                #     world.restart_flag = True

                
            # Vehicle Control
            v = world.player.get_velocity()
            curr_v = math.sqrt(v.x**2+v.y**2+v.z**2)*3.6 # [km/h]
            localizationData = Float32MultiArray()# [x, y, heading, velocity]
            steerData = Float32MultiArray()
            steerData.data = [(world.steer / 540)*1.0]
            # steerData.data = [world.player.get_wheel_steer_angle()] // needs localization data of each wheel, not from single data
            gearData = Int32()
            gearData.data = int(controller._control.gear)

            p_loc = world.player.get_transform()
            # print("p: ", p_loc)
            x = p_loc.location.x
            y = p_loc.location.y
            th = p_loc.rotation.yaw*DEG2RAD
            localizationData.data = [x, y, th, round(curr_v, 2)]
            xx, yy, thh = global2local(localizationData.data[0], localizationData.data[1], localizationData.data[2], M_Goal_X, M_Goal_Y, M_Goal_Angle * DEG2RAD)

            # Parking path planning start
            if world.parking_flag0:
                world.parking_flag0 = False
                world.x0 = p_loc.location.x
                world.y0 = p_loc.location.y
                world.th0 = DEG2RAD*p_loc.rotation.yaw
                world.parking_flag1 = True

            # obstacle pose visualize
            obs_poses = PoseArray()
            obs_poses.header.frame_id = "map"
            obs_poses.header.stamp = rospy.Time.now()
            for data in world.obs_pose_lists:
                x__ = data[0]
                y__ = data[1]
                th__ = data[2] * DEG2RAD
                xx_, yy_, thh_ = global2local(world.x0, world.y0, world.th0, x__, y__, th__)
                obs_poses.poses.append(poseArray(xx_, yy_, thh_))
            world.pub_obs_poses.publish(obs_poses)


            # The origin of a global cooridnate is the center of the vehicle rear wheel from now on.
            if world.parking_flag1:
                xx, yy, thh = global2local(world.x0, world.y0, world.th0, localizationData.data[0], localizationData.data[1], localizationData.data[2])
                localizationData.data = [xx, yy, thh, round(curr_v, 2)] # thh \in [0, 2*pi]
                world.pubLocalizationData.publish(localizationData)# Publish Localization Data
                world.pubSteerData.publish(steerData)
                world.pubGearData.publish(gearData)
                world.pub_obs_info.publish(world.obs_info)
                if len(world.obs_info.data) != 100:
                    # print("not 100")
                    pass
                xx, yy, thh = global2local(world.x0, world.y0, world.th0, M_Goal_X, M_Goal_Y, M_Goal_Angle * DEG2RAD)
                # tmp = xx
                # xx = yy
                # yy = tmp
                # thh = -thh
                
            else:
                localizationData.data = [0.0, 0.0, 0.0, round(curr_v, 2)]    
                world.pubLocalizationData.publish(localizationData)
                world.pubSteerData.publish(steerData)
                world.pubGearData.publish(gearData)
                # print(len(world.obs_info.data))
                world.pub_obs_info.publish(world.obs_info)
                if len(world.obs_info.data) != 100:
                    # print("not 100")
                    pass

            if world.rl_train_mode and world.collision_sensor.is_collision:       
                collisionData = Int32()
                collisionData.data = int(1)
                world.pub_collision.publish(collisionData)
                terminal_case_Data = Int32()
                terminal_case_Data = int(2)
                world.pub_terminal_case.publish(terminal_case_Data)
                print("Collision Occur...Reset!")
                world.steer = 0.0
                world.target_velocity = 0.0
            # print(world.world.get_actors().filter('vehicle.*')[3])
            # print(world.world.get_actors().filter('vehicle.*')[3].bounding_box.extent)
            
            if world.rl_train_mode and (rospy.get_time() - scenario_start_time) > 30:
                terminal_case_Data = Int32()
                terminal_case_Data = int(3)
                world.pub_terminal_case.publish(terminal_case_Data)
                print("Timeout...Reset!")
                world.steer = 0.0
                world.target_velocity = 0.0
                

            if world.avm_on:
                # AVM
                raw_front_img = world.seg_avm_front_camera.image 
                raw_rear_img = world.seg_avm_back_camera.image                
                raw_right_img = world.seg_avm_right_camera.image                
                raw_left_img = world.seg_avm_left_camera.image
                raw_avm_image = AVM_maker(raw_front_img, raw_rear_img, raw_right_img, raw_left_img)
                if raw_avm_image is not None:
                    cv_msg = world.seg_camera.bridge.cv2_to_imgmsg(raw_avm_image, "rgb8")
                    world.pub_carla_AVM.publish(cv_msg)
                    # cv2.imshow('AVM', raw_avm_image)
                    # cv2.waitKey(1)

                seg_front_img = world.seg_avm_front_camera2.image 
                seg_rear_img = world.seg_avm_back_camera2.image
                seg_right_img = world.seg_avm_right_camera2.image
                seg_left_img = world.seg_avm_left_camera2.image
                seg_avm_image = AVM_maker(seg_front_img, seg_rear_img, seg_right_img, seg_left_img)
                if seg_avm_image is not None:
                    cv_msg = world.seg_camera.bridge.cv2_to_imgmsg(seg_avm_image, "rgb8")
                    world.pub_carla_AVMseg.publish(cv_msg)
                    # cv2.imshow('Seg AVM', seg_avm_image)
                    # cv2.waitKey(1)

            # Publish Parking cand. 
            cands = PoseArray()
            cands.header.frame_id = "map"
            cands.header.stamp = rospy.Time.now()
            if M_ACCURATE_GOAL:
                cands.poses.append(poseArray(xx, yy, thh))
            else:
                cands.poses.append(poseArray(np.random.normal(xx, goal_std(math.sqrt(xx**2 + yy**2))), 
                                            np.random.normal(yy, goal_std(math.sqrt(xx**2 + yy**2))), 
                                            np.random.normal(thh, goal_std(math.sqrt(xx**2 + yy**2), stdmax=2, stdmin=1)*DEG2RAD)))
            world.pubParkingCand.publish(cands)

            #########################################################################################################
            # img = np.copy(world.seg_camera.image)
            seg_img = np.copy(world.seg_camera.image)
            if seg_img is not None and np.size(seg_img) > 1: # publish the segmented image 
                world.ImageToOccupancyGrid(seg_img)
            #########################################################################################################

            if world.seg_camera.image is not None:
                cv_msg = world.seg_camera.bridge.cv2_to_imgmsg(world.seg_camera.image, "rgb8")
                world.pub_carla_image.publish(cv_msg)


            if world.autonomous_mode:
                controller._control.gear = 1 if world.target_velocity>=0 else -1
                world.gear_state = 'D' if world.target_velocity > 0 else 'R'
                controller._control.steer = (world.steer / 540)*1.0
                v.x = world.target_velocity*math.cos(th)
                v.y = world.target_velocity*math.sin(th)
                v.z = 0

                world.player.apply_control(controller._control)# apply steering command
                world.player.set_target_velocity(v) # cannot be exact velocity of the car at carla simulator since physics of the car such as friction affects the resulting velocity


            if vel_pub_flag and loop_counter > 20 and world.player.get_transform().location.z < 1.0:
                print("@@@@@@@@@@ Let's plan!!!!!!!!!!!!!")
                world.autonomous_mode = True
                velData = Float32MultiArray()
                velData.data = [0.0]
                world.pubVelData.publish(velData)
                vel_pub_flag = False
                startflag = Int32()
                startflag.data = 1
                world.pub_start_new_scenario.publish(startflag)
        

        

            world.tick(clock) # tick for HUD
            # world.render(display)
            pygame.display.flip()
            loop_counter += 1

            # world.tick(clock)
            world.render(display)
            pygame.display.flip()
            if rospy.is_shutdown():
                print('shutdown')
                break

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()
        
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='500 x 500',
        help='window resolution (default: 1280x720)')#416 x 416 or 640 x 360
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    
    argparser.add_argument(
        '-s',
        default=int(M_SCENARIO),
        type=int,
        help=' which scenario? ')

    argparser.add_argument(
        '-c',
        default=M_CASE,
        type=int,
        help=' which case? ')
    
    argparser.add_argument(
        '-init_pose',
        type=str,
        action='store',
        default=[],
        nargs='*',
        help="set the vehicle's initial pose"
        )
    
    argparser.add_argument(
        '-goal_pose',
        type=str,
        action='store',
        default=[],
        nargs='*',
        help="set the target pose"
        )
    
    argparser.add_argument(
        '-load',
        type=str,
        default='',
        # nargs='*',
        help="load a scenario info."
        )
    
    argparser.add_argument(
        '-log',
        type=bool,
        default=False,
        help="enabling a logging mode"
        )

    argparser.add_argument(
        '-cam_on',
        type=int,
        default=1,
        help="enabling a cam mode"
        )
    
    argparser.add_argument(
        '-avm_on',
        type=int,
        default=0,
        help="enabling avm camera"
        )
    
    argparser.add_argument(
        '-rl',
        type=int,
        default=1,
        help="enabling reinforcement training mode"
        )
    
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
