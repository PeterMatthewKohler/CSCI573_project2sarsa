#!/usr/bin/python2

# ROS Imports
import rospy
import rospkg
from std_msgs.msg import String
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState, ModelStates

# Std Imports
import sys
import time
import math
import numpy as np
import random

# Prevent .pyc file creation when saving Q_TABLE
sys.dont_write_bytecode = True 

# State Class Definition
class State:
    def __init__(self, Right = None, RightFront = None, Left = None, Front = None):
        self.Right = Right
        self.RightFront = RightFront
        self.Left = Left
        self.Front = Front

# Define Initial State
robot_state = State()

# Pose Variables to determine if robot is stuck (Used in poseCallBack)
pose_x = 0
pose_y = 0
pose_z = 0

# Messages and Publisher Initializations
msg_velocity = Pose2D()
msg_state = ModelState()
msg_state.model_name = 'triton_lidar' # Model Name

velocity_publisher = rospy.Publisher("/triton_lidar/vel_cmd", Pose2D, queue_size = 10)
model_state_publisher = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size = 10)


# Input Arguments (Determine if performing Q_Learning or using Prior Learned Q_table)
inp_args = rospy.myargv(argv=sys.argv)
if str(inp_args[1]) == "True":
    sarsa = True
else:
    sarsa = False

if sarsa:  # Perform Q_Learning Algorithm to fill up Q table
    # Initialize Empty Q table
    Q_TABLE = {}
    
    #Define States:
    RightStates = ["Right_TooClose", "Right_Close", "Right_Medium", "Right_Far", "Right_TooFar"]
    RightFrontStates = ["RightFront_Close", "RightFront_Far"]
    FrontStates = ["Front_TooClose", "Front_Close", "Front_Medium", "Front_Far"]
    LeftStates = ["Left_Close", "Left_Far"]

    # Q table initialized to all zeros
    for rightState in RightStates:
        for rightFrontState in RightFrontStates:
            for frontState in FrontStates:
                for leftState in LeftStates:
                    Q_TABLE[rightState + "||" + rightFrontState + "||" + frontState + "||" + leftState] = {"turn_left": 0, "go_forward": 0, "turn_right": 0}

else:
    from qtable_storage import Q_TABLE

# Data Collection
# Used for analysis in Deliverable 3
rewardData = []
qConvergence = []

# Determine Robot State every Laser Scan Callback
def scanCallBack(sensor_message):
    # Each direction is the minimum depth for the span of the angles associated with that direction
    Right = min(sensor_message.ranges[345:360] + sensor_message.ranges[0:15])
    RightFront = min(sensor_message.ranges[30:60])
    Front = min(sensor_message.ranges[60:120])
    Left = min(sensor_message.ranges[150:210])

    # Conditionals to determine state (Must Match State Definitions exactly!)
    # Right
    if Right < 0.4:
        robot_state.Right = "TooClose"
    elif Right >= 0.4 and Right < 0.6:
        robot_state.Right = "Close"
    elif Right >= 0.6 and Right < 0.8:
        robot_state.Right = "Medium"
    elif Right >= 0.8 and Right < 1.2:
        robot_state.Right = "Far"
    else:
        robot_state.Right = "TooFar"
    
    # RightFront
    if RightFront <= 1.2:
        robot_state.RightFront = "Close"
    else:
        robot_state.RightFront = "Far"
    
    # Front
    if Front < 0.4:
        robot_state.Front = "TooClose"
    elif Front >= 0.4 and Front < 0.6:
        robot_state.Front = "Close"
    elif Front >= 0.6 and Front < 1.2:
        robot_state.Front = "Medium"
    else:
        robot_state.Front = "Far"
    
    # Left
    if Left <= 0.4:
        robot_state.Left = "Close"
    else:
        robot_state.Left = "Far"
    
def poseCallBack(pose_message):
    # Use Global Variables
    global pose_x, pose_y, pose_z
    pose_x = pose_message.pose[1].position.x
    pose_y = pose_message.pose[1].position.y
    pose_z = pose_message.pose[1].position.z


# Function to determine reward based on current state of robot
def getReward():    
    if robot_state.Right == "TooClose" or robot_state.Right == "TooFar" or robot_state.Front == "TooClose" or robot_state.Left == "Close":
        reward = -1
    else:
        reward = 0
    return reward

# With Previous State info determine next robot action
def determineAction(prevState):
    if prevState.Right != None and prevState.RightFront != None and prevState.Front != None and prevState.Left != None:
        qIndex = "Right_" + str(prevState.Right) + "||" + "RightFront_" + str(prevState.RightFront) + "||" + "Front_" + str(prevState.Front) + "||" + "Left_" + str(prevState.Left)
        # If Q values at current state are not all the same
        if (Q_TABLE[qIndex]["turn_left"] != Q_TABLE[qIndex]["go_forward"]) and (Q_TABLE[qIndex]["go_forward"] != Q_TABLE[qIndex]["turn_right"]):
            # Return action (dictionary key) of largest Q value (dictionary value) in index
            return max(Q_TABLE[qIndex], key=Q_TABLE[qIndex].get)
        # If they are all the same, pick a random one
        else:
            actions = Q_TABLE[qIndex].keys() 
            return random.choice(actions)
    # If states are all None, robot is in some undefined state or is just being initialized(?)
    else:
        return "Current state does not exist in table"

def determineReward(prevState):
    if prevState.Right != None and prevState.RightFront != None and prevState.Front != None and prevState.Left != None:
        qIndex = "Right_" + str(prevState.Right) + "||" + "RightFront_" + str(prevState.RightFront) + "||" + "Front_" + str(prevState.Front) + "||" + "Left_" + str(prevState.Left)
        # If Q values at current state are not all the same
        if (Q_TABLE[qIndex]["turn_left"] != Q_TABLE[qIndex]["go_forward"]) and (Q_TABLE[qIndex]["go_forward"] != Q_TABLE[qIndex]["turn_right"]):
            # Return highest reward value at that state
            return max(Q_TABLE[qIndex].values())
        # If they are all the same, pick a random reward value
        else:
            reward_values = Q_TABLE[qIndex].values()
            return random.choice(reward_values)
    # If states are all None, robot is in some undefined state or is just being initialized(?)
    else:
        return "Current state does not exist in table"



# With Previous State info update Q Table using SARSA Algorithm Formula
def updateQTable(prevState, reward, firstAction, secondAction, discountFactor, learningRate):
    if prevState.Right != None and prevState.RightFront != None and prevState.Front != None and prevState.Left != None:
        prev_index = "Right_" + str(prevState.Right) + "||" + "RightFront_" + str(prevState.RightFront) + "||" + "Front_" + str(prevState.Front) + "||" + "Left_" + str(prevState.Left)
        current_index = "Right_" + str(robot_state.Right) + "||" + "RightFront_" + str(robot_state.RightFront) + "||" + "Front_" + str(robot_state.Front) + "||" + "Left_" + str(robot_state.Left)
        # Update Q Table using SARSA formula
        Q_TABLE[prev_index][firstAction] = Q_TABLE[prev_index][firstAction] + learningRate * (reward + discountFactor * Q_TABLE[current_index][secondAction] - Q_TABLE[prev_index][firstAction])


def resetRobot():
    # Reset at position (0,0,0) with orientation of (0,0,0,1)
    msg_state.pose.position.x = 0
    msg_state.pose.position.y = 0
    msg_state.pose.position.z = 0
    msg_state.pose.orientation.x = 0
    msg_state.pose.orientation.y = 0
    msg_state.pose.orientation.z = 0
    msg_state.pose.orientation.w = 0
    model_state_publisher.publish(msg_state)

def doDefinedAction(action, duration):
    # Defined Action is to be taken
    if action == "turn_left":                   # Turn Left
        msg_velocity.y = 0.3
        msg_velocity.theta = (math.pi/4)
    elif action == "go_forward":                # Go Forward
        msg_velocity.y = 0.3
        msg_velocity.theta = 0
    else:                                       # Turn Right
        msg_velocity.y = 0.3
        msg_velocity.theta = -(math.pi/4)
    
    # Execute action for defined duration(timestep)
    timeCurrent = rospy.Time.now()
    timeStop = timeCurrent + duration
    while rospy.Time.now() < timeStop:
        velocity_publisher.publish(msg_velocity)
    # Once timestep has passed, stop robot
    msg_velocity.y = 0
    msg_velocity.theta = 0
    velocity_publisher.publish(msg_velocity)

def doRandomAction(duration):
    # Random Action is to be taken, so pick random number
    action = random.randint(0,2)    # Three possible actions
    if action == 0:                 # Turn Left
        action = "turn_left"
        msg_velocity.y = 0.3
        msg_velocity.theta = (math.pi/4)
    elif action == 1:               # Go Forward
        action = "go_forward"
        msg_velocity.y = 0.3
        msg_velocity.theta = 0
    else:                           # Turn Right
        action = "turn_right"
        msg_velocity.y = 0.3
        msg_velocity.theta = -(math.pi/4)
    
    # Execute action for defined duration(timestep)
    time_current = rospy.Time.now()
    time_stop = time_current + duration
    while rospy.Time.now() < time_stop:
        velocity_publisher.publish(msg_velocity)
    # Once timestep has passed, stop robot
    msg_velocity.y = 0
    msg_velocity.theta = 0
    velocity_publisher.publish(msg_velocity)

    # Return action for logging and state definition purposes
    return action
            
def doGreedyAction(prevState, duration):
    # Use Q Table to look up action with maximum reward
    action = determineAction(prevState)
    # 3 Possibilities
    if action == "turn_left":           # Turn Left
        msg_velocity.y = 0.3
        msg_velocity.theta = (math.pi/4)
    elif action == "go_forward":        # Go Forward
        msg_velocity.y = 0.3
        msg_velocity.theta = 0
    else:                               # Turn Right
        msg_velocity.y = 0.3
        msg_velocity.theta = -(math.pi/4)
    # Only execute action for designated time
    time_current = rospy.Time.now()
    time_stop = time_current + duration
    while rospy.Time.now() < time_stop:
        velocity_publisher.publish(msg_velocity)
    # Once timestep has passed, stop robot
    msg_velocity.y = 0
    msg_velocity.theta = 0
    velocity_publisher.publish(msg_velocity)

    # Return action for logging and state definition purposes
    return action

# Main Function for Performing Training or Using Best Trained Policy
def main():
    # Initialize ROS node
    rospy.init_node("wallFollow", anonymous=True)

    # Initialization variables
    trainingDone = False                    # Logic Variable to determine when to exit loop
    discountFactor = 0.8                    # Discount Factor (alpha)
    learningRate = 0.2                      # Learning Rate (gamma)
    epsilon_0 = 0.9                         # Initial Epsilon Value
    epsilon = epsilon_0                     # epsilon variable that will be updated over course of algorithm
    d = 0.985                               # Epsilon Greedy Calculation Parameter    
    episodeNumber = 0                       # Count number of episodes                  
    timestep = 0                            # Count number of 0.5 second timesteps elapsed
    numTrappedStates = 3                    # Number of consecutive states robot needs to be in same position to be considered "trapped" and reset
    episodeReward = 0                       # Total reward accrued over individual episode
    totalAccruedReward = 0                  # Total reward accrued over all episodes
    prevState = State()                     # State class initialization for "Previous State"
    rate = rospy.Rate(25)                   # Loop Rate = 0.04 seconds (25 Hz)
    timestepDuration = rospy.Duration(0.5)  # 0.5 seconds
    timestepsThisEpisode = 0                # Number of timesteps elapsed on a particular episode
    longestEpisode = 0                      # Variable to track longest episode in (counted in timesteps)
    listActions = [ "turn_left",            # List of actions robot can take at particular state
                    "go_forward", 
                    "turn_right"]

    #declare topic subscribers
    scan_topic = '/scan'
    scan_subscriber = rospy.Subscriber(scan_topic, LaserScan, scanCallBack)
    model_topic = '/gazebo/model_states'
    model_subscriber = rospy.Subscriber(model_topic, ModelStates, poseCallBack)

    # Sleep for 5 seconds to allow for everything to load
    time.sleep(5)

    # While loop to execute the training or best wall following policy
    # Loop for each episode
    while not rospy.is_shutdown() and not trainingDone:
        terminate = False
        # Reset robot position and orientation
        resetRobot()
        # Define list of prior poses for use to determine if robot is trapped and needs to be reset
        # Row 0 is 3 previous X coordinates of robot, Row 1 is 3 previous Y coordinates of robot
        prevPose = np.zeros((2,3))
        # Data Collection for Deliverable 3
        if timestepsThisEpisode != 0:
            totalAccruedReward += float(episodeReward)/timestepsThisEpisode
            rewardData.append((episodeNumber, totalAccruedReward))
            sum = 0
            for key in Q_TABLE:
                sum += Q_TABLE[key]["turn_left"] + Q_TABLE[key]["go_forward"] + Q_TABLE[key]["turn_right"]
            qConvergence.append((episodeNumber, sum))
            print("qConvergence: ", (episodeNumber, sum))
        # Reset accrued episode reward and timesteps elapsed
        episodeReward = 0
        timestepsThisEpisode = 0

        # Choose A from S using policy derived from Q (Epsilon Greedy)
        if random.uniform(0,1) > epsilon or sarsa == False: # Pick best action from policy
            firstAction = determineAction(robot_state)
        else:                                               # Pick random action
            firstAction = random.choice(listActions)


        # Loop for each step of single episode
        while not terminate and not trainingDone:
            # If training, perform SARSA on-policy TD control
            if sarsa:
                # Timestep and State Printing for Error catching
                print("\n Timestep = ", timestep, "Episode Number = ", episodeNumber)
                print("\n Current State is: ", "Right = ", robot_state.Right, " RightFront = ", robot_state.RightFront, " Front = ", robot_state.Front, " Left = ", robot_state.Left)
                # Update Prior State
                prevState.Right = robot_state.Right
                prevState.RightFront = robot_state.RightFront
                prevState.Front = robot_state.Front
                prevState.Left = robot_state.Left
                # Update Prior Poses for trapped determination (First Row is X, Second Row is Y)
                prevPose[0][timestep % numTrappedStates] = pose_x
                prevPose[1][timestep % numTrappedStates] = pose_y
                # Update Epsilon
                epsilon = epsilon_0 * (d ** episodeNumber)
                # Take Action, observe reward and new state (S') (robot_state updated automatically via callback)
                doDefinedAction(firstAction, timestepDuration)
                # Incrememt number of timesteps this episode
                timestepsThisEpisode += 1
                # Determine reward from action
                reward = getReward()
                # Add reward to total accrued reward for this episode
                episodeReward += reward
                # Determine next action from new state (S') 
                if random.uniform(0,1) > epsilon: # Take Greedy Action
                    secondAction = determineAction(robot_state)
                else:           # Choose random next action
                    secondAction = random.choice(listActions)

                # Update Q table for this action
                priorQ = Q_TABLE[]
                updateQTable(prevState, reward, firstAction, secondAction, discountFactor, learningRate)
                # Action <- Action'
                firstAction = secondAction
                # Update Longest Episode Counter
                if timestepsThisEpisode > longestEpisode:
                    longestEpisode = timestepsThisEpisode
                # Epsilon and Longest Episode Printing for Error catching
                print("\n Epsilon = ", epsilon)
                print("\n Longest Episode (seconds) = ", longestEpisode*0.5)
                # Determine if robot is trapped, if so terminate the episode
                if (max(prevPose[0]) - min(prevPose[0])) < 0.05 and (max(prevPose[1]) - min(prevPose[1])) < 0.05 or pose_z > 0.05:
                    terminate = True
                    print("\n Trapped. Resetting robot position...")
                # Declare training done after 20000 timesteps total or 10000 timesteps in a single episode
                elif timestep > 20000 or timestepsThisEpisode > 10000: 
                    trainingDone = True
                    # Write Q_Learning policy and reward data to storage files
                    f = open("/home/hcr-student/Stingray-Simulation/catkin_ws/src/project2sarsa/src/qtable_storage.py", "w")
                    f.write("Q_TABLE = " + str(Q_TABLE))
                    f.close()
                    f = open("/home/hcr-student/Stingray-Simulation/catkin_ws/src/project2sarsa/src/reward_storage.py", "w")
                    f.write("Reward Data from Training = " + str(rewardData))
                    f.close()
                    print("\n Q_Learning Complete, Q Table and reward data saved.")
                    print("\n FOR ERROR TESTING: " + str(rewardData))

                # Update number of timesteps by one
                timestep += 1
            # If not training, use Q_TABLE policy from qtable_storage.py to choose optimal action
            else: 
                doGreedyAction(robot_state, timestepDuration)
            
        # Increment episode number
        episodeNumber += 1
        # Sleep
        rate.sleep()
        

if __name__ == '__main__':    
    try:
        main()
    
    except rospy.ROSInterruptException:
        rospy.loginfo("node terminated.")