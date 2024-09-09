#!/usr/bin/env python3

"""
Start ROS node to publish linear and angular velocities to mymobibot in order to perform wall following.
"""

# Ros handlers services and messages
import rospy, roslib
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Range
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
#Math imports
from math import sin, cos, atan2, pi, sqrt
from numpy.linalg import inv, det, norm, pinv
import numpy as np
import time as t

# from tf.transformations import euler_from_quaternion
# from tf.transformations import quaternion_matrix
# matrix = quaternion_matrix([1, 0, 0, 0])

def quaternion_to_euler(w, x, y, z):
    """Converts quaternions with components w, x, y, z into a tuple (roll, pitch, yaw)"""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class mymobibot_follower():
    """Class to compute and publish joints positions"""
    def __init__(self,rate):

        # linear and angular velocity
        self.velocity = Twist()
        # joints' states
        self.joint_states = JointState()
        # Sensors
        self.imu = Imu()
        self.imu_yaw = 0.0 # (-pi, pi]
        self.sonar_F = Range()
        self.sonar_FL = Range()
        self.sonar_FR = Range()
        self.sonar_L = Range()
        self.sonar_R = Range()

        # ROS SETUP
        # initialize subscribers for reading encoders and publishers for performing position control in the joint-space
        # Robot
        self.velocity_pub = rospy.Publisher('/mymobibot/cmd_vel', Twist, queue_size=1)
        self.joint_states_sub = rospy.Subscriber('/mymobibot/joint_states', JointState, self.joint_states_callback, queue_size=1)
        # Sensors
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1)
        self.sonar_front_sub = rospy.Subscriber('/sensor/sonar_F', Range, self.sonar_front_callback, queue_size=1)
        self.sonar_frontleft_sub = rospy.Subscriber('/sensor/sonar_FL', Range, self.sonar_frontleft_callback, queue_size=1)
        self.sonar_frontright_sub = rospy.Subscriber('/sensor/sonar_FR', Range, self.sonar_frontright_callback, queue_size=1)
        self.sonar_left_sub = rospy.Subscriber('/sensor/sonar_L', Range, self.sonar_left_callback, queue_size=1)
        self.sonar_right_sub = rospy.Subscriber('/sensor/sonar_R', Range, self.sonar_right_callback, queue_size=1)

        #Publishing rate
        self.period = 1.0/rate
        self.pub_rate = rospy.Rate(rate)

        self.publish()

    #SENSING CALLBACKS
    def joint_states_callback(self, msg):
        # ROS callback to get the joint_states

        self.joint_states = msg
        # (e.g. the angular position of the left wheel is stored in :: self.joint_states.position[0])
        # (e.g. the angular velocity of the right wheel is stored in :: self.joint_states.velocity[1])

    def imu_callback(self, msg):
        # ROS callback to get the /imu

        self.imu = msg
        # (e.g. the orientation of the robot wrt the global frome is stored in :: self.imu.orientation)
        # (e.g. the angular velocity of the robot wrt its frome is stored in :: self.imu.angular_velocity)
        # (e.g. the linear acceleration of the robot wrt its frome is stored in :: self.imu.linear_acceleration)

        #quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        #(roll, pitch, self.imu_yaw) = euler_from_quaternion(quaternion)
        (roll, pitch, self.imu_yaw) = quaternion_to_euler(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)

    def sonar_front_callback(self, msg):
        # ROS callback to get the /sensor/sonar_F

        self.sonar_F = msg
        # (e.g. the distance from sonar_front to an obstacle is stored in :: self.sonar_F.range)

    def sonar_frontleft_callback(self, msg):
        # ROS callback to get the /sensor/sonar_FL

        self.sonar_FL = msg
        # (e.g. the distance from sonar_frontleft to an obstacle is stored in :: self.sonar_FL.range)

    def sonar_frontright_callback(self, msg):
        # ROS callback to get the /sensor/sonar_FR

        self.sonar_FR = msg
        # (e.g. the distance from sonar_frontright to an obstacle is stored in :: self.sonar_FR.range)

    def sonar_left_callback(self, msg):
        # ROS callback to get the /sensor/sonar_L

        self.sonar_L = msg
        # (e.g. the distance from sonar_left to an obstacle is stored in :: self.sonar_L.range)

    def sonar_right_callback(self, msg):
        # ROS callback to get the /sensor/sonar_R

        self.sonar_R = msg
        # (e.g. the distance from sonar_right to an obstacle is stored in :: self.sonar_R.range)

    def publish(self):

        # set configuration
        self.velocity.linear.x = 0.0
        self.velocity.angular.z = 0.0
        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()
        print("The system is ready to execute your algorithm...")

        rostime_now = rospy.get_rostime()
        time_now = rostime_now.to_nsec()

        
        #INITIALIAZTIONS 
        safe_distance = 0.3  #safe distance between robot and walls 
        max_linvel = 0.3     #maximum linear speed in x-axis
        max_angvel = 0.5     #maximum angular speed in z-axis
        linerror_previous = 0.0          #calculating error between desired and real distance from walls 
        angvel_desired = 0.0
        angerror_prev = 0.0
        #for PD controller 
        Kp = 1.3
        Kd = 0.5
        Kp2 = 1.2
        Kd2 = 0.4
        correction_flag = False  #flag that turns true if i have to correct my angle to avoid wall
        #its initial value is False because i start in the center
        
        while not rospy.is_shutdown():
            
            #variable for sonars
            sonar_front = self.sonar_F.range
            sonar_frontleft = self.sonar_FL.range
            sonar_frontright = self.sonar_FR.range
            sonar_left = self.sonar_L.range
            sonar_right = self.sonar_R.range
            
            # X = X1 + X2 (X1=6 ,X2=2) ==> X=8 
            #  INITIAL ANGLE POSITION angle=mod(8, pi)
            #  8 is even number so we want to turn clockwise
            # for turning clockwise we check for obstacle with front, frontleft and left sonars
            
            """
            Cases:
             1) -No obstacle nearby, distance is greater than the safe distance - we have linear velocity no anglular 
             2) - Robot is within less than 0.3 meters from the walls and need to turn around clockwise in order to correct its position
            """
            
            #PHASE 1 
            #____________________________________
            #calculation of the desired linear velocity of the robot to 
            if (sonar_front+0.2)>=0.3 and (sonar_frontright+0.2)>= 0.3 and (sonar_right >= 0.295) and (correction_flag == False):
               #Calculate error between front sensor and desired value    
               error = min(sonar_front - safe_distance, sonar_frontleft-safe_distance)
               # Calculate the derivative of the error
               error_derivative = error - linerror_previous
               
               # Calculate control signal using PD controller
               # Use this signal as the desired linear velocity of the robot
               control_signal = min( max_linvel , Kp * error + Kd* error_derivative )
               control_signal = round(control_signal ,4)
               
               linerror_previous = error
               
               vel_desired = control_signal
               #print("Linear velocity :",vel_desired, "with error", error)
               if ( error <=0.005 and vel_desired<=0.01): 
                  vel_desired = 0.0
                  linerror_previous = 0.0 
                  correction_flag= True
                  print("!!!!!Avoiding the wall!!!!!!!")
                  print("_______________________________")
                  pass
            
            #PHASE 2 
            #________________________________________________
            if (correction_flag == True and self.velocity.linear.x==0.0):
               
               #if the edge is very sharp and robot is surroundedd by walls 
               if(sonar_front<=safe_distance+0.05 and sonar_frontright<=safe_distance+0.05):
                   control_signal = max_angvel
                   error2 = sonar_left - (sonar_frontleft*cos(pi/4)-0.05)  
               #if robot is just too close to a wall 
               else:
                 #Calculate the error between left sensor and front_left 
                 #if error2 =0.0 that means robot is absolutely parallel with the wall
                 error2 = sonar_left - (sonar_frontleft*cos(pi/4)-0.05)    
                 # Calculate the derivative of the error
                 error_derivative = error2 - angerror_prev
                 # Calculate the control signal using PD controller
                 # Use this signal as the desired angular velocity of the robot
                 control_signal = min( max_angvel , Kp2*error2 + Kd2*error_derivative )
                 control_signal = round(control_signal ,4)
               
                 #Change previous angle velocity error
                 angerror_prev = error2
               
               angvel_desired = control_signal
               
               #print("Angular velocity :",angvel_desired, "with error", error2)
               if(angvel_desired<=0.008 and error2<=0.009):
                  angvel_desired = 0.0
                  angerror_prev = 0.0
                  correction_flag = False
               
               
            #control equations 
            if(vel_desired == max_linvel):
               print("KATCHAAAOW")
            self.velocity.angular.z = angvel_desired     
            self.velocity.linear.x = vel_desired

            
            
            # Calculate time interval (in case is needed)
            time_prev = time_now
            rostime_now = rospy.get_rostime()
            time_now = rostime_now.to_nsec()
            dt = (time_now - time_prev)/1e9

            # Publish the new joint's angular positions
            self.velocity_pub.publish(self.velocity)

            self.pub_rate.sleep()

    def turn_off(self):
        pass

def follower_py():
    # Starts a new node
    rospy.init_node('follower_node', anonymous=True)
    # Reading parameters set in launch file
    rate = rospy.get_param("/rate")

    follower = mymobibot_follower(rate)
    rospy.on_shutdown(follower.turn_off)
    rospy.spin()

if __name__ == '__main__':
    try:
        follower_py()
    except rospy.ROSInterruptException:
        pass
