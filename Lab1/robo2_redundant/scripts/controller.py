#!/usr/bin/env python3

"""
Start ROS node to publish angles for the position control of the xArm7.
"""

# Ros handlers services and messages
import rospy, roslib
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
#Math imports
from math import sin, cos, atan2, pi, sqrt
from numpy.linalg import inv, det, norm, pinv
import numpy as np
#import matlplotlib.pyplot as plt

# Arm parameters
# xArm7 kinematics class
from kinematics import xArm7_kinematics

# from tf.transformations import quaternion_matrix
# matrix = quaternion_matrix([1, 0, 0, 0])

class xArm7_controller():
    """Class to compute and publish joints positions"""
    def __init__(self,rate):

        # Init xArm7 kinematics handler
        self.kinematics = xArm7_kinematics()

        # joints' angular positions
        self.joint_angpos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # joints' angular velocities
        self.joint_angvel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # joints' states
        self.joint_states = JointState()
        # joints' transformation matrix wrt the robot's base frame
        self.A01 = self.kinematics.tf_A01(self.joint_angpos)
        self.A02 = self.kinematics.tf_A02(self.joint_angpos)
        self.A03 = self.kinematics.tf_A03(self.joint_angpos)
        self.A04 = self.kinematics.tf_A04(self.joint_angpos)
        self.A05 = self.kinematics.tf_A05(self.joint_angpos)
        self.A06 = self.kinematics.tf_A06(self.joint_angpos)
        self.A07 = self.kinematics.tf_A07(self.joint_angpos)
        # gazebo model's states
        self.model_states = ModelStates()

        # ROS SETUP
        # initialize subscribers for reading encoders and publishers for performing position control in the joint-space
        # Robot
        self.joint_states_sub = rospy.Subscriber('/xarm/joint_states', JointState, self.joint_states_callback, queue_size=1)
        self.joint1_pos_pub = rospy.Publisher('/xarm/joint1_position_controller/command', Float64, queue_size=1)
        self.joint2_pos_pub = rospy.Publisher('/xarm/joint2_position_controller/command', Float64, queue_size=1)
        self.joint3_pos_pub = rospy.Publisher('/xarm/joint3_position_controller/command', Float64, queue_size=1)
        self.joint4_pos_pub = rospy.Publisher('/xarm/joint4_position_controller/command', Float64, queue_size=1)
        self.joint5_pos_pub = rospy.Publisher('/xarm/joint5_position_controller/command', Float64, queue_size=1)
        self.joint6_pos_pub = rospy.Publisher('/xarm/joint6_position_controller/command', Float64, queue_size=1)
        self.joint7_pos_pub = rospy.Publisher('/xarm/joint7_position_controller/command', Float64, queue_size=1)
        # Obstacles
        self.model_states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback, queue_size=1)

        #Publishing rate
        self.period = 1.0/rate
        self.pub_rate = rospy.Rate(rate)

        self.publish()

    #SENSING CALLBACKS
    def joint_states_callback(self, msg):
        # ROS callback to get the joint_states

        self.joint_states = msg
        # (e.g. the angular position of joint 1 is stored in :: self.joint_states.position[0])

    def model_states_callback(self, msg):
        # ROS callback to get the gazebo's model_states

        self.model_states = msg
        # (e.g. #1 the position in y-axis of GREEN obstacle's center is stored in :: self.model_states.pose[1].position.y)
        # (e.g. #2 the position in y-axis of RED obstacle's center is stored in :: self.model_states.pose[2].position.y)
    
    def publish(self):

        # set configuration
        # total pitch: j2-j4+j6+pi (upwards: 0rad)
        j2 = 0.7 ; j4 = np.pi/2
        j6 = - (j2-j4)
        self.joint_angpos = [0.0, j2, 0.0, j4, 0.0, j6, 0.0]
        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()
        self.joint4_pos_pub.publish(self.joint_angpos[3])
        tmp_rate.sleep()
        self.joint2_pos_pub.publish(self.joint_angpos[1])
        self.joint6_pos_pub.publish(self.joint_angpos[5])
        tmp_rate.sleep()
        print("The system is ready to execute your algorithm...")

        rostime_now = rospy.get_rostime()
        time_now = rostime_now.to_nsec()
        
        #INITIALIZATIONS
        T = 2 #(seconds)-Period
        Dt = 0.02
        t = 0  
        y0 = 0    #initial y-axis end point
        yf = 0.19    #initial final y-axis position
        poly_dot = 0.0
        #Flag variables to decide in which direction the xarm7 should go when it's on final position(yf)
        right = 0  
        left =  1
        #Initial distance from obstacles
        d_previous=0.02
                
        while not rospy.is_shutdown():

    # Compute each transformation matrix wrt the base frame from joints' angular positions
            
            self.A01 = self.kinematics.tf_A01(self.joint_angpos)
            self.A02 = self.kinematics.tf_A02(self.joint_angpos)
            self.A03 = self.kinematics.tf_A03(self.joint_angpos)
            self.A04 = self.kinematics.tf_A04(self.joint_angpos)
            self.A05 = self.kinematics.tf_A05(self.joint_angpos)
            self.A06 = self.kinematics.tf_A06(self.joint_angpos)
            self.A07 = self.kinematics.tf_A07(self.joint_angpos)

            # Compute jacobian matrix
            J = self.kinematics.compute_jacobian(self.joint_angpos)
            # pseudoinverse jacobian
            pinvJ = pinv(J)
            
            
            #MAIN CODE ----CALCULATION OF ANGLE VELOCITIES OF JOINTS
            # q_dot = q(1)_dot + q(2)_dot
            # opou q(1) q(2) oi genikeumenes taxitites twn arthrwsewn gia kathe task
            
            
            #---TASK 1--- Periodiki kinisi telikou stoixeiou drasis
            #______________________________________________________________
            # 2 cases:
            # A) From initial position to  pB positio
            # B) Move between pA<-->pB
            #
            # For each movement i have a  polynomial 5th order with these boundaries:
            # f(0)=y0 _____f(T)=yf               0<=t<=T    
            # f'(0)=0______f'(T)=0
            # f''(0)=0_____f''(T)=0
            
            x_eef =  self.A07[0,3]
            y_eef =  self.A07[1,3]
            z_eef =  self.A07[2,3]
   
            xf = 0.617
            zf = 0.199
            
            # move to the left if end point exceeds y=0.2 
            if (y_eef+0.063>= 0.17) & (left==1) & (poly_dot==0.0 or poly_dot==-0.0):
                t = 0 
                y0 = 0.19
                yf = -0.1
                T= 3
                left = 0
                right = 1
                #print("Time:",t)
                #print("Poly:",poly)
                #print("Poly_der:",poly_dot)
                #print("Poly_der2:",poly_dot2)
                print("The position of the end point y-axis is : ", y_eef)
                #print("The position of the end point x-axis is : ", x_eef)
                #print("The position of the end point z-axis is : ", z_eef)
                print("______________________________________________________")
            
            # move to the right if end point exceeds y=-0.2
            if (y_eef-0.063<= -0.05) & (right==1) & (poly_dot==0.0 or poly_dot==-0.0):
                t = 0
                y0= -0.1
                yf= 0.19
                T= 3
                right = 0
                left = 1
                #print("Time:",t)
                #print("Poly:",poly)
                #print("Poly_der:",poly_dot)
                #print("Poly_der2:",poly_dot2)
                print("The position of the end point y-axis is : ", y_eef)
                #print("The position of the end point x-axis is : ", x_eef)
                #print("The position of the end point z-axis is : ", z_eef)
                print("______________________________________________________")
            
                        
            # Calculate polynomial(coefficients) for y position 
            a0 = y0
            a1 = 0
            a2 = 0
            a3 = -(10/(T**3))*(y0-yf)
            a4 = (15/(T **4))*(y0-yf)
            a5 = -(6/(T **5))*(y0-yf)
            poly =  round(a5*t **5 + a4*t **4 + a3*t **3 + a2*t **2 + a1*t + a0,4)
            #derivative 
            poly_dot = round(5*a5*t **4 + 4*a4*t **3 + 3*a3*t **2 + 2*a2*t  + a1,4)
            poly_dot2 = round(20*a5*t **3 + 12*a4*t **2 + 6*a3*t +2*a2,4)
            
            
            #Errors
            #print("Error in x - axis is:", x_eef-xf)
            #print("Error in y - axis is:", y_eef-poly)
            #print("Error in z - axis is:", z_eef-zf)
            #print("______________________________________________________")
            
            #p1d = np.matrix([xf,\
            #                 poly,\
            #                 zf]).T
            
            p1d_vel = np.matrix([0.0,\
                                poly_dot,\
                                 0.0]).T
            
            joints_vel_task1 = pinvJ@ p1d_vel #+ K1*(p1d - self.A07[:3,3]))
            t = t + Dt# current time
            
            
            #print(joints_vel_task1)
            
            #---TASK 2--- Apofigi 2 empodiwn mesw sinartisis kritiriou 
            #___________________________________________________________
            K2=0.12
            kc=0.15
            #safe point distance 
            d0 = 0.02
            
            #Red obstacle position 
            A = np.matrix([self.model_states.pose[2].position.x , self.model_states.pose[2].position.y, self.model_states.pose[2].position.z ])
            #Green obstacle position
            B = np.matrix([self.model_states.pose[1].position.x , self.model_states.pose[1].position.y, self.model_states.pose[1].position.z])
            
            ## We will divide each link 1,2,3,4,5 to a certain number of points in order to find the minimum distance between robot and obstacles
            num_points =150 #numberof points division for each link 
            pos_joints = np.array([self.A01[:3,3].T, self.A02[:3,3].T, self.A03[:3,3].T, self.A04[:3,3].T, self.A05[:3,3].T, self.A06[:3,3].T, self.A07[:3,3].T])
            pos_joints = np.reshape (pos_joints , (7,3) )
            
            d_from_obs = np.inf
            #Calculating minimun distance between obstacles and xarm7
          
            for i in range(1,7):
              p1 = pos_joints[i-1]
              p2 = pos_joints[i]
              direction_vector = p2-p1
              points_on_link = np.array([p1 + t*direction_vector  for t in np.linspace(0,1,num_points)])
              #for the fist halfPeriod avoiding red obstacle  y>0
              if y_eef>= 0 :
                  min_value = min( [ norm(A - joints ,ord=2)-0.113 for joints in points_on_link ])
                  if  min_value <= d_from_obs :
                    min_index = i 
                    d_from_obs = min_value
               #for the second HalfPeriod avoiding green obstacle y<0
              if y_eef<0 :
                min_value = min( [ norm(B - joints,ord=2)-0.1445 for joints in points_on_link ])
                if min_value <= np.abs(d_from_obs) :
                  d_from_obs = -min_value
                  min_index = i
               
            lst =  [float(x) for x in self.joint_angpos]
            lst2 = [float(y) for y in self.joint_states.position]
            
            #if the distace from obstacle is smaller than the safedistance find the new desired joints angle velocities
            if np.abs(d_from_obs) <= d0 :
               #print("Distance from obstacle :", d_from_obs,"with index",min_index)
               obj_function_obstacles_grad = np. matrix([(np.abs(d_from_obs) - np.abs(d_previous))/(lst2[0]-lst[0]),\
                                                       (np.abs(d_from_obs) - np.abs(d_previous))/(lst2[1]-lst[1]),\
                                                       (np.abs(d_from_obs) - np.abs(d_previous))/(lst2[2]-lst[2]),\
                                                       (np.abs(d_from_obs) - np.abs(d_previous))/(lst2[3]-lst[3]),\
                                                       (np.abs(d_from_obs) - np.abs(d_previous))/(lst2[4]-lst[4]),\
                                                       (np.abs(d_from_obs) - np.abs(d_previous))/(lst2[5]-lst[5]),\
                                                       (np.abs(d_from_obs) - np.abs(d_previous))/(lst2[6]-lst[6])]).T 
               for i in range(min_index,6):
                   obj_function_obstacles_grad[i+1] = 0.0   
                              
               d_previous = d_from_obs
               if y_eef<0 :
                   q_reference_dot = -kc*(d_from_obs - d0) * obj_function_obstacles_grad
                   for i in range(0,7):
                      if q_reference_dot[i]<-pi :
                         q_reference_dot[i]+=2*pi
                   #print(q_reference_dot)
               else:
                   q_reference_dot = kc*(d_from_obs - d0) * obj_function_obstacles_grad
                   for i in range(0,7):
                      if q_reference_dot[i]> pi :
                         q_reference_dot[i]-=2*pi
                   #print(q_reference_dot)
            else:
               q_reference_dot = np.zeros((7,1))
  	    
  
            joints_vel_task2 = K2*(np.eye(7) - pinvJ @ J)@ q_reference_dot
            
            
            #print(A, B)
            
            
          
            
            #NEW ANGLE VELOCITIES OF THE JOINTS
            
            self.joint_angvel[0] = float(joints_vel_task1[0]+ joints_vel_task2[0])
            self.joint_angvel[1] = float(joints_vel_task1[1]+ joints_vel_task2[1])
            self.joint_angvel[2] = float(joints_vel_task1[2]+ joints_vel_task2[2])
            self.joint_angvel[3] = float(joints_vel_task1[3]+ joints_vel_task2[3])
            self.joint_angvel[4] = float(joints_vel_task1[4]+ joints_vel_task2[4])
            self.joint_angvel[5] = float(joints_vel_task1[5]+ joints_vel_task2[5])
            self.joint_angvel[6] = float(joints_vel_task1[6]+ joints_vel_task2[6])
            
            
            
            #Convertion to angular position after integrating the angular speed in time
            #Calculate time interval
            time_prev = time_now
            rostime_now = rospy.get_rostime()
            time_now = rostime_now.to_nsec()
            dt = (time_now - time_prev)/1e9
            
            # Integration
            self.joint_angpos = np.add( self.joint_angpos, [index * dt for index in self.joint_angvel] )
            
            
            # Publish the new joint's angular positions
            self.joint1_pos_pub.publish(self.joint_angpos[0])
            self.joint2_pos_pub.publish(self.joint_angpos[1])
            self.joint3_pos_pub.publish(self.joint_angpos[2])
            self.joint4_pos_pub.publish(self.joint_angpos[3])
            self.joint5_pos_pub.publish(self.joint_angpos[4])
            self.joint6_pos_pub.publish(self.joint_angpos[5])
            self.joint7_pos_pub.publish(self.joint_angpos[6])
            
            self.pub_rate.sleep()

    def turn_off(self):
        pass

def controller_py():
    # Starts a new node
    rospy.init_node('controller_node', anonymous=True)
    # Reading parameters set in launch file
    rate = rospy.get_param("/rate")

    controller = xArm7_controller(rate)
    rospy.on_shutdown(controller.turn_off)
    rospy.spin()

if __name__ == '__main__':
    try:
        controller_py()
    except rospy.ROSInterruptException:
        pass
