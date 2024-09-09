#!/usr/bin/env python3

"""
Compute state space kinematic matrices for xArm7 robot arm (5 links, 7 joints)
"""

import numpy as np
from math import sin,  cos


"""    
def R_x(q):
        return Matrix([[1, 0, 0,0],\
                      [0, cos(q), -sin(q),0],\
                      [0, sin(q), cos(q),0],\
                      [0, 0, 0, 1]])
def R_z(q):
        return Matrix([[cos(q), -sin(q), 0,0],\
                      [sin(q), cos(q), 0,0],\
                      [0, 0, 1,0],\
                      [0, 0, 0, 1]])
def T_x(d):
       return Matrix([[1, 0, 0, d],\
                      [0, 1, 0, 0],\
                      [0, 0, 1, 0],\
                      [0, 0, 0, 1]])
def T_z(d):
        return Matrix([[1, 0, 0, 0],\
                      [0, 1, 0, 0],\
                      [0, 0, 1, d],\
                      [0, 0, 0, 1]])
"""

class xArm7_kinematics():
    
    
    def __init__(self):

        self.l1 = 0.267
        self.l2 = 0.293
        self.l3 = 0.0525
        self.l4 = 0.3512
        self.l5 = 0.1232

        self.theta1 = 0.2225 #(rad) (=12.75deg)
        self.theta2 = 0.6646 #(rad) (=38.08deg)
        
        self.d1 = self.l4 * sin(self.theta1)
        self.d2 = self.l4 * cos(self.theta1)
        self.d3 = self.l5 * sin(self.theta2)
        self.d4 = self.l5 * cos(self.theta2)

        pass

    def compute_jacobian(self, r_joints_array):
    	
        #Finding the r0_E, r0_i vectors
        R0_arrays = []
        R0_arrays.append(self.tf_A01(r_joints_array)[:3,3] )
        R0_arrays.append(self.tf_A02(r_joints_array)[:3,3] )
        R0_arrays.append(self.tf_A03(r_joints_array)[:3,3] )
        R0_arrays.append(self.tf_A04(r_joints_array)[:3,3] )
        R0_arrays.append(self.tf_A05(r_joints_array)[:3,3] )
        R0_arrays.append(self.tf_A06(r_joints_array)[:3,3] )
        R0_arrays.append(self.tf_A07(r_joints_array)[:3,3] )
        
        #Finding b0,b1,b2.....b6
        # J_1* = b0 x R_07
        b0 = [0, 0, 1]
        J_q1= np.reshape(np.cross( b0 , R0_arrays[6].T ) , 3 )
        #R0_array[6] is 3x1 array and is transposed to
        # 1x3 for  cross product 
        J_11 = J_q1[0]
        J_21 = J_q1[1]
        J_31 = J_q1[2]
        
        #Rest of the J_qi i=2,3,4...7 
        #J_q1= [ J_qix = J_1i
        #        J_qiy = J_2i
        #        J_qiz = J_3i  ]
        
        b1 = self.tf_A01(r_joints_array)[:3,2]
        J_q2 = np.reshape( np.cross( b1.T , (R0_arrays[6]-R0_arrays[0]).T ) , 3 )
        J_12 = J_q2[0]
        J_22 = J_q2[1]
        J_32 = J_q2[2]
        
        b2 = self.tf_A02(r_joints_array)[:3,2]
        J_q3 = np.reshape(np.cross( b2.T , (R0_arrays[6]-R0_arrays[1]).T ) , 3 )
        J_13 = J_q3[0]
        J_23 = J_q3[1]
        J_33 = J_q3[2]
        
        b3 = self.tf_A03(r_joints_array)[:3,2]
        J_q4 = np.reshape(np.cross( b3.T , (R0_arrays[6]-R0_arrays[2]).T ) , 3 )
        J_14 = J_q4[0]
        J_24 = J_q4[1]
        J_34 = J_q4[2]
        
        b4 = self.tf_A04(r_joints_array)[:3,2]
        J_q5 = np.reshape(np.cross( b4.T , (R0_arrays[6]-R0_arrays[3]).T ) , 3 )
        J_15 = J_q5[0]
        J_25 = J_q5[1]
        J_35 = J_q5[2]
        
        b5 = self.tf_A05(r_joints_array)[:3,2]
        J_q6 = np.reshape(np.cross( b5.T , (R0_arrays[6]-R0_arrays[4]).T ) , 3 )
        J_16 = J_q6[0]
        J_26 = J_q6[1]
        J_36 = J_q6[2]
        
        b6 = self.tf_A06(r_joints_array)[:3,2]
        J_q7 = np.reshape(np.cross( b6.T , (R0_arrays[6]-R0_arrays[5]).T ) , 3 )
        J_17 = J_q7[0]
        J_27 = J_q7[1]
        J_37 = J_q7[2]
        
        J = np.matrix([ [ J_11 , J_12 , J_13 , J_14 , J_15 , J_16 , J_17 ],\
                        [ J_21 , J_22 , J_23 , J_24 , J_25 , J_26 , J_27 ],\
                        [ J_31 , J_32 , J_33 , J_34 , J_35 , J_36 , J_37 ]])
        
        return J

    def tf_A01(self, r_joints_array):
        q1 = r_joints_array[0]
        
        #tf = np.matrix( R_z(q1) @ T_z(self.l1) )
        
        tf = np.matrix([[cos(q1) , -sin(q1)  , 0 , 0],\
                        [sin(q1) , cos(q1) , 0 , 0],\
                        [0 , 0 , 1 , self.l1],\
                        [0 , 0 , 0 , 1]])
        return tf

    def tf_A02(self, r_joints_array):
        q2 = r_joints_array[1]
        
        #tf = np.matrix( self.tf_A01(r_joints_array) @ R_x(-pi/2) @ R_z(q2))
        
        tf_A12 = np.matrix([[cos(q2) , -sin(q2)  , 0 , 0],\
                            [0 , 0 , 1 , 0],\
                            [-sin(q2)  , -cos(q2)  , 0 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A01(r_joints_array), tf_A12 )
        return tf

    def tf_A03(self, r_joints_array):
        q3 = r_joints_array[2]

        #tf = np.matrix( self.tf_A02(r_joints_array) @ R_x(pi/2) @ R_z(q3) @ T_z(self.l2))
        tf_A23 = np.matrix([[cos(q3) , -sin(q3)  , 0 , 0],\
                            [0 , 0 , -1 , -self.l2],\
                            [sin(q3)  , cos(q3)  , 0 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A02(r_joints_array), tf_A23 )
        return tf

    def tf_A04(self, r_joints_array):
        q4 = r_joints_array[3]

        #tf = np.matrix( self.tf_A03(r_joints_array) @ R_x(pi/2) @ T_x(self.l3) @ R_z(q4) )
        tf_A34 = np.matrix([[cos(q4) , -sin(q4)  , 0 , self.l3],\
                            [0 , 0 , -1 , 0],\
                            [sin(q4)  , cos(q4)  , 0 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A03(r_joints_array), tf_A34 )
        return tf

    def tf_A05(self, r_joints_array):
        q5 = r_joints_array[4]
        
        #tf = self.tf_A04(r_joints_array) @ R_x(pi/2) @ T_x(self.d1)@ R_z(q5) @ T_z(self.d2)
        tf_A45 = np.matrix([[cos(q5) , -sin(q5)  , 0 , self.d1],\
                            [0 , 0 , -1 , -self.d2],\
                            [sin(q5)  , cos(q5)  , 0 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A04(r_joints_array), tf_A45 )
        return tf

    def tf_A06(self, r_joints_array):
        q6 = r_joints_array[5]
        
        #tf = np.matrix( self.tf_A05(r_joints_array) @ R_x(pi/2) @ R_z(q6) )
        
        tf_A56 = np.matrix([[cos(q6) , -sin(q6)  , 0 , 0],\
                            [0 , 0 , -1 , 0],\
                            [sin(q6)  , cos(q6)  , 0 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A05(r_joints_array), tf_A56 )
        return tf

    def tf_A07(self, r_joints_array):
        q7 = r_joints_array[6]
        
        #tf = np.matrix( self.tf_A06(r_joints_array) @ R_x(-pi/2) @ T_x(self.d3) @ R_z(q7) @ T_z(self.d4) )
        
        
        tf_A67 = np.matrix([[cos(q7) , -sin(q7)  , 0 , self.d3],\
                            [0 , 0 , 1 , self.d4],\
                            [-sin(q7)  , -cos(q7)  , 0 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A06(r_joints_array), tf_A67 )
         
        return tf
        
        
        
        
