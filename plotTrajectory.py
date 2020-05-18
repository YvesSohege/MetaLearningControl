import gym
from gym import spaces
import quadcopter,controller
from stable_baselines.common.env_checker import check_env
import numpy as np
import gui
import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
import sys

BLENDED_CONTROLLER_PARAMETERS = {'Motor_limits': [0, 9000],
                         'Tilt_limits': [-10, 10],
                         'Yaw_Control_Limits': [-900, 900],
                         'Z_XY_offset': 500,
                         'Linear_PID': {'P': [300, 300, 7000], 'I': [0.04, 0.04, 4.5], 'D': [450, 450, 5000]},
                         'Linear_To_Angular_Scaler': [1, 1, 0],
                         'Yaw_Rate_Scaler': 0.18,
                         'Angular_PID': {'P': [24000, 24000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]},
                         'Angular_PID2': {'P': [4000, 4000, 1500], 'I': [0, 0, 1.2], 'D': [1500, 1500, 0]},
                         }




controller1 = []

controller2 = []
n = 10
steps1 = []
steps2 = []
total_steps = [ ]
starttime = 1800
trajectories = []
fault_mag = 0.3
for i in range(n):
    quad_id = i

    QUADCOPTER = {
        str(quad_id): {'position': [0, 0, 0], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1, 'prop_size': [10, 4.5],
                       'weight': 1.2}}

    # Make objects for quadcopter

    quad = quadcopter.Quadcopter(QUADCOPTER)
    Quads = {str(quad_id): QUADCOPTER[str(quad_id)]}
    #gui_object = gui.GUI(quads=Quads)


    # create blended controller and link it to quadcopter object
    ctrl = controller.Blended_PID_Controller(quad.get_state, quad.get_time,
                                             quad.set_motor_speeds, quad.get_motor_speeds,
                                             quad.stepQuad, quad.set_motor_faults,
                                             params=BLENDED_CONTROLLER_PARAMETERS, quad_identifier=str(quad_id))

   # gui_object = gui.GUI(quads=Quads)
    steps = 7
   # t = np.linspace(0, , 100)
   # x_path = np.linspace(-5, 5, steps)
   # y_path = np.linspace(-5, 5, steps)
   # z_path = np.linspace(1, 10, steps)
    x_path = [0,5,0,-5,0,5,0]
    y_path = [0,0,5,0,-5,0,0]
    z_path = [5,5,5,5,5,5,5]

    yaws = np.zeros(steps)
    goals = []
    for i in range(steps):
        goals.append([x_path[i], y_path[i], z_path[i]])

    #print(str(goals))
    #goal = [goalx, goaly, goalz]
   #goal = [-10, -10, 5]
    current = 0
    ctrl.update_target(goals[current])
    faults = [0, 0, 0, 0]
   # fault_mag = np.random.uniform(0, 0.3)
    #fault_mag += 0.003
    #rotor = np.random.randint(0, 4)
    rotor = 1

    faults[rotor] = fault_mag
    ctrl.setMotorFault(faults)


    #starttime = np.random.randint(1500, 2000)
   # endtime = np.random.randint(2500, 3500)
    endtime = 31000

    ctrl.setFaultTime(starttime, endtime)

    print("fault : " + str(rotor) + " " + str(fault_mag) + " s: "+str(starttime) +" to "+ str(endtime))
    mu = 0.5
    sigma = 0.1
    obs = ctrl.set_action([mu, sigma])

    done = False
    stepcount= 0
    while not done:
        stepcount+=1
        obs = ctrl.step()
        #print("C 1:" + str(ctrl.getTotalSteps()))
        # if(stepcount%50==0):
        #     gui_object.quads[str(quad_id)]['position'] = [obs[0],obs[1],obs[2]]
        #     gui_object.update()
         #   print(" ")
        if(stepcount > 30000):
            done = True
            c1Traj = ctrl.getTrajectory()
            steps1.append(ctrl.getTotalSteps())

            trajectories.append(c1Traj)
        if( ctrl.isAtPos(goals[current])):
            print(" Current :" + str(current) + " at step :" + str(stepcount))
            current += 1
            if (current < len(goals)):
                ctrl.update_target(goals[current])
            else:
                print("No more targets")
                done = True
                if (done):
                    if controller1 == []:
                        controller1 = ctrl.getTrackingErrors()

                    else:
                        controller1 = np.vstack((controller1, ctrl.getTrackingErrors()))
                    print("C1 : " + str(ctrl.getTotalSteps()))
                    c1Traj = ctrl.getTrajectory()
                    steps1.append(ctrl.getTotalSteps())

                    trajectories.append(c1Traj)

print("Controller 2 average error" + str(controller1.mean(0)))
bar_width = 0.2
print(controller1)

fig = plt.figure()
ax = Axes3D.Axes3D(fig)
ax.set_xlim3d([-6.0, 6.0])
ax.set_xlabel('X')
ax.set_ylim3d([-6.0, 6.0])
ax.set_ylabel('Y')
ax.set_zlim3d([0, 6.0])
ax.set_zlabel('Z')
ax.set_title('Quadcopter Simulation')

m = 0
for traj in trajectories:
    m += 1
    x_c1 = []
    y_c1 = []
    z_c1 = []
    error = [0, 0, 0]
    for i in range(len(traj)):
        x_c1.append(traj[i][0])
        y_c1.append(traj[i][1])
        z_c1.append(traj[i][2])
        if(i == starttime):
            error[0] = traj[i][0]
            error[1] = traj[i][1]
            error[2] = traj[i][2]
    ax.plot3D(x_c1, y_c1, z_c1, linewidth=0.2, c=plt.cm.jet(m/len(trajectories)))
    ax.scatter(error[0], error[1], error[2], c="r")
plt.show()
