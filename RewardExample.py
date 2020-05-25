from MetaLearningControl import quadcopter,controller,gui
import numpy as np


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
n = 2
steps1 = []
steps2 = []
total_steps = [ ]
starttime = 1800
trajectories = []
trajectories2 = []
trajectories3 = []
fault_mag = 0.1

# gui_object = gui.GUI(quads=Quads)

# t = np.linspace(0, , 100)
# x_path = np.linspace(-5, 5, steps)
# y_path = np.linspace(-5, 5, steps)
# z_path = np.linspace(1, 10, steps)
stepsToGoal = 0
steps = 8
x_path = [0, 0, 5, 0, -5, 0, 5, 0]
y_path = [0, 0, 0, 5, 0, -5, 0, 0]
z_path = [0, 5, 5, 5, 5, 5, 5, 5]
interval_steps = 50
yaws = np.zeros(steps)
goals = []
safe_region = []
for i in range(steps-1):

    #create linespace between waypoint i and i+1
    x_lin = np.linspace(x_path[i], x_path[i+1], interval_steps)
    y_lin =  np.linspace(y_path[i], y_path[i+1], interval_steps)
    z_lin =  np.linspace(z_path[i], z_path[i+1], interval_steps)
    goals.append([x_path[i], y_path[i], z_path[i]])
    #for each pos in linespace append a goal
    safe_region.append([])
    for j in range(interval_steps):
        safe_region[i].append([x_lin[j], y_lin[j], z_lin[j]])
        stepsToGoal +=1

print("Safe Region :" + str(safe_region))

for i in range(n):
    quad_id = i

    QUADCOPTER = {
        str(quad_id): {'position': [0, 0, 0], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1, 'prop_size': [10, 4.5],
                       'weight': 1.2}}

    # Make objects for quadcopter

    quad = quadcopter.Quadcopter(QUADCOPTER)
    Quads = {str(quad_id): QUADCOPTER[str(quad_id)]}



    # create blended controller and link it to quadcopter object
    ctrl = controller.Blended_PID_Controller(quad.get_state, quad.get_time,
                                             quad.set_motor_speeds, quad.get_motor_speeds,
                                             quad.stepQuad, quad.set_motor_faults,
                                             params=BLENDED_CONTROLLER_PARAMETERS, quad_identifier=str(quad_id))

    gui_object = gui.GUI(quads=Quads, ctrl=ctrl)


    #print(str(goals))
    #goal = [goalx, goaly, goalz]
   #goal = [-10, -10, 5]
    current = 0
    ctrl.update_target(goals[current] , safe_region[current])
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
    reward = 0
    mu = 0.5
    sigma = 0.1
    obs = ctrl.set_action([mu, sigma])
    ctrl.setController("C2")
    done = False
    stepcount= 0
    while not done:
        stepcount+=1
        obs = ctrl.step()
        reward += ctrl.getReward()
        #print("C 1:" + str(ctrl.getTotalSteps()))
        if(stepcount%20==0):
            gui_object.quads[str(quad_id)]['position'] = [obs[0],obs[1],obs[2]]
            gui_object.update()

        if(stepcount > 30000):
            done = True
            c1Traj = ctrl.getTrajectory()
            steps1.append(ctrl.getTotalSteps())

            trajectories.append(c1Traj)
        if( ctrl.isAtPos(goals[current])):

            current += 1
            if (current < len(goals)):
                ctrl.update_target(goals[current], safe_region[current-1])
            else:

                done = True
                if (done):
                    if controller1 == []:
                        controller1 = ctrl.getTrackingErrors()

                    else:
                        controller1 = np.vstack((controller1, ctrl.getTrackingErrors()))

                    c1Traj = ctrl.getTrajectory()
                    steps1.append(ctrl.getTotalSteps())

                    trajectories.append(c1Traj)

    gui_object.close()
    print("Cumu Reward = " + str(reward))