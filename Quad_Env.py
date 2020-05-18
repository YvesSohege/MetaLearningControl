import gym
from gym import spaces
import quadcopter,controller
import numpy as np
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary, Dict



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

#original angular rate = {'P': [22000, 22000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]}
steps = 7
x_path = [0, 5, 0, -5, 0, 5, 0]
y_path = [0, 0, 5, 0, -5, 0, 0]
z_path = [5, 5, 5, 5, 5, 5, 5]

yaws = np.zeros(steps)
goals = []
for i in range(steps):
    goals.append([x_path[i], y_path[i], z_path[i]])




class Quad_Env(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    #metadata = {'render.modes': ['console']}
    # Define constants for clearer code


    def __init__(self):
        super(Quad_Env, self).__init__()

        self.quad_id = 1
        QUADCOPTER = {str(self.quad_id): {'position': [0, 0, 0], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1, 'prop_size': [10, 4.5],
                            'weight': 1.2}}
        # Make objects for quadcopter
        self.quad = quadcopter.Quadcopter(QUADCOPTER)
        #self.gui_object = gui.GUI(quads=QUADCOPTER)
        self.sampleTime = 0.01
        #create blended controller and link it to quadcopter object
        self.ctrl = controller.Blended_PID_Controller(self.quad.get_state, self.quad.get_time,
                                                      self.quad.set_motor_speeds, self.quad.get_motor_speeds,
                                                      self.quad.stepQuad, self.quad.set_motor_faults,
                                                      params=BLENDED_CONTROLLER_PARAMETERS, quad_identifier=str(self.quad_id))

        self.current = 0
        self.ctrl.update_target(goals[0])
        self.setRandomFault()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 3
        #PID LOW and HIHG
        self.actionlow = 0.01
        self.actionhigh =1
        self.action_low_state = np.array([self.actionlow, self.actionlow], dtype=np.float)

        self.action_high_state = np.array([self.actionhigh, self.actionhigh], dtype=np.float)

        self.action_space = spaces.Box(low=self.action_low_state, high=self.action_high_state, dtype=np.float)


        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space

       # print(self.observation_space)
        self.low_error=-1
        self.high_error=1
        self.low_dest= -10
        self.high_dest =10
        self.low_act = -10000
        self.high_act = 10000
        self.low = 0
        self.high = 10000
        self.low_state = np.array([self.low_dest, self.low_dest,self.low_dest, self.low_dest, self.low_dest, self.low_dest,
                                   ], dtype=np.float32)
       # self.low_state = np.array([self.low_dest,self.low_dest,  self.low_error,self.low_error, self.low_act, self.low_act, self.low_act, self.low_act])
        #self.high_state = np.array([self.high_dest,self.high_dest, self.high_error, self.high_error, self.high_act, self.high_act, self.high_act,self.high_act])

        self.high_state = np.array([self.high_dest, self.high_dest,self.high_dest, self.high_dest, self.high_dest, self.high_dest,
                                    ], dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,dtype=np.float32)





    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent at the right of the grid
        self.quad_id += 1
        QUADCOPTER = {str(self.quad_id): {'position': [0, 0, 0], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1,
                                          'prop_size': [10, 4.5],
                                          'weight': 1.2}}
        self.quad = quadcopter.Quadcopter(QUADCOPTER)
        #self.gui_object = gui.GUI(quads=QUADCOPTER)
        self.ctrl = controller.Blended_PID_Controller(self.quad.get_state, self.quad.get_time,
                                                      self.quad.set_motor_speeds, self.quad.get_motor_speeds,
                                                      self.quad.stepQuad, self.quad.set_motor_faults,
                                                      params=BLENDED_CONTROLLER_PARAMETERS, quad_identifier=str(self.quad_id))

        self.current = 0
        self.ctrl.update_target(goals[0])
        #print(type(self.quad.get_state("q1")))
        self.setRandomFault()
        obs = self.ctrl.get_updated_observations()
        obs_array = []
        for key, value in obs.items():

            obs_array.append(value)
        return obs_array
        # here we convert to float32 to make it more general (in case we want to use continuous actions)

    def step(self, action):

        #print("Action: " + str(action))
        #action will be probability distribution for blend space
        #one step is a full trajectory with given blending dist.
        obs = self.ctrl.set_action(action)
        # obs = ctrl.set_action([mu, sigma])

        done = False
        failed = False
        if (self.ctrl.isAtPos(goals[self.current])):
            self.current += 1
            if (self.current < len(goals)):
                self.ctrl.update_target(goals[self.current])
            else:
                print("No more targets")
                done = True

        if self.ctrl.getTotalSteps() > 10000:
            done = True
            failed = True
            print("Failed")



        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = self.ctrl.getReward()

        # if(failed):
        #     reward = -100
        if(done):
            print("Ep done " +str(reward))
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        #self.render()
        return np.array(obs), reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        #self.gui_object.quads['q1']['position'] = self.quad.get_position('q1')
        #self.gui_object.quads['q1']['orientation'] = self.quad.get_orientation('q1')
        #self.gui_object.update()


    def close(self):
        pass


    def setRandomFault(self):
        faults = [0, 0, 0, 0]
        #fault_mag = np.random.uniform(0, 0.3)
        fault_mag = 0.3
        #rotor = np.random.randint(0, 4)
        rotor = 1
        faults[rotor] = fault_mag
        self.ctrl.setMotorFault(faults)

        starttime = 1800
        endtime = 31000

        self.ctrl.setFaultTime(starttime, endtime)
