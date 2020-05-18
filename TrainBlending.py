
from Quad_Env import Quad_Env
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_vec_env



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


env = Quad_Env()
env = make_vec_env(lambda: env, n_envs=1)
# If the environment don't follow the interface, an error will be thrown
obs = env.reset()

model = PPO2(MlpLnLstmPolicy, env ,nminibatches=1 , tensorboard_log="./stationary_env_ppo/" )

model.learn(total_timesteps=100000,  log_interval=4000)

model.save("ppo_30rotor_fault_blending")

print("Training complete - agent saved")