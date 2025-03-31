from train_model import train
import gymnasium
import racecar_gym.envs.gym_api
import yaml
from racecarDQNAgent import RacecarDQNAgent
from playGame import play
import torch

scenario = './scenarios/custom.yml'
frame_size = dict(width=640, height=480) # default frame size 

env = gymnasium.make(
    id='SingleAgentRaceEnv-v0', 
    scenario=scenario,
    render_mode='rgb_array_birds_eye', # optional: 'rgb_array_birds_eye'
    render_options=frame_size
)

with open(f'{scenario}','r') as stream:
    config = yaml.load(stream, Loader=yaml.BaseLoader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env.reset()
# observation, rewards, done, _, states = env.step(env.action_space.sample())
# states = list(states.values())
# print(states)

agent = RacecarDQNAgent(action_space=env.action_space, input_shape=[4, 1080], device=device)
# agent.load('./qnet/qnet_349.pt')

# train model
train(env=env,agent=agent,max_round=500)

# evaluate
output_path = f'./videos/racecar-episode-evaluate.mp4'
agent.load('qnet.pt')
play(env=env, agent=agent, output_path=output_path)
