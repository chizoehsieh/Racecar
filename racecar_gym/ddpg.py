import argparse
import numpy as np
import gymnasium
import racecar_gym.envs.gym_api
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from itertools import count
from gymnasium.spaces import flatten_space
import time
import cv2
import sys
import os




class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action


    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = torch.tanh(x)

        return x
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, device, directory):
        self.device = device
        self.directory = directory
        self.memory_size = 50000
        self.state_dim, self.action_dim, self.max_action = state_dim, action_dim, max_action
        
        self.actor = Actor(1098, action_dim, max_action).to(self.device)
        self.actor_target = Actor(1098, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = 1e-4)

        self.critic = Critic(1098, action_dim).to(self.device)
        self.critic_target = Critic(1098, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.loss_func = nn.MSELoss()

        self.init_memory()

        # self.replay_buffer = Replay_buffer()
        # self.writer = SummartWriter(directory)

        # self.num_critic_update_iteration = 0
        # self.num_actor_update_iteration = 0

        # self.num_training = 0

    def convert(self, envState, isMemory):
        p = torch.FloatTensor(envState['pose'])
        l = torch.FloatTensor(envState['lidar'])
        v = torch.FloatTensor(envState['velocity'])
        a = torch.FloatTensor(envState['acceleration'])
        return [p, l, v, a]
        
    def init_memory(self):
        self.memory = {
            "s" : np.zeros((50000, 1098),dtype=np.float64),

            "a" : np.zeros((50000, 3),dtype=np.float64),
            "r" : np.zeros((50000, 1),dtype=np.float64),
            "s_" : np.zeros((50000, 1098),dtype=np.float64),
            "d" : np.zeros((50000, 1),dtype=bool),
        }

    def store_transition(self, s, a, r, s_, d):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        if self.memory_counter <= self.memory_size:
            index = self.memory_counter % self.memory_size
        else:
            index = np.random.randint(self.memory_size)
        self.memory["s"][index] = s
        self.memory["a"][index] = a
        self.memory["r"][index] = np.array(r).reshape(-1,1)
        self.memory["s_"][index] = s_
        self.memory["d"][index] = np.array(d).reshape(-1,1)
        self.memory_counter += 1

    def sample(self):
        ind = np.random.randint(0, self.memory_counter, size=100)
        b_s, b_a, b_r, b_s_, b_d = [], [], [], [], []
        for i in ind:
            b_s.append(np.array(self.memory['s'][i], copy=False))
            b_a.append(np.array(self.memory['a'][i], copy=False))
            b_r.append(np.array(self.memory['r'][i], copy=False))
            b_s_.append(np.array(self.memory['s_'][i], copy=False))
            b_d.append(np.array(self.memory['d'][i], copy=False))

        return np.array(b_s), np.array(b_a), np.array(b_r), np.array(b_s_), np.array(b_d)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state)[0].cpu().detach()
    
    def reward_func(self, observation, state, observation_, state_, best_progress):
        if(state_['progress'] == 1):
            return 100
        elif (state_['progress'] > best_progress):
            return (best_progress - state['progress']) * 100 + (state_['progress'] - best_progress) * 500
        elif(state_['progress'] > state['progress']):
            return (state_['progress'] - state['progress']) * 100
        elif(state_['progress'] == state['progress']):
            return -0.0001
        elif(state_['progress'] < state['progress']):
            return -0.00005
    

    def update(self):
        
        for it in range(200):
            s, a, r, s_, d = self.sample()
            state = torch.FloatTensor(s).to(device)
            action = torch.FloatTensor(a).to(device)
            reward = torch.FloatTensor(r).to(device)
            state_ = torch.FloatTensor(s_).to(device)
            done = torch.FloatTensor(d).to(device)

            target_Q = self.critic_target(state_, self.actor_target(state_))
            target_Q = reward + (done * 0.99 * target_Q).detach()

            current_Q = self.critic(state, action)


            critic_loss = F.mse_loss(current_Q, target_Q)
            # self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step = self.num_critic_update_iteration)

            self.critic_optimizer.zero_grad()
            # critic_loss.backward()
            self.critic_optimizer.step()


            actor_loss = -self.critic(state, self.actor(state)).mean()
            # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step = self.num_actor_update_iteration)

            self.actor_optimizer.zero_grad()
            # actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

            # self.num_actor_update_iteration += 1
            # self.num_critic_update_iteration += 1

        
    def save(self):
        torch.save(self.actor.state_dict(), self.directory + 'actor.pth')
        torch.save(self.critic.state_dict(), self.directory + 'critic.pth')

    def load(self):
        self.actor.load_state_dict(torch.load(self.directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(self.directory + 'critic.pth'))
        print("model has been loaded")


def train(env, agent, device):
    # agent.load()
    total_step = 0
    total_round = 1000
    var = 2
    best_progress = 0
    for i in range(total_round):
        total_reward = 0
        step = 0
        ep_r = 0
        wall = 0
        end_episode = False

        
        

        with torch.no_grad():
            state = env.reset()
            envState = state[0]

            envState_convert = np.concatenate((envState['pose'],envState['acceleration'],envState['velocity'],envState['lidar']))
            state = state[1]

            
            while True:
                action = agent.select_action(envState_convert)
                action = np.clip(np.random.normal(action, var), a_low_bound, a_bound)
                action = dict(zip(env.action_space.keys(), action))
                
                observation, rewards, done, _, states = env.step(action)
                envState_ = observation

                reward = agent.reward_func(envState, state, envState_, states, best_progress)

                
                envState__convert = np.concatenate((envState_['pose'],envState_['acceleration'],envState_['velocity'],envState_['lidar']))
                action_convert = [action['motor'], action['speed'], action['steering']]

                agent.store_transition(envState_convert, action_convert, reward, envState__convert, done)
                
                if(states['progress'] > best_progress):
                    best_progress = states['progress']
                
                if(states['progress'] == state['progress'] and np.array(envState_['lidar']).min() < 0.4):
                    wall += 1
                elif(states['progress'] == state['progress']):
                    wall += 0.5
                elif(states['progress'] < state['progress']):
                    wall += 0.8
                elif(states['progress'] > state['progress']):
                    wall = 0
                # else:
                #     wall = 0

                if wall > 600:
                    end_episode = True
                


                # agent.replay_buffer.push((envState, action, reward, envState_, np.float(done)))
                if total_step + step % 1000 == 0:
                    agent.update()
                if total_step + step > 50000:
                    var *= 0.9995
                

                if step % 100 == 0 :
                    sys.stdout.write(u"\u001b[1000D")
                    sys.stdout.flush()
                    sys.stdout.write('Round: '+ str(i) + ' | Step: ' + str(step) + ' | Reward: ' + str(rewards) + ' / ' + str(total_reward))

                envState = envState_.copy()
                envState_convert = envState__convert.copy()
                state = states.copy()

                step += 1
                total_reward += reward

                if done or end_episode:
                    print('\rRound: ', i, ' | Step: ',step,' | Reward: ',reward,' / ',total_reward,)
                    play(env, agent, i=i, best_progress=best_progress)
                    agent.save()
                    break
        total_step += step + 1
        


    
def evaluate(env, agent):
    
    ep_r = 0
    agent.load()
    output_path = f'./videos_ddpg/racecar-episode-evaluate.mp4'
    play(env, agent, output_path)

def play(env, agent, output_path = None, i: int | None=0, best_progress = 0):
    var = 3
    end_episode = False
    with torch.no_grad():
        #Reset env
        state = env.reset()

        if output_path is None:
            output_path = f'./videos_ddpg/racecar-episode-{i}.mp4'

        if not os.path.exists(os.path.dirname(output_path)):
            os.mkdir(os.path.dirname(output_path))

        step = 0
        total_reward = 0

        output_video_writer = cv2.VideoWriter(filename=output_path,
                                                fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                                fps=10,
                                                frameSize=(640,480))
        
        if(len(state) == 2):
            envState = state[0]
            state = state[1]


        envState_convert = np.concatenate((envState['pose'],envState['acceleration'],envState['velocity'],envState['lidar']))
        wall = 0
        while True:
            action = agent.select_action(envState_convert)
            action = np.clip(np.random.normal(action, var), a_low_bound, a_bound)
            action = dict(zip(env.action_space.keys(), action))
                
            observation, rewards, done, _, states = env.step(action)
            envState_ = observation
            envState__convert = np.concatenate((envState_['pose'],envState_['acceleration'],envState_['velocity'],envState_['lidar']))

            reward = agent.reward_func(envState, state, envState_, states, best_progress)

            if step % 10 == 0:
                image = env.render().astype(np.uint8)
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                output_video_writer.write(frame)

            # if(states['progress'] > best_progress):
            #     best_progress = states['progress']
                
            if(states['progress'] == state['progress'] and np.array(envState_['lidar']).min() < 0.4):
                wall += 1
            elif(states['progress'] == state['progress']):
                wall += 0.5
            elif(states['progress'] < state['progress']):
                wall += 0.8
            elif(states['progress'] > state['progress'] + 0.0001):
                wall = 0
            
            step += 1
            total_reward += rewards

            envState = envState_.copy()
            envState_convert = envState__convert.copy()
            state = states.copy()

            

            if wall > 600:
                    end_episode = True

            if done or end_episode:
                break
        output_video_writer.release()

scenario = './scenarios/custom.yml'
frame_size = dict(width=640, height=480) # default frame size 

env = gymnasium.make(
    id='SingleAgentRaceEnv-v0', 
    scenario=scenario,
    render_mode='rgb_array_birds_eye', # optional: 'rgb_array_birds_eye'
    render_options=frame_size
)

s_dim = flatten_space(env.observation_space).shape[0]
a_dim = flatten_space(env.action_space).shape[0]
a_bound = flatten_space(env.action_space).high
a_low_bound =flatten_space(env.action_space).low
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
directory = './qnet_ddpg/'
agent = DDPG(s_dim, a_dim, a_bound, device, directory)
t1 = time.time()
best_reward = float("-inf")
# train(env, agent, device) 
evaluate(env, agent)

env.close()
