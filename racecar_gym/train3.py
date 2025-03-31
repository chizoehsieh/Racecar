import gymnasium
import torch
import random
import os
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import racecar_gym.envs.gym_api
from play_and_evaluate import play
import torch.nn.functional as F
import yaml
import cv2
from PIL import Image
import gym
from itertools import chain
from gymnasium.spaces import flatten_space
import time

#####################  hyper parameters  ####################
EPISODES = 10000
LR_ACTOR = 0.001
LR_CRITIC = 0.002
GAMMA = 0.95
TAU = 0.01
MEMORY_CAPACITY = 50000
BATCH_SIZE = 64
RENDER = False
SCENARIO='./scenarios/custom.yml'
FRAME_SIZE = dict(width=640, height=480) # default frame size 
ENV_NAME = 'SingleAgentRaceEnv-v0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_DIR = './Model/test.qt'

########################## DDPG Framework ######################
class ActorNet(nn.Module): # define the network structure for actor and critic
    def __init__(self, s_dim, a_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1) # initialization of FC1
        self.out = nn.Linear(30, a_dim)
        self.out.weight.data.normal_(0, 0.1) # initilizaiton of OUT
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions = x
        return actions

class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)
    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        actions_value = self.out(F.relu(x+y))
        return actions_value
    
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,device):
        self.device = device
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim+1), dtype=np.float32)
        self.pointer = 0 # serves as updating the memory data 
        # Create the 4 network objects
        self.actor_eval = ActorNet(s_dim, a_dim).to(self.device)
        self.actor_target = ActorNet(s_dim, a_dim).to(self.device)
        self.critic_eval = CriticNet(s_dim, a_dim).to(self.device)
        self.critic_target = CriticNet(s_dim, a_dim).to(self.device)
        # create 2 optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_CRITIC)
        # Define the loss function for critic network update
        self.loss_func = nn.MSELoss()
    
    def convert_state(self,state,isMemory):
        tensor1 = torch.FloatTensor( state['pose'])
        tensor2 = torch.FloatTensor( state['lidar'])
        tensor3 = torch.FloatTensor( state['velocity'])
        scalar_value =state['time']
        scalar_tensor = torch.tensor(scalar_value).unsqueeze(0)
        if isMemory:      
            return torch.cat((tensor1, tensor2, tensor3, scalar_tensor), dim=0)
        else:
            return torch.cat((tensor1, tensor2, tensor3, scalar_tensor), dim=0).to(self.device)

    def store_transition(self, s, a, r, s_): # how to store the episodic data to buffer
        s_convert =self.convert_state(s,True)
        s_next_convert=self.convert_state(s_,True)
        transition = np.hstack((s_convert, a, [r], s_next_convert))
        index = self.pointer % MEMORY_CAPACITY # replace the old data with new data 
        self.memory[index, :] = transition
        self.pointer += 1
    
    def choose_action(self, s):
        input_tensor = self.convert_state(s,False)
        return self.actor_eval(input_tensor)[0].cpu().detach()
    
    def learn(self):
        # softly update the target networks
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')           
        # sample from buffer a mini-batch data
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_trans = self.memory[indices, :]
        # extract data from mini-batch of transitions including s, a, r, s_
        batch_s = torch.FloatTensor(batch_trans[:, :self.s_dim]).to(self.device)
        batch_a = torch.FloatTensor(batch_trans[:, self.s_dim:self.s_dim + self.a_dim]).to(self.device)
        batch_r = torch.FloatTensor(batch_trans[:, -self.s_dim - 1: -self.s_dim]).to(self.device)
        batch_s_ = torch.FloatTensor(batch_trans[:, -self.s_dim:]).to(self.device)
        # make action and evaluate its action values
        a = self.actor_eval(batch_s)
        q = self.critic_eval(batch_s, a)
        actor_loss = -torch.mean(q)
        actor_loss.requires_grad_(True)
        actor_loss.to(self.device)
        # optimize the loss of actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # compute the target Q value using the information of next state
        a_target = self.actor_target(batch_s_)
        q_tmp = self.critic_target(batch_s_, a_target)
        q_target = batch_r + GAMMA * q_tmp
        # compute the current q value and the loss
        q_eval = self.critic_eval(batch_s, batch_a)
        td_error = self.loss_func(q_target, q_eval)
        td_error.requires_grad_(True)
        td_error.to(self.device)
        # optimize the loss of critic network
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()
    
    def save_load_model(self, op, path="save"):
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        file_path_a = os.path.join(path, 'actorNet.qt')
        file_path_c = os.path.join(path, 'criticNet.qt')
        if op == "save":
            torch.save(self.actor_eval.state_dict(), file_path_a)
            torch.save(self.critic_eval.state_dict(), file_path_c)
        elif op == "load":
            self.actor_eval.load_state_dict(torch.load(file_path_a, map_location=self.device))
            self.actor_target.load_state_dict(torch.load(file_path_a, map_location=self.device))
            self.critic_eval.load_state_dict(torch.load(file_path_c, map_location=self.device))
            self.critic_target.load_state_dict(torch.load(file_path_c, map_location=self.device))

env = gymnasium.make(
    id=ENV_NAME, 
    scenario=SCENARIO,
    render_mode='rgb_array_birds_eye', # optional: 'rgb_array_birds_eye'
    render_options=FRAME_SIZE
)

s_dim = flatten_space(env.observation_space).shape[0]
a_dim = flatten_space(env.action_space).shape[0]
a_bound = flatten_space(env.action_space).high
a_low_bound =flatten_space(env.action_space).low
ddpg = DDPG(a_dim, s_dim, a_bound,DEVICE)
var = 3 # the controller of exploration which will decay during training process
t1 = time.time()

best_reward = float("-inf")

for i in range(EPISODES):

    best_check_point = 0
    best_progress = 0

    with torch.no_grad():
        render_env = env
        reset_options = dict(mode='grid')
        s = render_env.reset(options=reset_options)
        s=s[0]
        ep_r = 0
        length = 0
        total_rewards = 0
        stock_count=0
        end_episode=False

        output_path = f'./videos/racecar-episode-{i}.mp4'

        output_video_writer = cv2.VideoWriter(filename=output_path,
                                              fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                              fps=10,
                                              frameSize=(640,480))

        print(f'[Evaluation] Running evaluation and recording video for episode-{i}')

        while True:
            if RENDER: render_env.render()
            # add explorative noise to action
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), a_low_bound, a_bound)
            a_convert = dict(zip(render_env.action_space.keys(), a)) # convert to dict type action
            s_, r, done,_, info = render_env.step(a_convert)

            if best_check_point<info['checkpoint']:
                best_check_point=info['checkpoint']
                t=info['time']
                point=info['checkpoint']
                r += info['checkpoint']*100/t
                print(f'[checkpoint] Arrive at checkpoint {point} get reward:{r},use time {t}')
            else:
                if info['wrong_way']==True:
                    r -= 10
                elif info['wall_collision']==True:
                    r -= 2

            if info['progress']-best_progress>0.0005:
                r += 0.5
            if info['progress']>best_progress:
                best_progress=info['progress']
            
            if info['wall_collision']==True and info['progress']-best_progress<0.1:
                stock_count += 1
            else:
                stock_count = 0

            if stock_count>300:
                r-=50
                end_episode=True

            ddpg.store_transition(s, a, r / 10, s_) # store the transition to memory
        
            if ddpg.pointer > MEMORY_CAPACITY:
                var *= 0.9995 # decay the exploration controller factor
                ddpg.learn()

            if length % 10 == 0: # save 1 frame per 10 step
                image = render_env.render().astype(np.uint8)
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                output_video_writer.write(frame)

            total_rewards += r
            s = s_
            length += 1

            if done or stock_count>500:
                print("[Evaluation] total reward = {:.6f}, length = {:d}".format(total_rewards/length, length), flush=True)
                # save the best model with best reward
                
                if (total_rewards/length > best_reward) and (length > 5000):
                    best_reward = total_rewards/length
                    print("\nSaving the best model ... ", end="")
                    ddpg.save_load_model(op="save", path=MODEL_SAVE_DIR)
                    print("Done.")
                break

        output_video_writer.release()
        render_env.close()

print('Running time: ', time.time() - t1)