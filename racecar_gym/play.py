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




class Actor(nn.Module):  # 根據輸入的狀態，輸出動作的策略網路
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
    

class Critic(nn.Module):  # 根據輸入的狀態與動作，評估Q值
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

        self.init_memory()

        
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
        state = torch.FloatTensor(state).to(self.device)
        return self.actor(state)[0].cpu().detach()
    
    def reward_func(self, observation, state, observation_, state_, best_progress):
        if(state_['progress'] == 1):  # 若完成一圈
            return 100                # 給予高額獎勵
        elif (state_['progress'] > best_progress):   # 若探索到新的地方(好奇心獎勵)，新的進度有5倍獎勵
            return (best_progress - state['progress']) * 100 + (state_['progress'] - best_progress) * 500
        elif(state_['progress'] > state['progress']):  # 若車子有前進
            return (state_['progress'] - state['progress']) * 100 # 獎勵為前進的距離
        elif(state_['progress'] == state['progress']): # 若車子停在原地
            return -0.0001                             # 給予懲罰
        elif(state_['progress'] < state['progress']):  # 若車子往回走
            return -0.00005                            # 給予小懲罰
    

    def update(self):
        
        for it in range(200):
            s, a, r, s_, d = self.sample()
            state = torch.FloatTensor(s).to(self.device)
            action = torch.FloatTensor(a).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            state_ = torch.FloatTensor(s_).to(self.device)
            done = torch.FloatTensor(d).to(self.device)

            target_Q = self.critic_target(state_, self.actor_target(state_))
            target_Q = reward + (done * 0.99 * target_Q).detach()


            self.critic_optimizer.zero_grad()
            self.critic_optimizer.step()



            self.actor_optimizer.zero_grad()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)


        
    def save(self):
        torch.save(self.actor.state_dict(), self.directory + 'actor.pth')
        torch.save(self.critic.state_dict(), self.directory + 'critic.pth')

    def load(self):
        self.actor.load_state_dict(torch.load(self.directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(self.directory + 'critic.pth'))
        print("model has been loaded")


def play(env, agent, output_path = None, i: int | None=0, best_progress = 0):
    var = 3   # 動作探索率
    end_episode = False  # 是否結束此episode
    with torch.no_grad():
        #Reset env
        state = env.reset()

        if output_path is None:             # 若沒有指定影片輸出路徑
            output_path = f'./videos_ddpg/racecar-episode-{i}.mp4'  # 存成訓練episod的影片

        if not os.path.exists(os.path.dirname(output_path)):  # 若影片輸出路徑不存在
            os.mkdir(os.path.dirname(output_path))            # 建立路徑

        step = 0                            # 代理人執行步數
        total_reward = 0                    # 得到的總獎勵

        output_video_writer = cv2.VideoWriter(filename=output_path,           # 建立影片輸出的格式
                                                fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                                fps=10,
                                                frameSize=(640,480))
        

        envState = state[0]   # 儲存observation的內容
        state = state[1]      # 儲存state的內容


        # 將observation的內容串聯成一個一維陣列
        envState_convert = np.concatenate((envState['pose'],envState['acceleration'],envState['velocity'],envState['lidar']))
        wall = 0  # 儲存撞牆次數
        while True:  # 當episode未停止
            action = agent.select_action(envState_convert)  # 代理人選擇動作
            action = np.clip(np.random.normal(action, var), a_low_bound, a_bound)  # 對動作做正態擬合
            action = dict(zip(env.action_space.keys(), action)) # 將動作轉成字典型態
                
            observation, rewards, done, _, states = env.step(action)  # 與環境互動，得到回傳參數
            envState_ = observation
            # 將envState_的內容串聯成一個一維陣列
            envState__convert = np.concatenate((envState_['pose'],envState_['acceleration'],envState_['velocity'],envState_['lidar']))

            reward = agent.reward_func(envState, state, envState_, states, best_progress)  # 從獎勵函數取得獎勵值

            if step % 10 == 0:          # 若執行10個step，儲存一幀影片
                image = env.render().astype(np.uint8)
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                output_video_writer.write(frame)

            # 若車子沒有前進，且雷達最小值小於0.4，代表車子撞牆了，撞牆次數加1
            if(states['progress'] == state['progress'] and np.array(envState_['lidar']).min() < 0.4):  
                wall += 1
            # 若車子沒有前進，有可能是環境尚未更新，也有可能是撞牆了，所以撞牆次數+ 0.5
            elif(states['progress'] == state['progress']):  
                wall += 0.5
            # 若車子向後退，有可能是在倒車調整方向，但大多都是走錯路，因此+ 0.8
            elif(states['progress'] < state['progress']):
                wall += 0.8
            # 若車子向前走，撞牆次數歸零，避免下次一停下，就結束遊戲
            elif(states['progress'] > state['progress'] + 0.0001):
                wall = 0
            
            step += 1    # 執行步數增加
            total_reward += rewards   # 加上此step所得獎勵

            envState = envState_.copy()  # 將new state放入原state
            envState_convert = envState__convert.copy()  # 將new state放入原state
            state = states.copy()        # 將new state放入原state

            

            if wall > 600:       # 若撞牆次數累積600次，大略無法再前進，結束遊戲
                    end_episode = True

            if done or end_episode:  # 若車子抵達終點、時間到，或自行設定的結束遊戲
                break                # 跳出迴圈，結束此episode
        output_video_writer.release()   # 釋放影片更改的權限

scenario = './scenarios/custom.yml'     # 設定環境檔路徑
frame_size = dict(width=640, height=480) # default frame size 

env = gymnasium.make(                  # 建立環境
    id='SingleAgentRaceEnv-v0', 
    scenario=scenario,
    render_mode='rgb_array_birds_eye', 
    render_options=frame_size
)

s_dim = flatten_space(env.observation_space).shape[0]   # 定義狀態的維度
a_dim = flatten_space(env.action_space).shape[0]        # 定義動作空間的維度
a_bound = flatten_space(env.action_space).high          # 定義動作的最大值
a_low_bound =flatten_space(env.action_space).low        # 定義動作的最小值
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 決定模型訓練的地方
directory = './qnet_ddpg/'                              # 定義模型存放的路徑
agent = DDPG(s_dim, a_dim, a_bound, device, directory)  # 建立代理人
agent.load()                                            # 代理人讀取訓練完的模型檔
output_path = f'./videos_ddpg/racecar-episode-evaluate.mp4'  # 定義影片輸出路徑
play(env, agent, output_path)                           # 遊玩一次遊戲
env.close()                                             # 關閉環境