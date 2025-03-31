import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, OrderedDict
# import intel_extension_for_pytorch as ipex


class RacecarDQNAgent:
    def __init__(
            self,
            action_space ,
            input_shape,
            device,
            memory_size = 5000,
            gamma = 0.95,
            epsilon = 1.0,
            epsilon_min = 0.1,
            epsilon_decay = 0.9999,
            learning_rate = 0.001
    ):
        self.action_space = action_space
        self.input_shape = input_shape
        self.device = device
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.eval()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
        self.learn_step_counter = 0
        self.init_memory()

    def build_model(self):
        # model = nn.Sequential(
        #     nn.Conv1d(12, 32, kernel_size = 3, padding = 1),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 64, kernel_size = 3, padding = 1),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 16, kernel_size = 3, padding = 1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.ReLU(),
        #     nn.Linear(16*6 , 3)
            
        # )

        model = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(16, 32, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(32, 64, kernel_size = 6),
            nn.ReLU(),
            nn.MaxPool1d(6),
            nn.Conv1d(64, 16, kernel_size = 6),
            nn.ReLU(),
            nn.MaxPool1d(6),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(16 * 2 ,3)
        )

        
        return model
    
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def memorize(self, state, action, reward, next_sate, done):
        # print('action: ',action)
        self.memory.append((state, action, reward, next_sate, done))

    def reward_func(self, state, states, envStates):
        if(states[5] == 1):
            return 1000
        elif(states[5] > state[5]):
            return (states[5] - state[5]) * 100 * envStates[3].min()
        elif(states[5] == state[5]):
            return -0.0001
        else:
            return 0

    def act(self, state, env, epsilon, wall):
        if np.random.rand() > epsilon:
            state = torch.FloatTensor([state]).to(self.device)
            act_values = self.model(state)
            act_values = act_values.detach().numpy()
            
            action = OrderedDict([('motor',np.array([act_values[0][0]], dtype=np.float32)), ('speed', np.array([act_values[0][1]], dtype=np.float32)), ('steering', np.array([act_values[0][2]], dtype=np.float32))])
        else:
            action = env.action_space.sample()
        
        return action

    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory,batch_size)
        train_state = []
        train_target = []
        for state, action_index, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[action_index] = reward
            else:
                t = self.target_model.predict(state)
                target[action_index] = reward + self.gamma * np.amax(t)

            train_state.append(state)
            train_target.append(target)

        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self):
        if self.learn_step_counter % 1000 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=6)
        else:
            sample_index = np.random.choice(self.memory_counter, size=6)
        

        b_s = torch.FloatTensor(self.memory['s'][sample_index]).to(self.device)
        
        b_a = torch.LongTensor(self.memory["a"][sample_index]).to(self.device)
        b_r = torch.FloatTensor(self.memory["r"][sample_index]).to(self.device)



        b_s_ = torch.FloatTensor(self.memory["s_"][sample_index]).to(self.device)
        


        b_d = torch.FloatTensor(self.memory["d"][sample_index]).to(self.device)


        q_curr_eval = self.model(b_s)
        q_next_target = self.target_model(b_s_).detach()

        q_next_eval = self.model(b_s_).detach()
        q_next_max = q_next_eval.max(1)[0].unsqueeze(1)

        q_curr_recur = b_r + (1 - b_d) * self.gamma * q_next_target
        self.loss = F.smooth_l1_loss(q_curr_eval, q_curr_recur)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1
        return self.loss.detach().cpu().numpy()

        

    
    def init_memory(self):
        self.memory = {
            "s" : np.zeros((self.memory_size, *self.input_shape),dtype=np.float64),

            "a" : np.zeros((self.memory_size, 3),dtype=np.float64),
            "r" : np.zeros((self.memory_size, 1),dtype=np.float64),
            "s_" : np.zeros((self.memory_size, *self.input_shape),dtype=np.float64),
            "d" : np.zeros((self.memory_size, 1),dtype=bool),
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

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.target_model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.target_model.state_dict(), name)
        