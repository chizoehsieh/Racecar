import cv2
import torch
import os
import numpy as np

def play(env, agent, output_path = None, episode_index:int | None=0):
    wall = False

    with torch.no_grad():
        #Reset env
        reset_options = dict(mode='grid')
        # state, info = env.reset(options=reset_options)
        state = env.reset()

        if output_path is None:
            output_path = f'./videos/racecar-episode-{episode_index}.mp4'

        if not os.path.exists(os.path.dirname(output_path)):
            os.mkdir(os.path.dirname(output_path))

        step = 0
        total_reward = 0

        output_video_writer = cv2.VideoWriter(filename=output_path,
                                                fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                                fps=10,
                                                frameSize=(640,480))

        # print('state: ',state)
        if(len(state) == 2):
            envState = [state[0]['pose'], state[0]['acceleration'], state[0]['velocity'], state[0]['lidar']]
            state = list(state[1].values())
        else:
            state = list(state.values())
        
        for j in range(len(state)):
            i = state[j]
            if (isinstance(i, np.ndarray)) == False and (isinstance(i, list)) == False:
                if i == True and isinstance(i, bool):
                    i = np.zeros(6,dtype=np.float64)
                    i[0] = 1
                elif i == False and isinstance(i, bool):
                    i = np.zeros(6,dtype=np.float64)
                else:
                    l = np.zeros(6,dtype=np.float64)
                    l[0] = i
                    i = l
            if(isinstance(i, list)):
                l = np.zeros(6,dtype=np.float64)
                for m in range(len(i)):
                    l[m] = i[m]
                i = l
            state[j] = i
        
        for i in range(len(envState)):
            l = np.zeros(1080, dtype=np.float64)
            for m in range(len(envState[i])):
                l[m] = envState[i][m]
            envState[i] = l


        while True:
            # print('state: ',state)
            action = agent.act(envState,env,episode_index, wall)

            observation, rewards, done, _, states = env.step(action)

            lidarIndex = list(observation['lidar']).index(observation['lidar'].min())
            if lidarIndex >520 and lidarIndex < 575 and observation['lidar'].min() < 0.4:
                wall = True
            else:
                wall = False

            states = list(states.values())

            envStates = [observation['pose'], observation['acceleration'], observation['velocity'], observation['lidar']]
            for i in range(len(envStates)):
                l = np.zeros(1080, dtype=np.float64)
                for m in range(len(envStates[i])):
                    l[m] = envStates[i][m]
                envStates[i] = l

            if(len(state) == 2):
                state = list(state[1].values())

            for j in range(len(state)):
                i = state[j]
                k = states[j]
                if (isinstance(i, np.ndarray)) == False and (isinstance(i, list)) == False:
                    if i == True and isinstance(i, bool):
                        i = np.zeros(6,dtype=np.float64)
                        i[0] = 1
                    elif i == False and isinstance(i, bool):
                        i = np.zeros(6,dtype=np.float64)
                    else:
                        l = np.zeros(6,dtype=np.float64)
                        l[0] = i
                        i = l
                        # i = np.pad(np.ndarray(l),(0,5),'constant',constant_values=(0,0))
                if (isinstance(k, np.ndarray)) == False and (isinstance(k, list)) == False:
                    if isinstance(k, bool) and k == True:
                        k = np.zeros(6, dtype=np.float64)
                        k[0] = 1
                    elif isinstance(k, bool) and k == False:
                        k = np.zeros(6, dtype=np.float64)
                    else:
                        lk = np.zeros(6, dtype=np.float64)
                        lk[0] = k
                        k = lk
                if(isinstance(i, list)):
                    l = np.zeros(6,dtype=np.float64)
                    for m in range(len(i)):
                        l[m] = i[m]
                    i = l
                if(isinstance(k, list)):
                    lk = np.zeros(6, dtype=np.float64)
                    for m in range(len(k)):
                        lk[m] = k[m]
                    k = lk
                state[j] = i
                states[j] = k

            if step % 10 == 0:
                image = env.render().astype(np.uint8)
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                output_video_writer.write(frame)
            
            # state = np.array(list(states.values()))
            step += 1
            total_reward += rewards

            if done:
                break
        