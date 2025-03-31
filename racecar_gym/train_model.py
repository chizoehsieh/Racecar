import numpy as np
from playGame import play
import sys
def train(env, agent, max_round=15):
    total_round = 0
    output_path = './videos/racecar_test_env.mp4'
    best_reward = -1000
    while True:
        # Reset env
        state = env.reset()

        # Initialize info
        step = 0
        total_reward = 0
        loss = 0
        wall = False

        if(len(state) == 2):
            envState = [state[0]['pose'], state[0]['acceleration'], state[0]['velocity'], state[0]['lidar']]
            state = list(state[1].values())
        else:
            state = list(state.values())
        
        # for j in range(len(state)):
        #     i = state[j]
        #     if (isinstance(i, np.ndarray)) == False and (isinstance(i, list)) == False:
        #         if i == True and isinstance(i, bool):
        #             i = np.zeros(6,dtype=np.float64)
        #             i[0] = 1
        #         elif i == False and isinstance(i, bool):
        #             i = np.zeros(6,dtype=np.float64)
        #         else:
        #             l = np.zeros(6,dtype=np.float64)
        #             l[0] = i
        #             i = l
        #     if(isinstance(i, list)):
        #         l = np.zeros(6,dtype=np.float64)
        #         for m in range(len(i)):
        #             l[m] = i[m]
        #         i = l
        #     state[j] = i
        
        for i in range(len(envState)):
            l = np.zeros(1080, dtype=np.float64)
            for m in range(len(envState[i])):
                l[m] = envState[i][m]
            envState[i] = l

        while True:
                # Select action
            action = agent.act(envState, env, total_round, wall)
                #Get next state
            observation, rewards, done, _, states = env.step(action)
            # print('observation: ',observation)
            lidarIndex = list(observation['lidar']).index(observation['lidar'].min())

            states = list(states.values())
            envStates = [observation['pose'], observation['acceleration'], observation['velocity'], observation['lidar']]
            for i in range(len(envStates)):
                l = np.zeros(1080, dtype=np.float64)
                for m in range(len(envStates[i])):
                    l[m] = envStates[i][m]
                envStates[i] = l

            if(len(state) == 2):
                state = list(state[1].values())
            
            rewards = agent.reward_func(state,states,envStates)

            # for j in range(len(state)):
            #     i = state[j]
            #     k = states[j]
            #     if (isinstance(i, np.ndarray)) == False and (isinstance(i, list)) == False:
            #         if i == True and isinstance(i, bool):
            #             i = np.zeros(6,dtype=np.float64)
            #             i[0] = 1
            #         elif i == False and isinstance(i, bool):
            #             i = np.zeros(6,dtype=np.float64)
            #         else:
            #             l = np.zeros(6,dtype=np.float64)
            #             l[0] = i
            #             i = l
            #             # i = np.pad(np.ndarray(l),(0,5),'constant',constant_values=(0,0))
            #     if (isinstance(k, np.ndarray)) == False and (isinstance(k, list)) == False:
            #         if isinstance(k, bool) and k == True:
            #             k = np.zeros(6, dtype=np.float64)
            #             k[0] = 1
            #         elif isinstance(k, bool) and k == False:
            #             k = np.zeros(6, dtype=np.float64)
            #         else:
            #             lk = np.zeros(6, dtype=np.float64)
            #             lk[0] = k
            #             k = lk
            #     if(isinstance(i, list)):
            #         l = np.zeros(6,dtype=np.float64)
            #         for m in range(len(i)):
            #             l[m] = i[m]
            #         i = l
            #     if(isinstance(k, list)):
            #         lk = np.zeros(6, dtype=np.float64)
            #         for m in range(len(k)):
            #             lk[m] = k[m]
            #         k = lk
            #     state[j] = i
            #     states[j] = k
            agent.store_transition(

                                   envState,
                                   [action['motor'][0],action['speed'][0],action['steering'][0]], 
                                   rewards, 
                                   envStates,
                                   done)
            # print('print store')
            if step > 128:
                loss = agent.learn()

            state = states.copy()
            envState = envStates.copy()
            step += 1
            total_reward += rewards

            if step % 100 == 0 :
                # print('\033[2kRound: ',total_round,' | Step: ',step,' | Reward: ',rewards,' / ',total_reward, flush=True)
                sys.stdout.write(u"\u001b[1000D")
                sys.stdout.flush()
                sys.stdout.write('Round: '+ str(total_round) + ' | Step: ' + str(step) + ' | Reward: ' + str(rewards) + ' / ' + str(total_reward))

            if done:
                print('\rRound: ',total_round,' | Step: ',step,' | Reward: ',rewards,' / ',total_reward,' | Loss: ',loss)
                # print('\033[2kRound: ',total_round,' | Step: ',step,' | Reward: ',rewards,' / ',total_reward, flush=True)
                agent.save('./qnet/qnet_' + str(total_round) + '.pt')
                if total_reward > best_reward:
                    agent.save('qnet.pt')
                play(env=env,episode_index=total_round,agent=agent)
                total_round += 1
                print()
                break
            
        if total_round >= max_round:
            break