#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from feature import FeatureAgent
from model import MyModel
import numpy as np
import torch
import sys
import random

# seatWind = -1
# agent = None
# angang = None
# zimo = False

def obs2action(agent, model, obs):
    model.train(False)

    legal_actions = np.where(obs['action_mask'] == 1)[0]
    assert legal_actions.size

    if 1 in legal_actions:
        return 1
    
    next_states_obs_list = []
    choices_for_value_eval = []

    for action_idx in legal_actions:
        next_state_observation_np = agent.get_next_state(action_idx)
        next_states_obs_list.append(next_state_observation_np)
        choices_for_value_eval.append((action_idx, next_state_observation_np))

    batched_next_states_tensor = torch.from_numpy(np.stack(next_states_obs_list)).float()

    with torch.no_grad():
        values_tensor = model(batched_next_states_tensor)

    epsilon = 0.00

    if random.random() < epsilon:
        my_action = random.choice(legal_actions)
    else:
        actions_with_values = []
        for i, (action_idx, _) in enumerate(choices_for_value_eval):
            actions_with_values.append((action_idx, values_tensor[i].item()))
        
        random.shuffle(actions_with_values)
        max_value = -float('inf')
        best_action_idx = -1
        for action_idx, value in actions_with_values:
            if value > max_value:
                max_value = value
                best_action_idx = action_idx
        my_action = best_action_idx

    return my_action

def obs2response(agent, model, obs):
    action = obs2action(agent, model, obs)
    return agent.action2response(action)

# ----------------------------------------------------
if __name__ == '__main__':
    model = MyModel()
    model.load_state_dict(torch.load('data/0626202920_37000.pt', map_location=torch.device('cpu')))
    model.eval()

    input()

    while True:
        request = input()
        while not request.strip(): request = input()
        request = request.split()
        if request[0] == '0':
            seatWind = int(request[1])
            agent = FeatureAgent(seatWind)
            agent.request2obs('Wind %s' % request[2])
            print('PASS')
        elif request[0] == '1':
            agent.request2obs(' '.join(['Deal', *request[5:]]))
            print('PASS')
        elif request[0] == '2':
            obs = agent.request2obs('Draw %s' % request[1])
            response = obs2response(agent, model, obs)
            response = response.split()
            if response[0] == 'Hu':
                print('HU')
            elif response[0] == 'Play':
                print('PLAY %s' % response[1])
            elif response[0] == 'Gang':
                print('GANG %s' % response[1])
                angang = response[1]
            elif response[0] == 'BuGang':
                print('BUGANG %s' % response[1])
        elif request[0] == '3':
            p = int(request[1])
            if request[2] == 'DRAW':
                agent.request2obs('Player %d Draw' % p)
                zimo = True
                print('PASS')
            elif request[2] == 'GANG':
                if p == seatWind and angang:
                    agent.request2obs('Player %d AnGang %s' % (p, angang))
                elif zimo:
                    agent.request2obs('Player %d AnGang' % p)
                else:
                    agent.request2obs('Player %d Gang' % p)
                print('PASS')
            elif request[2] == 'BUGANG':
                obs = agent.request2obs('Player %d BuGang %s' % (p, request[3]))
                if p == seatWind:
                    print('PASS')
                else:
                    response = obs2response(agent, model, obs)
                    if response == 'Hu':
                        print('HU')
                    else:
                        print('PASS')
            else:
                zimo = False
                if request[2] == 'CHI':
                    agent.request2obs('Player %d Chi %s' % (p, request[3]))
                elif request[2] == 'PENG':
                    agent.request2obs('Player %d Peng' % p)
                obs = agent.request2obs('Player %d Play %s' % (p, request[-1]))
                if p == seatWind:
                    print('PASS')
                else:
                    response = obs2response(agent, model, obs)
                    response = response.split()
                    if response[0] == 'Hu':
                        print('HU')
                    elif response[0] == 'Pass':
                        print('PASS')
                    elif response[0] == 'Gang':
                        print('GANG')
                        angang = None
                    elif response[0] in ('Peng', 'Chi'):
                        obs = agent.request2obs('Player %d '% seatWind + ' '.join(response))
                        response2 = obs2response(agent, model, obs)
                        print(' '.join([response[0].upper(), *response[1:], response2.split()[-1]]))
                        agent.request2obs('Player %d Un' % seatWind + ' '.join(response))
        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        sys.stdout.flush()