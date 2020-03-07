import numpy as np
from copy import deepcopy
from functools import reduce
from operator import add
import os

HEALTH_RANGE = 5
ARROWS_RANGE = 4
STAMINA_RANGE = 3

HEALTH_VALUES = tuple(range(HEALTH_RANGE))
ARROWS_VALUES = tuple(range(ARROWS_RANGE))
STAMINA_VALUES = tuple(range(STAMINA_RANGE))

HEALTH_FACTOR = 25 # 0, 25, 50, 75, 100
ARROWS_FACTOR = 1 # 0, 1, 2, 3
STAMINA_FACTOR = 50 # 0, 50, 100

NUM_ACTIONS = 3
ACTION_SHOOT = 0
ACTION_DODGE = 1
ACTION_RECHARGE = 2

TEAM = 85
Y = [1/2, 1,2]
PRIZE = 10
COST = -10/Y[TEAM%3]

GAMMA = 0.99
DELTA = 0.001

REWARD = np.zeros((HEALTH_RANGE, ARROWS_RANGE, STAMINA_RANGE))
REWARD[0, :, :] = PRIZE

class State:
    def __init__(self, enemy_health, num_arrows, stamina):
        if (enemy_health not in HEALTH_VALUES) or (num_arrows not in ARROWS_VALUES) or (stamina not in STAMINA_VALUES):
            raise ValueError
        
        self.health = enemy_health 
        self.arrows = num_arrows 
        self.stamina = stamina 

    def show(self):
        return (self.health, self.arrows, self.stamina)

    def __str__(self):
        return f'({self.health},{self.arrows},{self.stamina})'

def action(action_type, state, costs):
    # returns cost, array of tuple of (probability, state)

    state = State(*state)

    if action_type == ACTION_SHOOT:
        if state.arrows == 0 or state.stamina == 0:
            return None, None

        new_arrows = state.arrows - 1
        new_stamina = state.stamina - 1

        choices = []
        choices.append(
            (0.5, State(max(HEALTH_VALUES[0], state.health-1), new_arrows, new_stamina)))
        choices.append((0.5, State(state.health, new_arrows, new_stamina)))

        cost = 0
        for choice in choices:
            cost += choice[0] * (costs[ACTION_SHOOT] + REWARD[choice[1].show()])

        return cost, choices

    elif action_type == ACTION_RECHARGE:
        choices = []
        choices.append((0.8, State(state.health, state.arrows,
                                   min(STAMINA_VALUES[-1], state.stamina+1))))
        choices.append((0.2, State(state.health, state.arrows, state.stamina)))

        cost = 0
        for choice in choices:
            cost += choice[0] * (costs[ACTION_RECHARGE] + REWARD[choice[1].show()])

        return cost, choices

    elif action_type == ACTION_DODGE:
        if state.stamina == 0:
            return None, None

        choices = []
        choices.append((0.64, State(state.health, min(
            ARROWS_VALUES[-1], state.arrows+1), max(STAMINA_VALUES[0], state.stamina - 1))))
        choices.append((0.16, State(state.health, state.arrows,
                                    max(STAMINA_VALUES[0], state.stamina - 1))))
        choices.append((0.04, State(state.health, state.arrows,
                                    max(STAMINA_VALUES[0], state.stamina - 2))))
        choices.append((0.16, State(state.health, min(
            ARROWS_VALUES[-1], state.arrows+1), max(STAMINA_VALUES[0], state.stamina - 2))))

        cost = 0
        for choice in choices:
            cost += choice[0] * (costs[ACTION_DODGE] + REWARD[choice[1].show()])

        return cost, choices

def show(i, utilities, policies, path):
    with open(path, 'a+') as f:
        f.write('iteration={}\n'.format(i))
        utilities = np.around(utilities, 3)
        for state, util in np.ndenumerate(utilities):
            state = State(*state)
            if state.health == 0:
                f.write('{}:-1=[{:.3f}]\n'.format(state, util))
                continue

            if policies[state.show()] == ACTION_SHOOT:
                act_str = 'SHOOT'
            elif policies[state.show()] == ACTION_DODGE:
                act_str = 'DODGE'
            elif policies[state.show()] == ACTION_RECHARGE:
                act_str = 'RECHARGE'

            f.write('{}:{}=[{:.3f}]\n'.format(state, act_str, util))
        f.write('\n\n')

def value_iteration(delta_inp, gamma_inp, costs_inp, path):
    utilities = np.zeros((HEALTH_RANGE, ARROWS_RANGE, STAMINA_RANGE))
    policies = np.full((HEALTH_RANGE, ARROWS_RANGE,
                        STAMINA_RANGE), -1, dtype='int')

    index = 0
    done = False
    while not done:  # one iteration of value iteration
        #print(index)
        temp = np.zeros(utilities.shape)
        delta = np.NINF

        for state, util in np.ndenumerate(utilities):
            if state[0] == 0:
                continue
            new_util = np.NINF
            for act_index in range(NUM_ACTIONS):
                cost, states = action(act_index, state,costs_inp)

                if cost is None:
                    continue

                expected_util = reduce(
                    add, map(lambda x: x[0]*utilities[x[1].show()], states))
                new_util = max(new_util, cost + gamma_inp * expected_util)

            temp[state] = new_util
            delta = max(delta, abs(util - new_util))

        utilities = deepcopy(temp)

        for state, _ in np.ndenumerate(utilities):
            if state[0] == 0:
                continue
            best_util = np.NINF
            best_action = None

            for act_index in range(NUM_ACTIONS):
                cost, states = action(act_index, state, costs_inp )

                if states is None:
                    continue

                action_util = cost + gamma_inp * \
                    reduce(
                        add, map(lambda x: x[0]*utilities[x[1].show()], states))

                if action_util > best_util:
                    best_action = act_index
                    best_util = action_util

            policies[state] = best_action

        show(index, utilities, policies, path)
        index += 1
        if delta < delta_inp:
            done = True
    return index

# PREP
os.makedirs('outputs', exist_ok=True)

# TASK 1
path = 'outputs/task_1_trace.txt'
value_iteration(DELTA, GAMMA, (COST,COST,COST), path)

# TASK 2 PART 1
path = 'outputs/task_2_part_1_trace.txt' 
COST = -2.5
SHOOT_COST = -0.25
value_iteration(DELTA, GAMMA, (SHOOT_COST, COST, COST), path)

# TASK 2 PART 2
COST = -2.5
GAMMA = 0.1
path = 'outputs/task_2_part_2_trace.txt'
value_iteration(DELTA, GAMMA, (COST, COST, COST), path)

# Task 2 part 3
COST = -2.5
GAMMA = 0.1
DELTA = 1e-10
path = 'outputs/task_2_part_3_trace.txt'
value_iteration(DELTA, GAMMA, (COST,COST,COST), path)
