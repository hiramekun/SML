import copy

import numpy as np


class State:
    def __init__(self, field, cost, times):
        self.field = field
        self.cost = cost
        self.times = times


g_hat = [0]
g_now = 0
states = []
N_EMPTY = -1
field_now = np.array([[4, 3, 5],
                      [2, 1, N_EMPTY]])
field_correct = np.array([[1, 2, 3],
                          [4, 5, N_EMPTY]])


def add_or_update(g):
    idx = get_past_state_idx()
    cost_h = calc_h_hat()
    if idx >= 0:
        cost_g = min(g, states[idx].times)
        g_now = cost_g
        cost = cost_g + cost_h
        if states[idx].cost > cost:
            update_state(idx, cost, cost_g)
            return True
        else:
            return False
    else:
        cost = g + cost_h
        times = g
        add_state(cost, times)
        return True


def add_state(cost, times):
    new_state = State(copy.deepcopy(field_now), cost, times)
    states.append(new_state)


def update_state(idx, cost, times):
    new_state = State(copy.deepcopy(field_now), cost, times)
    states[idx] = new_state


def get_past_state_idx():
    for i in range(len(states)):
        if (states[i].field == field_now).all():
            return i
    return -1


def move(x_empty, y_empty, g_now):
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    for i in range(4):
        nx = x_empty + dx[i]
        ny = y_empty + dy[i]
        if 0 <= nx < 2 and 0 <= ny < 3:
            field_now[x_empty, y_empty] = field_now[nx, ny]
            field_now[nx, ny] = N_EMPTY
            g_now = g_now + 1
            if add_or_update(g_now):
                print(field_now)
                print('========')
                break
            else:
                field_now[nx, ny] = field_now[x_empty, y_empty]
                field_now[x_empty, y_empty] = N_EMPTY


def calc_h_hat():
    ret = 0
    for i in range(2):
        for j in range(3):
            if field_now[i, j] != field_correct[i, j]:
                ret += 1
    return ret


if __name__ == '__main__':
    while not (field_now == field_correct).all():
        for i in range(2):
            for j in range(3):
                if field_now[i, j] == -1:
                    move(i, j, g_now)
                    g_now += 1
