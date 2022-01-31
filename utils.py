import numpy as np
from itertools import combinations


def check_outcome(state, action, shape, row_size):
    row = None
    next_state = state.copy()

    # check if player made an illegal movement
    value = state.item(action)
    if value != 0:
        return "mistake", next_state, row

    # update board
    next_state[action] = 1

    # identify coordinates with player moves
    grid = np.reshape(next_state, shape)
    coords = list(zip(*np.where(grid == 1)))

    # check win
    win, row = check_collinearity(coords, row_size)
    if win:
        return "win", next_state, row

    # check if there is a withdraw
    if len(np.where(next_state == 0)[0]) == 0:
        return "withdraw", next_state, row

    return "", next_state, row


def check_collinearity(coords, row_size):
    row = None

    # create combinations of points
    candidate_rows = list(combinations(coords, row_size))
    for candidate_row in candidate_rows:
        # create vectors based on points
        vectors = []
        previous = None
        for i, coord in enumerate(candidate_row):
            if i != 0:
                vector = np.array(coord) - previous
                vectors.append(vector)
            previous = np.array(coord)
        det = np.linalg.det(vectors)
        if det == 0:
            row = candidate_row
            return True, row

    return False, row


def reduce_row(row):
    candidate_rows = list(combinations(row, 2))
    max_row = None
    max_dist = 0
    for row in candidate_rows:
        dist = np.linalg.norm(np.array(row[1])-np.array(row[0]))
        if dist > max_dist:
            max_dist = dist
            max_row = row
    return max_row