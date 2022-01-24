import numpy as np
from itertools import combinations


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