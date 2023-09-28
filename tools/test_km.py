import numpy
import torch
from munkres import Munkres, print_matrix

m = Munkres()
sim_scores = torch.rand(4,6).cuda()
print(sim_scores)
indexes = m.compute(-sim_scores)

print_matrix(sim_scores, msg='Lowest cost through this matrix:')
total = 0
for row, column in indexes:
    value = sim_scores[row][column]
    total += value
    print(f'({row}, {column}) -> {value}')
print(f'total cost: {total}')
