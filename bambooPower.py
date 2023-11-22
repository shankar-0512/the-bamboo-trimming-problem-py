task_id = 3
growth_rate = [96, 54, 54, 48, 24, 18, 18, 12, 6, 6, 6, 3, 3, 2, 2, 2]

# Don't change anything above this line
# =====================================

# generate your solution as a list
queue = [(0, 1), (3, 4), (0, 2), 
(1, 7), (0, 6), (3, 2), (0, 1), (10, 4), (0, 2), (3, 1), (0, 5), 
(2, 8), (0, 1), (3, 6), (0, 2), (4, 1), (0, 3), (2, 7), (0, 1), 
(5, 3), (0, 2), (1, 9), (0, 4), (3, 2), (0, 6), (1, 12), (0, 2), 
(3, 13), (0, 1), (2, 5), (0, 4), (3, 1), (0, 2), (7, 6), (0, 1), 
(3, 2), (0, 10), (1, 4), (0, 2), (3, 5), (0, 1), (2, 8), (0, 6), 
(3, 1), (0, 2), (4, 11), (0, 1), (3, 2), (0, 7), (1, 5), (0, 2), 
(3, 9), (0, 1), (6, 4), (2, 0), (3, 1), (0, 14), (2, 5), (0, 1), 
(3, 4), (0, 2), (1, 7), (0, 6), (3, 2), (0, 1), (10, 4), (0, 2), 
(3, 1), (0, 5), (2, 8), (0, 1), (3, 6), (0, 2), (4, 1), (0, 3), 
(2, 7), (0, 1), (5, 3), (0, 2), (1, 9), (0, 4), (3, 2), (0, 6), 
(1, 12), (0, 2), (3, 15), (0, 1), (2, 5), (0, 4), (3, 1), (0, 2), 
(7, 6), (0, 1), (3, 2), (0, 10), (1, 4), (0, 2), (3, 5), (0, 1), 
(2, 8), (0, 6), (3, 1), (0, 2), (4, 11), (0, 1), (3, 2), (0, 7), 
(1, 5), (0, 2), (3, 9), (0, 1), (6, 4), (2, 0), (3, 1), (0, 13), 
(2, 5), (0, 1), (3, 4), (0, 2), (1, 7), (0, 6), (3, 2), (0, 1), 
(10, 4), (0, 2), (3, 1), (0, 5), (2, 8), (0, 1), (3, 6), (0, 2), 
(4, 1), (0, 3), (2, 7), (0, 1), (5, 3), (0, 2), (1, 9), (0, 4), 
(3, 2), (0, 6), (1, 12), (0, 2), (3, 14), (0, 1), (2, 5), (0, 4), 
(3, 1), (0, 2), (7, 6), (0, 1), (3, 2), (0, 10), (1, 4), (0, 2), 
(3, 5), (0, 1), (2, 8), (0, 6), (3, 1), (0, 2), (4, 11), (0, 1), 
(3, 2), (0, 7), (1, 5), (0, 2), (3, 9), (0, 1), (6, 4), (2, 0), 
(3, 1), (0, 15), (2, 5)]

# =====================================
# Don't change anything below this line

from collections import deque

solution = deque()
# add each element to the solution
for i in queue:
    solution.append(i)

import bamboo

# records your solution
bamboo.calculate_height(growth_rate, solution, task_id)