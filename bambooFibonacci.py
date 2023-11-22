task_id = 4
growth_rate = [55, 34, 21, 13, 8, 5, 3, 2, 1, 1]

# Don't change anything above this line
# =====================================

# generate your solution as a list
queue = [(0, 1), (0, 2), (1, 0), (0, 3), 
(1, 2), (0, 4), (1, 0), (2, 0),
(1, 3), (0, 5), (1, 2), (0, 4),
(1, 0), (3, 2), (0, 1), (0, 6),
(1, 2), (0, 3), (1, 4), (0, 2),
(1, 0), (5, 0), (1, 3), (0, 2),
(1, 0), (4, 0), (1, 2), (0, 3),
(1, 7), (0, 2), (1, 0), (0, 3),
(1, 2), (0, 4), (1, 5), (0, 2),
(1, 3), (0, 6), (1, 2), (0, 4),
(1, 0), (3, 2), (0, 1), (0, 5),
(1, 2), (0, 3), (1, 4), (0, 2),
(1, 0), (0, 3), (1, 2), (0, 8),
(1, 0), (2, 4), (0, 1), (3, 5),
(0, 1), (2, 6), (0, 1), (7, 9)]

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
