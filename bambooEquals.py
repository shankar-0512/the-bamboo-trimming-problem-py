task_id = 0
growth_rate = [10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5]

# Don't change anything above this line
# =====================================

# generate your solution as a list
queue = [(0, 4), (1, 5), (2, 6), (3, 7), (0, 8), (1, 9), (2, 10), (3, 11)]

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
