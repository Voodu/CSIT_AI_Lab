import numpy as np

class Node:
    __node_count = 0

    def __init__(self):
        self.id = Node.__node_count
        self.x, self.y = np.random.randint(-100, 101, 2)
        self.z = np.random.randint(0, 51)
        Node.__node_count += 1

    def __str__(self):
        return f"[{self.id}, ({self.x}, {self.y}, {self.z})]"

    def __distance(self, x_diff, y_diff, z_diff):
        diff = np.array([x_diff, y_diff, z_diff])
        return np.sqrt(np.sum(diff**2))

    def distance(self, other):
        diff = [self.x-other.x, self.y-other.y, self.z-other.z]
        return self.__distance(*diff)

    def distance_directed(self, other):
        diff = [self.x-other.x, self.y-other.y, self.z-other.z]
        if self.z - other.z > 0:  # going down
            diff[2] *= 0.9
        elif self.z - other.z < 0:  # going up
            diff[2] *= 1.1
        return self.__distance(*diff)