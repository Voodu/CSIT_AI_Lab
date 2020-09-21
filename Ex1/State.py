from functools import total_ordering

@total_ordering
class State:
    def __init__(self, path, neighbors):
        self.path = path
        self.neighbors = neighbors
    
    def __repr__(self):
        return f"{self.path}, {self.neighbors}"

    def __eq__(self, o):
        # if not self._is_valid_operand(o):
        #     return NotImplemented
        return self.path == o.path

    def __lt__(self, o):
        # if not self._is_valid_operand(o):
        #     return NotImplemented
        return self.path < o.path