from constants import Direction
import math

def is_opposite_direction(current: Direction, new: Direction) -> bool:
    match current:
        case Direction.LEFT:
            return new == Direction.RIGHT
        case Direction.UP:
            return new == Direction.DOWN
        case Direction.RIGHT:
            return new == Direction.LEFT
        case Direction.DOWN:
            return new == Direction.UP
        
def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
