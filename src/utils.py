from constants import Direction

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
