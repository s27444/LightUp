from typing import List, Tuple, Dict
import copy

class Board:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
    
    @classmethod
    def from_file(cls, filename: str) -> "Board":
        with open(filename, "r") as f:
            lines = f.readlines()
        
        dims = lines[0].strip().split()
        if len(dims) != 2:
            raise ValueError("First line must contain width and height")
        
        width, height = int(dims[0]), int(dims[1])
        board = cls(width, height)
        
        lines = [line.strip() for line in lines[1:height+1]]
        
        for y, line in enumerate(lines):
            if len(line) != width:
                raise ValueError(f"Line {y+1} has incorrect length. Expected {width}, got {len(line)}")
            
            for x, cell in enumerate(line):
                if cell == '.':
                    board.grid[y][x] = 0
                elif cell == '#' or cell == '-1':
                    board.grid[y][x] = -1
                elif cell in '1234':
                    board.grid[y][x] = int(cell)
                elif cell == '0':
                    board.grid[y][x] = -2
                else:
                    raise ValueError(f"Invalid cell value at position ({x}, {y}): {cell}")
        
        return board
    
    def to_file(self, filename: str) -> None:
        with open(filename, "w") as f:
            f.write(f"{self.width} {self.height}\n")
            for row in self.grid:
                line = ""
                for cell in row:
                    if cell == 0:
                        line += "."
                    elif cell == -1:
                        line += "#"
                    elif cell == -2:
                        line += "0"
                    else:
                        line += str(cell)
                f.write(line + "\n")
    
    def __str__(self) -> str:
        result = []
        for row in self.grid:
            line = ""
            for cell in row:
                if cell == 0:
                    line += "."
                elif cell == -1:
                    line += "#"
                elif cell == -2:
                    line += "0"
                else:
                    line += str(cell)
            result.append(line)
        return "\n".join(result)
    
    def get_white_cells(self) -> List[Tuple[int, int]]:
        result = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 0:
                    result.append((x, y))
        return result
    
    def get_black_cells(self) -> Dict[Tuple[int, int], int]:
        result = {}
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] > 0:
                    result[(x, y)] = self.grid[y][x]
                elif self.grid[y][x] == -2:
                    result[(x, y)] = 0
        return result
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append((nx, ny))
        return neighbors
    
    def is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_white_cell(self, x: int, y: int) -> bool:
        return self.is_valid_position(x, y) and self.grid[y][x] == 0
    
    def is_black_cell(self, x: int, y: int) -> bool:
        return self.is_valid_position(x, y) and (self.grid[y][x] == -1 or self.grid[y][x] == -2 or self.grid[y][x] > 0)
    
    def copy(self) -> "Board":
        new_board = Board(self.width, self.height)
        new_board.grid = copy.deepcopy(self.grid)
        return new_board