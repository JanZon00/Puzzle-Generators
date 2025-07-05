'''
1. Creating a closed and valid loop made up of edges connecting the grid's vertices. The algorithm searches for a random path. If no further moves are possible,
it backtracks and tries a different path until it returns to the starting point and closes the loop.
2. All cells in the puzzle are filled with numbers indicating how many edges of the loop touch that cell.
3. An empty grid is then created, and clues from the solved puzzle are added one by one.
There are two difficulty levels: easy and hard. In both, a specified percentage of clues is added to the empty grid, and the solver is run.
The 'hard' mode additionally controls the number of zeros and threes based on specified percentages. The result is saved to a file.
If no solution exists, additional clues from the clues list are added until the solver finds a valid solution.
'''
import random
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os

PUZZLE_SIZE = 6
LOOP_START_POINT = (0, 0)
CLUES_PERCENTAGE = 0.6
MAX_ZERO_PERCENT = 0.04 # maximum percentage of clues with value 0, the fewer, the harder the puzzle
MAX_THREE_PERCENT = 0.12 # maximum percentage of clues with value 3; the fewer, the harder the puzzle
DIFFICULTY = "easy"
PUZZLE_FILE_PATH = "slitherlink_backtracking.csv"
PUZZLE_IMAGE_PATH = "slitherlink.png"

class SlitherlinkBoard:
    def __init__(self):
        self.rows = PUZZLE_SIZE
        self.cols = PUZZLE_SIZE

        self.cells = [[None for _ in range(PUZZLE_SIZE)] for _ in range(PUZZLE_SIZE)]
        self.h_edges = [['x' for _ in range(PUZZLE_SIZE)] for _ in range(PUZZLE_SIZE + 1)]
        self.v_edges = [['x' for _ in range(PUZZLE_SIZE + 1)] for _ in range(PUZZLE_SIZE)]
        self.vertex_degree = {}

    def set_h_edge(self, row, col, value):
        '''Set the horizontal edge and update the degree of vertices connected to this edge'''
        self.h_edges[row][col] = value
        self.update_vertex_degree((row, col), value == '-')
        self.update_vertex_degree((row, col + 1), value == '-')

    def set_v_edge(self, row, col, value):
        '''Set the vertical edge and update the degree of vertices connected to this edge'''
        self.v_edges[row][col] = value
        self.update_vertex_degree((row, col), value == '|')
        self.update_vertex_degree((row + 1, col), value == '|')

    def update_vertex_degree(self, point, add):
        '''Update the vertex degree by increasing or decreasing it depending on add'''
        if point not in self.vertex_degree:
            self.vertex_degree[point] = 0
        self.vertex_degree[point] += 1 if add else -1

    def get_edge_value(self, point1, point2):
        '''Returns the value of the edge between two points. Returns X, |, or -'''
        row1, col1 = point1
        row2, col2 = point2
        if row1 == row2:
            col = min(col1, col2)
            return self.h_edges[row1][col]
        else:
            row = min(row1, row2)
            return self.v_edges[row][col1]

    def set_edge(self, point1, point2, value):
        '''Sets a vertical or horizontal edge'''
        row1, col1 = point1
        row2, col2 = point2
        if row1 == row2:
            self.set_h_edge(row1, min(col1, col2), value)
        else:
            self.set_v_edge(min(row1, row2), col1, value)

    def available_neighbors(self, point):
        '''Returns a list of available neighbors for a given point'''
        row, col = point
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                if self.get_edge_value(point, (new_row, new_col)) == 'x':
                    if self.vertex_degree.get(point, 0) < 2 and self.vertex_degree.get((new_row, new_col), 0) < 2:
                        neighbors.append((new_row, new_col))
        return neighbors

    def display_puzzle(self):
        for row in range(self.rows * 2 + 1):
            line = ''
            if row % 2 == 0:
                row_idx = row // 2
                for col in range(self.cols):
                    line += '. ' + self.h_edges[row_idx][col] + ' '
                line += '.'
            else:
                row_idx = row // 2
                for col in range(self.cols + 1):
                    line += self.v_edges[row_idx][col]
                    if col < self.cols:
                        cell_value = self.cells[row_idx][col]
                        line += f' {cell_value if cell_value is not None else " "} '
            print(line)

    def generate_loop(self):
        '''
        Creates an empty and closed Slitherlink loop. Uses backtracking to backtrack when there is no possibility to make the next move.
        Ends when the loop is successfully closed.
        '''
        start = LOOP_START_POINT
        path = [start]

        def backtrack(current, path):
            if len(path) > 4 and current == start:
                return True

            neighbors = self.available_neighbors(current)
            random.shuffle(neighbors)

            for nb in neighbors:
                if nb == start and len(path) > 4:
                    self.set_edge(current, nb, '-' if current[0] == nb[0] else '|')
                    path.append(nb)
                    return True
                if nb in path:
                    continue
                self.set_edge(current, nb, '-' if current[0] == nb[0] else '|')
                path.append(nb)
                if backtrack(nb, path):
                    return True

                self.set_edge(current, nb, 'x')
                path.pop()
            return False

        backtrack(start, path)

        
    def calculate_cell_clues(self):
        '''Returns a list of numbers indicating how many edges touch each cell'''
        clues = []
        for row in range(self.rows):
            for col in range(self.cols):
                count = 0
                if self.h_edges[row][col] == '-':
                    count += 1
                if self.h_edges[row + 1][col] == '-':
                    count += 1
                if self.v_edges[row][col] == '|':
                    count += 1
                if self.v_edges[row][col + 1] == '|':
                    count += 1
                clues.append(count)
                self.cells[row][col] = count
        return clues

def save_to_file(board):
    with open(PUZZLE_FILE_PATH, 'w', newline='') as f:
        f.write(f"{PUZZLE_SIZE} {PUZZLE_SIZE}\n")
        for row in board:
            row_to_save = ''.join(' ' if num == -1 else str(num) for num in row)
            f.write(row_to_save + '\n')
    
    _, ax = plt.subplots(figsize=(PUZZLE_SIZE, PUZZLE_SIZE))
    ax.set_xlim(-0.2, PUZZLE_SIZE + 0.2)
    ax.set_ylim(-0.2, PUZZLE_SIZE + 0.2)
    ax.set_aspect('equal')
    ax.axis('off')

    for i in range(PUZZLE_SIZE + 1):
        for j in range(PUZZLE_SIZE + 1):
            ax.plot(j, PUZZLE_SIZE - i, 'ko', markersize=3)

    for i in range(PUZZLE_SIZE):
        for j in range(PUZZLE_SIZE):
            val = board[i][j]
            if val != -1:
                ax.text(j + 0.5, PUZZLE_SIZE - i - 0.5, str(val), ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(PUZZLE_IMAGE_PATH, dpi=300)
    print(f"Found unique solution, slitherlink puzzle saved to file: {PUZZLE_FILE_PATH}")
    print(f"Slitherlink image saved to {PUZZLE_IMAGE_PATH}")    
    plt.close()

def run_solver():
    DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    NURISOLVER_PATH = os.path.join(DIR_PATH, "slsolve.py")
    
    command = f"python {NURISOLVER_PATH} {PUZZLE_FILE_PATH}"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if "Couldn't solve puzzle." in result.stdout or "Couldn't solve puzzle." in result.stderr:
            print("Error: The puzzle is unsolvable.")
            return False
        
        print("Solver solved the puzzle.")
        return True

    except subprocess.CalledProcessError:
        return False

def generate_clues(clues):
    '''
    Adds the specified percentage of clues to an empty board, then runs the solver.
    In 'hard' mode, controls the number of zeros and threes according to specified percentages.
    '''
    total_cells = PUZZLE_SIZE * PUZZLE_SIZE
    target_clues = int(total_cells * CLUES_PERCENTAGE)

    board = np.full((PUZZLE_SIZE, PUZZLE_SIZE), -1, dtype=int)

    if DIFFICULTY == 'hard':
        clues_by_value = {0: [], 1: [], 2: [], 3: []}
        for i, clue in enumerate(clues):
            if clue in clues_by_value:
                clues_by_value[clue].append(i)

        max_threes = int(target_clues * MAX_THREE_PERCENT)
        max_zeros = int(target_clues * MAX_ZERO_PERCENT)

        selected_indices = []

        random.shuffle(clues_by_value[3])
        selected_indices += clues_by_value[3][:max_threes]

        random.shuffle(clues_by_value[0])
        selected_indices += clues_by_value[0][:max_zeros]

        remaining = target_clues - len(selected_indices)
        mixed_1_2 = clues_by_value[1] + clues_by_value[2]
        random.shuffle(mixed_1_2)
        selected_indices += mixed_1_2[:remaining]

        if len(selected_indices) < target_clues:
            remaining = target_clues - len(selected_indices)
            additional_zeros = list(set(clues_by_value[0]) - set(selected_indices))
            random.shuffle(additional_zeros)
            selected_indices += additional_zeros[:remaining]

        clue_positions = [(i, clues[i]) for i in selected_indices]
        random.shuffle(clue_positions)

    else:
        clue_positions = [(i, clue) for i, clue in enumerate(clues) if clue >= 0]
        random.shuffle(clue_positions)

    added = 0
    for i in range(min(target_clues, len(clue_positions))):
        index, clue = clue_positions[i]
        row, col = divmod(index, PUZZLE_SIZE)
        board[row][col] = clue
        added += 1

    save_to_file(board)
    if run_solver():
        print(f"Found a solvable puzzle ({DIFFICULTY}) with the initial percentage of clues.")
        return True

    while added < len(clue_positions):
        index, clue = clue_positions[added]
        row, col = divmod(index, PUZZLE_SIZE)
        board[row][col] = clue
        print(f"Adding another clue: {clue} at position ({row}, {col}) (index {index})")
        added += 1

        save_to_file(board)
        if run_solver():
            print("Found a solvable puzzle after adding more clues.")
            return True

    print(f"Failed to generate a solvable puzzle ({DIFFICULTY}).")
    return False

def generate_slitherlink():
    while True:
        board = SlitherlinkBoard()
        board.generate_loop()
        clues = board.calculate_cell_clues()
        board.display_puzzle()
        if generate_clues(clues):
            break
        else:
            print("Failed to find a solvable puzzle, retrying...\n")

generate_slitherlink()
