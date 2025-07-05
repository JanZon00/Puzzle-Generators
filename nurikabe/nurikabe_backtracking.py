'''
1. Generating an empty grid.
2. Randomly selecting a cell that touches the edge of the puzzle and marking it as the starting point.
3. Running a DFS algorithm to generate water around the islands, ensuring that no 2x2 blocks of water are created.
The probability of filling a cell with water is controlled by the FILLING variable.
The higher the FILLING value, the more water is placed in the puzzle, which generally makes it easier.
4. Identifying the islands, calculating their sizes, storing the size in one of the island's cells, and clearing the rest.
5. Using a solver to check for a valid solution. If the puzzle is valid but unsolvable, a new one is generated.
'''
import random
import subprocess
import matplotlib.pyplot as plt
import os

PUZZLE_SIZE = 7
FILLING = 0.7 #The lower the number, the harder is nurikabe. At certain point, usually lower than 0.3 it's hard or impossible to generate valid nurikabe.
PUZZLE_FILE_PATH = "nurikabe_backtracking.csv"
PUZZLE_IMAGE_PATH = "nurikabe.png"

def is_edge(x, y):
    '''Checks whether the cell is on the edge of the puzzle, that means if it touches any of the walls.'''
    return x == 0 or y == 0 or x == PUZZLE_SIZE - 1 or y == PUZZLE_SIZE - 1

def is_valid(x, y):
    '''Checks if the cell coordinates are inside the puzzle'''
    return 0 <= x < PUZZLE_SIZE and 0 <= y < PUZZLE_SIZE

def has_2x2_block(grid):
    '''Checks if there is a 2x2 block consisting of water elements'''
    count = 0
    for i in range(PUZZLE_SIZE - 1):
        for j in range(PUZZLE_SIZE - 1):
            if all(grid[i + dx][j + dy] == '.' for dx in [0, 1] for dy in [0, 1]):
                count += 1
    return count

def dfs(grid, x, y, visited):
    '''Creates a random path of elements serving as water in the puzzle. Checks that no 2x2 blocks are formed from these elements.'''
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    random.shuffle(directions)

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if is_valid(nx, ny) and (nx, ny) not in visited:
            if random.random() < FILLING:
                grid[nx][ny] = '.'
                if has_2x2_block(grid) >= 1:
                    grid[nx][ny] = 0
                    continue
                visited.add((nx, ny))
                dfs(grid, nx, ny, visited)

def find_islands(grid):
    '''Creates a list of available islands'''
    visited = set()
    islands = []

    def dfs_island(x, y, island):
        if not is_valid(x, y) or (x, y) in visited or grid[x][y] != 0:
            return
        visited.add((x, y))
        island.append((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            dfs_island(x + dx, y + dy, island)

    for i in range(PUZZLE_SIZE):
        for j in range(PUZZLE_SIZE):
            if grid[i][j] == 0 and (i, j) not in visited:
                island = []
                dfs_island(i, j, island)
                islands.append(island)

    return islands

def assign_island_numbers(grid, islands):
    '''Assigns the island its size in a random cell and removes the other digits from that island'''
    for island in islands:
        size_island = len(island)
        chosen = random.choice(island)
        for x, y in island:
            grid[x][y] = size_island if (x, y) == chosen else '.'

def generate_nurikabe():
    grid = [[0 for _ in range(PUZZLE_SIZE)] for _ in range(PUZZLE_SIZE)]
    edge_cells = [(i, j) for i in range(PUZZLE_SIZE) for j in range(PUZZLE_SIZE) if is_edge(i, j)]
    start = random.choice(edge_cells)
    x, y = start
    grid[x][y] = '.'
    visited = set()
    visited.add((x, y))
    dfs(grid, x, y, visited)
    return grid

def print_grid(grid):
    for row in grid:
        print(" ".join(str(cell) for cell in row))

def save_to_file(board):
    with open(PUZZLE_FILE_PATH, 'w') as f:
        for row in board:
            f.write(" ".join(str(cell) for cell in row) + "\n")
    
    _, ax = plt.subplots(figsize=(PUZZLE_SIZE, PUZZLE_SIZE))
    ax.set_xlim(0, PUZZLE_SIZE)
    ax.set_ylim(0, PUZZLE_SIZE)
    ax.axis('off')

    for i in range(PUZZLE_SIZE + 1):
        ax.axhline(i, color='black', lw=1)
        ax.axvline(i, color='black', lw=1)

    for i in range(PUZZLE_SIZE):
        for j in range(PUZZLE_SIZE):
            val = board[i][j]
            if val != '.':
                ax.text(j + 0.5, PUZZLE_SIZE - 0.5 - i, str(val), va='center', ha='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(PUZZLE_IMAGE_PATH, dpi=300)
    print(f"Found a solution, Nurikabe puzzle saved to file: {PUZZLE_FILE_PATH}")
    print(f"Nurikabe image saved to {PUZZLE_IMAGE_PATH}")
    plt.close()

def run_solver():
    DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    NURIKABE_PATH = os.path.join(DIR_PATH, "nurisolver.py")
    
    command = f"python {NURIKABE_PATH} {PUZZLE_FILE_PATH}"
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def generate_puzzle():
    while True:
        nurikabe = generate_nurikabe()
        islands = find_islands(nurikabe)
        print_grid(nurikabe)
        assign_island_numbers(nurikabe, islands)
        save_to_file(nurikabe)

        if run_solver():
            break
        else:
            print("Not unique, regenerating...")

generate_puzzle()
