'''
1. Creating an empty grid.
2. The grid is randomly filled with paths of length 2. If the puzzle has an odd number of cells, one of the paths will have a length of 3.
3. Two adjacent paths are randomly merged into a single longer path if possible.
4. If the algorithm reaches a satisfactory number of paths equal to MAX_PATHS, a valid puzzle is returned. If the required number of paths is not reached 
and no further merging is possible, the algorithm uses backtracking to undo previous steps and attempt alternative path combinations.
5. The solver is run, and the completed puzzle is saved to a file.
'''
import random
import time
import matplotlib.pyplot as plt
from NumberlinkPuzzle import NumberlinkPuzzle

PUZZLE_SIZE = 6
MAX_PATHS = 6
PUZZLE_FILE_PATH = "numberlink_backtracking.csv"
PUZZLE_IMAGE_PATH = "numberlink.png"

def int_to_letter(n):
    letters = ""
    while n >= 0:
        letters = chr(n % 26 + ord('A')) + letters
        n = n // 26 - 1
        if n < 0:
            break
    return letters

def save_to_file(board):
    unique_values = sorted({cell for row in board for cell in row if cell > 0})
    letters = [int_to_letter(i) for i in range(len(unique_values))]
    random_letters = random.sample(letters, len(unique_values))
    mapping = dict(zip(unique_values, random_letters))

    with open(PUZZLE_FILE_PATH, 'w') as f:
        f.write(f"{PUZZLE_SIZE} {PUZZLE_SIZE}\n")
        for row in board:
            line = ''
            for cell in row:
                line += '.' if cell == 0 else str(mapping[cell])
            f.write(line + '\n')
            

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
            if val != 0:
                letter = mapping[val]
                ax.text(j + 0.5, PUZZLE_SIZE - 0.5 - i, letter, ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(PUZZLE_IMAGE_PATH, dpi=300)
    print(f"Numberlink puzzle saved to file: {PUZZLE_FILE_PATH}")
    print(f"Numberlink image saved to {PUZZLE_IMAGE_PATH}")
    plt.close()

def parse_clues():
    with open(PUZZLE_FILE_PATH, 'r') as f:
        f.readline().strip()
        board = [f.readline().rstrip('\n') for _ in range(PUZZLE_SIZE)]
    positions = {}
    
    for row_idx, row in enumerate(board):
        for col_idx, letter in enumerate(row):
            if letter.isalpha():
                if letter not in positions:
                    positions[letter] = []
                positions[letter].append((row_idx, col_idx))
    
    wynik = []
    for letter in sorted(positions.keys()):
        wynik.extend(positions[letter])
    
    return wynik

def run_solver():
    print("solving...")
    clues_locations = parse_clues()
    puzzle = NumberlinkPuzzle(PUZZLE_SIZE, PUZZLE_SIZE, clues_locations)
    cnf = puzzle.generate_cnf()
    result = puzzle.solve(cnf)
    
    if result is None:
        print("No solution found.")
    else:
        print("Found solution")
        return True

def create_board():
    return [[0 for _ in range(PUZZLE_SIZE)] for _ in range(PUZZLE_SIZE)]

def count_neighbors_with_value(board, i, j, value):
    '''Counts neighbors containing path elements'''
    count = 0
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < PUZZLE_SIZE and 0 <= nj < PUZZLE_SIZE:
            if board[ni][nj] == value:
                count += 1
    return count

def path_has_extra_neighbors(board, path, value):
    '''Checks if a given path element has more than two neighbors'''
    for i, j in path:
        if count_neighbors_with_value(board, i, j, value) > 2:
            return True
    return False

def find_path_ends(board, value):
    '''Finds the ends of the path'''
    ends = []
    for i in range(PUZZLE_SIZE):
        for j in range(PUZZLE_SIZE):
            if board[i][j] == value and count_neighbors_with_value(board, i, j, value) == 1:
                ends.append((i, j))
    return ends

def get_path_cells(board, value):
    '''Returns all cells belonging to the path with the given value'''
    return [(i, j) for i in range(PUZZLE_SIZE) for j in range(PUZZLE_SIZE) if board[i][j] == value]

def has_2x2_block(board, value):
    '''Checks if there are 2x2 blocks with the same elements'''
    for i in range(PUZZLE_SIZE - 1):
        for j in range(PUZZLE_SIZE - 1):
            if (board[i][j] == value and
                board[i+1][j] == value and
                board[i][j+1] == value and
                board[i+1][j+1] == value):
                return True
    return False

def filter_path_ends_only(board):
    '''Removes path elements leaving only the start and end'''
    new_board = create_board()
    for i in range(PUZZLE_SIZE):
        for j in range(PUZZLE_SIZE):
            value = board[i][j]
            if value > 0 and count_neighbors_with_value(board, i, j, value) == 1:
                new_board[i][j] = value
                
    unique_values = sorted({cell for row in board for cell in row if cell > 0})
    value_mapping = {value: idx+1 for idx, value in enumerate(unique_values)}
    new_board = [
        [value_mapping[cell] if cell in value_mapping else 0 for cell in row]
        for row in new_board
    ]
    
    return new_board

def merge_paths(board, MAX_PATHS, start_time=None, time_limit=120.0):
    '''
    Merges two random paths into one longer path, repeats as long as possible.
    If the path limit is not reached, uses backtracking with a time limit (stops after 120 seconds).
    '''
    if start_time is None:
        start_time = time.time()
    elif time.time() - start_time > time_limit:
        return False, None

    current_paths = sorted({cell for row in board for cell in row if cell > 0})
    if len(current_paths) <= MAX_PATHS:
        return True, board

    values = sorted(current_paths)
    random.shuffle(values)

    for v1 in values:
        ends1 = find_path_ends(board, v1)
        random.shuffle(ends1)

        for i1, j1 in ends1:
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i1 + di, j1 + dj
                if 0 <= ni < PUZZLE_SIZE and 0 <= nj < PUZZLE_SIZE:
                    v2 = board[ni][nj]
                    if v2 > 0 and v2 != v1:
                        old_v2_cells = get_path_cells(board, v2)
                        for i, j in old_v2_cells:
                            board[i][j] = v1

                        merged_path = get_path_cells(board, v1)

                        if not path_has_extra_neighbors(board, merged_path, v1) and not has_2x2_block(board, v1):
                            success, new_board = merge_paths(board, MAX_PATHS, start_time, time_limit)
                            if success:
                                return True, new_board

                        for i, j in old_v2_cells:
                            board[i][j] = v2
    return False, None

def divide_board():
    '''Divides the board into paths of length 2, or 2 and 3 if the total number of cells is odd'''
    board = create_board()
    current_value = 1
    i, j = 0, 0

    while i < PUZZLE_SIZE:
        while j < PUZZLE_SIZE:
            if board[i][j] != 0:
                j += 1
                continue

            directions = [(0, 1), (1, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < PUZZLE_SIZE and 0 <= nj < PUZZLE_SIZE:
                    if board[ni][nj] == 0:
                        board[i][j] = current_value
                        board[ni][nj] = current_value
                        current_value += 1
                        break
            j += 1
        i += 1
        j = 0

    for i in range(PUZZLE_SIZE):
        for j in range(PUZZLE_SIZE):
            if board[i][j] == 0:
                for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < PUZZLE_SIZE and 0 <= nj < PUZZLE_SIZE and board[ni][nj] > 0:
                        board[i][j] = board[ni][nj]
                        break
    return board

def generate_numberlink():
    while True:
        board = divide_board()   
        success, new_board = merge_paths(board, MAX_PATHS)
        if success:
            board = new_board
        else:
            print("Failed to reach the path limit")
        board = filter_path_ends_only(board)
        save_to_file(board)
        if run_solver():
            break
        else:
            print("Not unique, regenerating...")

generate_numberlink()
