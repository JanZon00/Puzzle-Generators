'''
1. Generating an empty grid.
2. Finding the first available empty cell.
3. Generating all possible rectangles that can start from that cell.
4. Shuffling the rectangles and placing one of them on the board.
5. When the puzzle is fully filled with rectangles: identifying rectangles and leaving only one number as a clue in each of them. If the puzzle has no solution 
or multiple solutions, the algorithm tries leaving different clues in rectangles of the same puzzle. This process is repeated until a valid puzzle is found.
6. The final puzzle is saved to a file, and the solver is run to verify its correctness.
'''
import random
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os

PUZZLE_SIZE = 5
PUZZLE_FILE_PATH = "shikaku_greedy.csv"
PUZZLE_IMAGE_PATH = "shikaku.png"

def fill_board(board, rectangles):
    '''Finds a free cell and inserts a random rectangle'''
    for row in range(PUZZLE_SIZE):
        for col in range(PUZZLE_SIZE):
            if board[row, col] != 0:
                continue
            possible_rectangles = get_possible_rectangles(row, col, board)
            if not possible_rectangles:
                return False
            height, width = random.choice(possible_rectangles)
            board[row:row+height, col:col+width] = height * width
            rectangles.append((row, col, height, width))
    return True

def get_possible_rectangles(start_row, start_col, board):
    '''Finds all possible rectangles for a given cell'''
    possible_rectangles = []
    
    max_height = PUZZLE_SIZE - start_row
    max_width = PUZZLE_SIZE - start_col

    for height in range(1, max_height + 1):
        for width in range(1, max_width + 1):
            if np.all(board[start_row:start_row+height, start_col:start_col+width] == 0):
                possible_rectangles.append((height, width))
                
    return possible_rectangles

def remove_extra_numbers(board, rectangles):
    '''Removes digits from a filled puzzle, leaving only one in each rectangle'''
    for top_row, left_col, height, width in rectangles:
        keep_row = top_row + random.randint(0, height - 1)
        keep_col = left_col + random.randint(0, width - 1)
        for row in range(top_row, top_row + height):
            for col in range(left_col, left_col + width):
                if (row, col) != (keep_row, keep_col):
                    board[row, col] = 0

def save_to_file(board):
    with open(PUZZLE_FILE_PATH, 'w') as f:
        f.write(f"{PUZZLE_SIZE} {PUZZLE_SIZE}\n")
        for row in board:
            f.write(" ".join(map(str, row)) + "\n")

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
                ax.text(j + 0.5, PUZZLE_SIZE - 0.5 - i, str(val), va='center', ha='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(PUZZLE_IMAGE_PATH, dpi=300)
    print(f"Found unique solution, shikaku puzzle saved to file: {PUZZLE_FILE_PATH}")
    print(f"Shikaku image saved to {PUZZLE_IMAGE_PATH}")
    plt.close()
    
def run_solver():
    DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    SHIKAKU_PATH = os.path.join(DIR_PATH, "shikaku_solver.py")
    
    command = f"python {SHIKAKU_PATH} -f {PUZZLE_FILE_PATH}"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if "Found solution" in result.stdout:
            return True
        else:
            print("not unique solution")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error while running solver: {e}")
        return False

def generate_shikaku():
    board = np.zeros((PUZZLE_SIZE, PUZZLE_SIZE), dtype=int)
    rectangles = []
    if fill_board(board, rectangles):
        while True:
            copy_board = board.copy()
            remove_extra_numbers(copy_board, rectangles)
            save_to_file(copy_board)
            if run_solver():
                break
            else:
                print("Not unique, regenerating...")

generate_shikaku()
