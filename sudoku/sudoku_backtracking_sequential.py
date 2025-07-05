'''
1. Generating an empty grid.  
2. Filling in cells sequentially, starting from the top-left corner until the entire grid is filled.
3. If no valid digit can be placed, backtracking is applied.
4. Once the puzzle is fully filled, random digits are removed according to the CLUES_NUMBER.
Alternatively, a clues mask can be used, where ones indicate clue positions. The algorithm then removes digits at positions marked with zeros.
If the puzzle has a valid solution, it is saved. Otherwise, clues are added back from the removed digits until a valid solution is found.
5. The solver is used to verify the correctness of the puzzle, and the puzzle is saved to a file.
'''
import random
import numpy as np
import matplotlib.pyplot as plt

CLUES_NUMBER = 55
PUZZLE_FILE_PATH = "sudoku_backtracking_sequential.csv"
# clues_mask = [0,1,1,0,1,0,0,0,0,
#               0,1,0,1,0,1,1,0,0,
#               0,0,1,1,0,0,0,0,0,
#               0,0,0,1,0,0,1,0,0,
#               0,1,1,0,0,0,0,0,0,
#               0,0,0,1,1,0,0,1,0,
#               0,1,0,0,0,1,1,1,0,
#               1,0,1,0,0,0,0,0,0,
#               1,0,1,0,0,1,0,0,0] # edit the mask by providing the positions of the clues. Use clues_mask = None if you want to receive random layouts
clues_mask = None
extra_clues = []

def is_valid(board, row, col, num):
    '''Check if a digit can be entered in a given cell without causing duplicates in the row, column, and square'''
    if num in board[row]:
        return False
    if num in board[:, col]:
        return False
    start_row, start_col = (row // 3) * 3, (col // 3) * 3
    if num in board[start_row:start_row+3, start_col:start_col+3]:
        return False
    return True

def get_possible_numbers(board, row, col):
    '''Returns a list of possible digits that can be entered in the cell (row, col)'''
    return [num for num in range(1, 10) if is_valid(board, row, col, num)]

def find_empty_cell(board):
    '''Finds an empty cell'''
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i, j] == 0:
                return i, j
    return None

def solver(board, solutions, limit=5):
    '''Solves a sudoku and tries to find multiple solutions'''
    cell = find_empty_cell(board)
    if cell is None:
        solutions.append(np.copy(board))
        return
    
    row, col = cell
    for val in get_possible_numbers(board, row, col):
        board[row, col] = val
        solver(board, solutions, limit)
        if len(solutions) >= limit:
            board[row, col] = 0
            return
        board[row, col] = 0

def run_solver(board):
    solutions = []
    solver(board.copy(), solutions, limit=5)
    print(f"Found at least {len(solutions)} solution(s).")
    return len(solutions)

def save_to_file(board):
    with open(PUZZLE_FILE_PATH, 'w', newline='') as f:
        for row in board:
            row_to_save = [' ' if num == 0 else str(num) for num in row]
            f.write(','.join(row_to_save) + '\n')
    
    _, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.axis('off')

    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i, color='black', lw=lw)
        ax.axvline(i, color='black', lw=lw)

    for i in range(9):
        for j in range(9):
            val = board[i][j]
            if val != 0:
                ax.text(j + 0.5, 8.5 - i, str(val), va='center', ha='center', fontsize=14)

    plt.tight_layout()
    plt.savefig("sudoku.png", dpi=300)
    plt.close()
    print("Sudoku image saved to sudoku.png")

def generate_sudoku(clues_mask=None):
    board = np.zeros((9, 9), dtype=int)

    def fill_sudoku():
        '''Fills the grid with numbers sequentially from the first empty cell to the last using backtracking.'''
        empty_cell = find_empty_cell(board)
        if not empty_cell:
            return True
        
        row, col = empty_cell
        numbers = list(range(1, 10))
        random.shuffle(numbers)
        
        for num in numbers:
            if is_valid(board, row, col, num):
                board[row, col] = num
                if fill_sudoku():
                    return True
                board[row, col] = 0
        
        return False
    
    def remove_numbers():
        board_copy = board.copy()
        removed = []

        if clues_mask is not None:
            if len(clues_mask) != 81:
                raise ValueError("clues_mask must have exactly 81 elements.")
            for idx, keep in enumerate(clues_mask):
                row, col = divmod(idx, 9)
                if keep == 0:
                    removed.append((row, col, board_copy[row, col]))
                    board_copy[row, col] = 0
        else:
            num_to_remove = 81 - CLUES_NUMBER
            positions = [(i, j) for i in range(9) for j in range(9)]
            random.shuffle(positions)

            for _ in range(num_to_remove):
                row, col = positions.pop()
                removed.append((row, col, board_copy[row, col]))
                board_copy[row, col] = 0

        solutions = run_solver(board_copy)
        added_back = 0
        
        while solutions != 1 and removed:
            idx = random.randint(0, len(removed) - 1)
            row, col, val = removed.pop(idx)
            board_copy[row, col] = val
            added_back += 1
            solutions = run_solver(board_copy)

        extra_clues.append(added_back)
        if solutions == 1:
            print(f"Sudoku has a unique solution. Restored {added_back} digits besides the initial {CLUES_NUMBER}")
            save_to_file(board_copy)
            print(f"Sudoku saved to file {PUZZLE_FILE_PATH}")
        else:
            print("No unique solution")

        return board_copy

    fill_sudoku()
    final_board = remove_numbers()
    return final_board

generate_sudoku(clues_mask)
