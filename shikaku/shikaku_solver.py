import argparse
from itertools import product
import numpy as np

def read_puzzle_from_file(filename):
    with open(filename) as file:
        height, width = map(int, file.readline().split())
        puzzle_grid = [list(map(int, file.readline().split())) for _ in range(height)]
    return puzzle_grid, height, width

def find_divisors_for_area(area, max_height, max_width):
    possible_rectangles = []
    for height in range(1, area + 1):
        if area % height == 0:
            width = area // height
            if height <= max_height and width <= max_width:
                possible_rectangles.append((height, width))
            if width < max_height and height <= max_width:
                possible_rectangles.append((width, height))
    return possible_rectangles

def can_place_rectangle(board, start_row, start_col, rect_height, rect_width):
    max_rows = len(board)
    max_cols = len(board[0])

    if start_row + rect_height > max_rows or start_col + rect_width > max_cols:
        return False

    for row in range(start_row, start_row + rect_height):
        for col in range(start_col, start_col + rect_width):
            if board[row][col] != 0:
                return False
    return True

def place_rectangle_on_board(board, start_row, start_col, rect_height, rect_width, value):
    for row in range(start_row, start_row + rect_height):
        for col in range(start_col, start_col + rect_width):
            board[row][col] = value

def backtracking_solver(clues, board, clue_index, height, width):
    if clue_index == len(clues):
        return True

    row, col, clue_value, possible_rects = clues[clue_index]

    for rect_height, rect_width in possible_rects:
        for start_row in range(row - rect_height + 1, row + 1):
            for start_col in range(col - rect_width + 1, col + 1):
                if 0 <= start_row <= height - rect_height and 0 <= start_col <= width - rect_width:
                    if can_place_rectangle(board, start_row, start_col, rect_height, rect_width):
                        place_rectangle_on_board(board, start_row, start_col, rect_height, rect_width, clue_value)

                        if backtracking_solver(clues, board, clue_index + 1, height, width):
                            return True

                        place_rectangle_on_board(board, start_row, start_col, rect_height, rect_width, 0)
    return False

def solve_puzzle(filename):
    puzzle, height, width = read_puzzle_from_file(filename)
    board = np.zeros((width, height), dtype=int)
    clues = []
    for row, col in product(range(height), range(width)):
        clue = puzzle[row][col]
        if clue > 0:
            possible_rectangles = find_divisors_for_area(clue, height, width)
            clues.append((row, col, clue, possible_rectangles))
    clues.sort(key=lambda clue: len(clue[3]))

    if backtracking_solver(clues, board, 0, height, width):
        print("Found solution")
        return True
    else:
        print("No solution")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    args = parser.parse_args()
    solve_puzzle(args.file)
