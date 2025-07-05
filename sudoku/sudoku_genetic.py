'''
1. Creating a pool of individuals, where each individual receives a fixed list of positions where Sudoku clues will be placed. 
The list of positions is the same for all individuals in a given population. It is possible to provide a custom clues_mask or generate a random one. 
Then, each individual randomly selects digits to place in these positions while checking if a clue is valid and possible to be places. If it is impossible
to completely fill the individual, a new one is created until the entire population is generated.
2. The fitness function evaluates individuals based on the number of solutions - 2 solutions equals 2 penalty points, and so on. Maximum of 5 solutions are checked.
3. Rank selection - individuals are selected for the next generation; the better the fitness, the higher the chance of being selected.
4. A certain number of elite individuals remains unchanged and proceeds directly to the next generation.
5. Crossover - the remaining individuals are created by crossing selected parents. 
The child inherits all rows from one parent, and then rows from the second parent are swapped in only if the swap doesn't violate Sudoku rules. 
If the swap breaks the rules, the row is left unchanged.
6. Mutation - each clue is mutated with a given probability. If mutation occurs, the digit is replaced with another valid digit, if available for this cell.
7. The solver is used to verify the correctness of the puzzle, and the puzzle is saved to a file.
'''
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

CLUES_NUMBER = 25
POPULATION_SIZE = 100
GENERATIONS = 100
ELITE_PERCENT = 0.1
MUTATION_RATE = 0.1
PUZZLE_FILE_PATH = "sudoku_genetic.csv"
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

def generate_initial_population(clue_positions):
    '''Creates an initial population consisting of boards randomly filled with digits in the clue_positions fields, whose layout is fixed for the entire population'''
    population = []
    max_attempts = 1000
    attempts = 0

    print("Generating population...")

    while len(population) < POPULATION_SIZE and attempts < max_attempts:
        attempts += 1
        board = np.zeros((9, 9), dtype=int)
        valid = True

        for (row, col) in clue_positions:
            options = get_possible_numbers(board, row, col)
            if not options:
                valid = False
                break
            board[row, col] = random.choice(options)

        if valid:
            population.append(board)

    if len(population) < POPULATION_SIZE:
        print("Failed to generate a sufficient population. Try again with different parameters or clue_positions.")
    else:
        print("Population generation completed, starting genetic algorithm...")

    return population

def fitness(board):
    '''
    Returns a penalty based on the number of solutions:
    0 -> penalty 6 (no solution), 1 -> 0 (solution), 2 -> penalty 2, 3 -> penalty 3, 4 -> penalty 4, 5 or more -> penalty 5.
    '''
    num_solutions = run_solver(board)
    if num_solutions == 0:
        return 6
    elif num_solutions == 1:
        return 0
    elif num_solutions >= 5:
        return 5
    else:
        return num_solutions

def crossover(parent1, parent2):
    '''
    The child copies parent1. Then it swaps corresponding rows with parent2 provided that the swap does not violate Sudoku rules.
    If the rules are broken, the row will not be swapped.
    '''
    child = parent1.copy()

    for row in range(9):
        candidate_row = parent2[row]
        valid_swap = True
        for col in range(9):
            num = candidate_row[col]
            original = child[row, col]
            child[row, col] = 0
            if not is_valid(child, row, col, num):
                valid_swap = False
            child[row, col] = original
            if not valid_swap:
                break

        if valid_swap:
            child[row] = candidate_row

    return child

def mutate(clue_board, clue_positions):
    '''Each of the initial digits has a 10% chance of mutation, meaning it can be changed to another possible digit.'''
    board = clue_board.copy()
    for (row, col) in clue_positions:
        if random.random() < MUTATION_RATE:
            original_value = board[row, col]
            possible_numbers = list(range(1, 10))
            random.shuffle(possible_numbers)
            for num in possible_numbers:
                if num != original_value:
                    board[row, col] = 0
                    if is_valid(board, row, col, num):
                        board[row, col] = num
                        break
            else:
                board[row, col] = original_value
    return board

def selection(population):
    '''Rank selection with probability'''
    selected = []
    size = len(population)
    for i, individual in enumerate(population):
        probability = max(1, 100 - int(100 * (i / size)))
        if random.randint(1, 100) <= probability:
            selected.append(copy.deepcopy(individual))
    return selected

def generate_sudoku():
    if clues_mask is not None:
        if len(clues_mask) != 81:
            raise ValueError("clues_mask must have exactly 81 elements.")
        clue_positions = [(i // 9, i % 9) for i, val in enumerate(clues_mask) if val == 1]
    else:
        all_positions = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(all_positions)
        clue_positions = all_positions[:CLUES_NUMBER]

    population = generate_initial_population(clue_positions)
    elite_size = int(POPULATION_SIZE * ELITE_PERCENT)

    for generation in range(GENERATIONS):
        population = sorted(population, key=fitness)
        best_fitness = fitness(population[0])
        if best_fitness == 6:
            print(f"Generation {generation}, best fitness: {best_fitness} (no solution)")
        elif(best_fitness > 1 and best_fitness < 6):
            print(f"Generation {generation}, best fitness: {best_fitness} (too many solutions)")
        else:
            print(f"Generation {generation}, best fitness: {best_fitness}")

        if best_fitness == 0:
            print(f"Found a puzzle with a unique solution in generation {generation}")
            sudoku = population[0]
            save_to_file(sudoku)
            return sudoku

        elite = population[:elite_size]
        rest = population[elite_size:]
        selected = selection(rest)
        next_generation = elite.copy()

        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = random.sample(elite + selected, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, clue_positions)
            next_generation.append(child)

        population = next_generation

    print("No complete or unique solution found.")
    sudoku = min(population, key=fitness)
    return sudoku

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

def find_best_cell(board):
    min_options = 10
    best_cell = None
    for row in range(9):
        for col in range(9):
            if board[row, col] == 0:
                options = get_possible_numbers(board, row, col)
                if len(options) < min_options:
                    min_options = len(options)
                    best_cell = (row, col)
                    if min_options == 1:
                        return best_cell
    return best_cell

def solver(board, solutions, limit=5):
    '''Solves a sudoku and tries to find multiple solutions'''
    if len(solutions) >= limit:
        return

    cell = find_best_cell(board)
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

generate_sudoku()
