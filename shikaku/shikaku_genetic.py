'''
1. Creating a pool of individuals. For each individual, a list of rectangles is generated and placed on the board until either all rectangles are placed 
or there are no more available positions.
2. Calculating the fitness of each individual. There are two types of fitness functions (easy and hard), depending on the desired puzzle difficulty.
   Easy - penalizes empty cells not belonging to any rectangle, 1-cell rectangles if they exceed 10% of the board, and 2- or 3-cell rectangles if they exceed 20% of the board.
   Hard - additionally penalizes rectangles that are too large and 1-cell rectangles if there is more than one.
3. Rank-based selection - individuals are selected for the next generation; the better the fitness, the higher the chance of being selected.
4. A certain number of elite individuals remains unchanged and proceeds directly to the next generation.
5. Crossover - the remaining individuals are created by crossing selected parents. The child takes half of the rectangles from parent 1 and then 
attempts to add rectangles from parent 2 until all cells are filled or no more rectangles can be added.
6. Mutation - with a probability defined by MUTATION_RATE, one rectangle is removed and replaced with a randomly selected new rectangle in an available space.
7. The algorithm is repeated for a defined number of generations until the best-fitting individual is found.
8. Identifying rectangles and leaving only one number as a clue in each of them. If the puzzle has no solution or multiple solutions,
the algorithm tries leaving different clues in rectangles of the same puzzle. This process is repeated until a valid puzzle is found.
9. Using a solver to verify the correctness of the puzzle and saving it to a file.
'''
import numpy as np
import random
import copy
import subprocess
import matplotlib.pyplot as plt
import os

PUZZLE_SIZE = 10
ELITE_PERCENT = 0.1
MUTATION_RATE = 0.1
POPULATION_SIZE = 100
GENERATIONS = 100
FITNESS = "easy" #easy/hard puzzle
PUZZLE_FILE_PATH = "shikaku_genetic.csv"
PUZZLE_IMAGE_PATH = "shikaku.png"

def generate_random_rectangle(remaining_area):
    '''Generates a random rectangle with area <= remaining_area and <= 1/4 of the board.'''
    possible_rectangles = []
    max_area = (PUZZLE_SIZE * PUZZLE_SIZE) // 4

    for h in range(1, PUZZLE_SIZE + 1):
        for w in range(1, PUZZLE_SIZE + 1):
            area = h * w
            if area <= remaining_area and area <= max_area:
                possible_rectangles.append((h, w))

    return random.choice(possible_rectangles)

def generate_individual():
    '''Generates an individual - a list of rectangles whose total area equals the entire puzzle area.'''
    total = PUZZLE_SIZE * PUZZLE_SIZE
    rectangles = []
    while total > 0:
        rect = generate_random_rectangle(total)
        if rect is None:
            break
        area = rect[0] * rect[1]
        if area <= total:
            rectangles.append(rect)
            total -= area
        else:
            break
    return rectangles

def place_rectangles(board, rectangles):
    '''Places rectangles on the board'''
    board.fill(0)
    placed = []
    for (height, width) in rectangles:
        area = height * width
        placed_flag = False
        for row in range(PUZZLE_SIZE - height + 1):
            for col in range(PUZZLE_SIZE - width + 1):
                if np.all(board[row:row+height, col:col+width] == 0):
                    board[row:row+height, col:col+width] = area
                    placed.append((row, col, height, width))
                    placed_flag = True
                    break
            if placed_flag:
                break
    return board, placed

def fitness_easy(board):
    '''
    - penalty (+) number of empty cells * 10
    - penalty (+) one point for each one when the number of ones > 10% of the board
    - penalty (+) one point for each 2 or 3 exceeding 20% of the board
    '''
    total = PUZZLE_SIZE * PUZZLE_SIZE
    
    empty_count = np.sum(board == 0)
    ones_count = np.sum(board == 1)
    small_count = np.sum((board == 2) | (board == 3))
    
    penalty = 0
    
    if ones_count > 0.1 * total:
        penalty += ones_count - int(0.1 * total)
    
    small_threshold = int(0.20 * total)
    if small_count > small_threshold:
        penalty += (small_count - small_threshold)
    
    return empty_count + penalty


def fitness_hard(board):
    '''
    - penalty (+) number of empty cells * 10
    - penalty (+) one point for each one, if the number of ones > 1
    - penalty (+) one point for each digit with size >= 25% of the board
    - penalty (+) one point for each 2 or 3 missing up to 20% of the board
    '''
    total = PUZZLE_SIZE * PUZZLE_SIZE
    empty_count = np.sum(board == 0)
    ones_count = np.sum(board == 1)
    small_count = np.sum((board == 2) | (board == 3))
    large_count = np.sum(board >= total * 0.25)

    
    penalty = 0
    penalty += empty_count * 10

    if ones_count > 1:
        penalty += ones_count
    penalty += large_count

    small_threshold = int(0.20 * total)
    if small_count < small_threshold:
        penalty += (small_threshold - small_count)

    return penalty

def selection(population):
    '''Rank selection with probability'''
    selected = []
    size = len(population)
    for i, individual in enumerate(population):
        probability = max(1, 100 - int(100 * (i / size)))
        if random.randint(1, 100) <= probability:
            selected.append(copy.deepcopy(individual))
    return selected

def crossover(parent1, parent2):
    '''Takes half of the rectangles from parent1, then tries to add rectangles from parent2.'''
    half = len(parent1) // 2
    child_rects = parent1[:half]

    board = np.zeros((PUZZLE_SIZE, PUZZLE_SIZE), dtype=int)
    _, placed = place_rectangles(board, child_rects)
    for rect in parent2:
        height, width = rect
        area = height * width
        can_place = False
        for row in range(PUZZLE_SIZE - height + 1):
            for col in range(PUZZLE_SIZE - width + 1):
                if np.all(board[row:row+height, col:col+width] == 0):
                    can_place = True
                    break
            if can_place:
                break
        if can_place:
            child_rects.append(rect)
            board[row:row+height, col:col+width] = area
    return child_rects

def mutate(individual):
    '''Mutation - removes one rectangle and adds a random new one in an available spot.'''
    if not individual:
        return individual

    num_to_remove = random.randint(1, min(5, len(individual)))
    indices_to_remove = random.sample(range(len(individual)), num_to_remove)
    for idx in sorted(indices_to_remove, reverse=True):
        individual.pop(idx)

    board = np.zeros((PUZZLE_SIZE, PUZZLE_SIZE), dtype=int)
    place_rectangles(board, individual)

    empty_count = np.count_nonzero(board == 0)

    while empty_count > 0:
        new_rect = generate_random_rectangle(empty_count)
        individual.append(new_rect)
        empty_count -= new_rect[0]

    return individual

def genetic_algorithm():
    population = [generate_individual() for _ in range(POPULATION_SIZE)]
    board = np.zeros((PUZZLE_SIZE, PUZZLE_SIZE), dtype=int)
    if FITNESS== 'easy':
        fitness = fitness_easy
    elif FITNESS == 'hard':
        fitness = fitness_hard
    elite_size = int(POPULATION_SIZE * ELITE_PERCENT)

    for generation in range(GENERATIONS):
        fitnesses = []
        placed_list = []
        for ind in population:
            b, placed = place_rectangles(board.copy(), ind)
            fit = fitness(b)
            fitnesses.append(fit)
            placed_list.append((b, placed))

        best_fitness = min(fitnesses)
        best_idx = fitnesses.index(best_fitness)
        print(f"Generation {generation}, best fitness: {best_fitness}")
        if best_fitness == 0:
            print("Found perfect individual")
            return placed_list[best_idx][0], placed_list[best_idx][1]
        
        sorted_population = [ind for ind, _ in sorted(zip(population, fitnesses), key=lambda x: x[1])]
        elites = [copy.deepcopy(ind) for ind in sorted_population[:elite_size]]
        rest = sorted_population[elite_size:]
        selected = selection(rest)

        next_generation = elites.copy()
        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = random.sample(elites + selected, 2)
            child = crossover(parent1, parent2)
            if random.random() < MUTATION_RATE:
                child = mutate(child)

            next_generation.append(child)

        population = next_generation

    fitnesses = []
    placed_list = []
    for p in population:
        puzzle, placed = place_rectangles(board.copy(), p)
        fit = fitness(puzzle)
        fitnesses.append(fit)
        placed_list.append((puzzle, placed))
    best_fitness = min(fitnesses)
    best_idx = fitnesses.index(best_fitness)
    print(f"Best score after {GENERATIONS} generations: {best_fitness}")
    return placed_list[best_idx][0], placed_list[best_idx][1]

def remove_extra_numbers(board, rectangles):
    '''Removes digits from a filled puzzle, leaving only one in each rectangle'''
    for (top_row, left_col, height, width) in rectangles:
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
    board, placed_rects = genetic_algorithm()
    if board is not None:
        while True:
            copy_board = board.copy()
            remove_extra_numbers(copy_board, placed_rects)
            save_to_file(copy_board)

            if run_solver():
                break
            else:
                print("Not unique, regenerating...")

generate_shikaku()
