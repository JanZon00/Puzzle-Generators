'''
1. Generating an empty grid.
2. The grid is randomly filled with paths of length 2. If the puzzle has an odd number of cells, one of the paths will have a length of 3.
3. Two adjacent paths are randomly merged into one longer path, if possible.
4. Fitness - each individual is evaluated based on the number of paths it contains. The ideal individual has a number of paths equal to MAX_PATHS.
5. Rank selection - individuals are selected for the next generation; the better the fitness, the higher the chance of being selected.
6. A certain number of elite individuals remains unchanged and proceeds directly to the next generation.
7. Crossover - the child takes half of the rows from parent 1 and half from parent 2. Any path connections at the row boundaries are completed if possible.
8. Mutation - one of the paths is split into two parts.
9. After a valid individual is found, all puzzle cells except the start and end points are removed, the solver is run, and the final puzzle is saved to a file.
'''
import random
import copy
import matplotlib.pyplot as plt
from NumberlinkPuzzle import NumberlinkPuzzle

PUZZLE_SIZE = 6
MAX_PATHS = 4
POPULATION_SIZE = 100
GENERATIONS = 100
ELITE_PERCENT = 0.1
MUTATION_RATE = 0.1
PUZZLE_FILE_PATH = "numberlink_genetic.csv"
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

def merge_paths(board, MAX_PATHS):
    '''Merges paths until no more merges are possible or until the path limit is reached'''
    current_paths = sorted({cell for row in board for cell in row if cell > 0})

    while len(current_paths) > MAX_PATHS:
        merged_any = False
        values = current_paths[:]
        random.shuffle(values)

        for v1 in values:
            ends1 = find_path_ends(board, v1)
            random.shuffle(ends1)

            merged_this_round = False
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
                                merged_any = True
                                merged_this_round = True
                                break

                            else:
                                for i, j in old_v2_cells:
                                    board[i][j] = v2
                if merged_this_round:
                    break
            if merged_any:
                break

        if not merged_any:
            break

        current_paths = sorted({cell for row in board for cell in row if cell > 0})
    success = len(current_paths) <= MAX_PATHS
    return success, board

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

def fitness(board):
    '''Penalty for each path over the limit'''
    paths = sorted({cell for row in board for cell in row if cell > 0})
    count_paths = len(paths)
    if count_paths <= MAX_PATHS:
        return 0
    else:
        return (count_paths - MAX_PATHS)

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
    '''Takes half of the rows from parent1 and the other half from parent2'''
    midpoint = PUZZLE_SIZE // 2

    max_val_p1 = max([cell for row in parent1 for cell in row], default=0)
    offset = max_val_p1 + 1

    child = [row[:] for row in parent1[:midpoint]]
    for row in parent2[midpoint:]:
        new_row = [(cell + offset) if cell > 0 else 0 for cell in row]
        child.append(new_row)

    _, merged = merge_paths(child, MAX_PATHS)
    return merged if merged is not None else child

def mutate(board):
    '''Splits one of the paths into two parts'''
    board = copy.deepcopy(board)
    path_values = sorted({cell for row in board for cell in row if cell > 0})

    if not path_values:
        return board

    num_to_split = random.randint(1, min(3, len(path_values)))
    random.shuffle(path_values)
    targets = path_values[:num_to_split]

    next_value = max(path_values) + 1

    for target_value in targets:
        path = get_path_cells(board, target_value)
        if len(path) < 2:
            continue

        random.shuffle(path)
        split_index = len(path) // 2
        split = path[split_index:]

        for i, j in split:
            board[i][j] = next_value
        next_value += 1

    return board

def generate_population():
    population = []
    base_board = divide_board()
    for _ in range(POPULATION_SIZE):
        board = copy.deepcopy(base_board)
        _, merged_board = merge_paths(board, MAX_PATHS)
        population.append(merged_board)
    return population

def genetic_algorithm():
    population = generate_population()

    elite_size = int(POPULATION_SIZE * ELITE_PERCENT)

    for generation in range(GENERATIONS):
        population = sorted(population, key=fitness)
        best_fitness = fitness(population[0])
        print(f"Generation {generation}, best fitness: {best_fitness}")

        if best_fitness == 0:
            best_individual = population[0]
            return best_individual

        elite = population[:elite_size]
        rest = population[elite_size:]
        selected = selection(rest)
        next_generation = elite.copy()

        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = random.sample(elite + selected, 2)
            child = crossover(parent1, parent2)
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            next_generation.append(child)

        population = next_generation

    population = sorted(population, key=fitness)
    best_individual = population[0]
    print(f"Best result after {GENERATIONS} generations: {best_fitness}")
    return best_individual

def generate_numberlink():
    while True:
        board = genetic_algorithm()
        board = filter_path_ends_only(board)
        save_to_file(board)
        if run_solver():
            break
        else:
            print("Not unique, regenerating...")

generate_numberlink()
