'''
1. Creating a pool of individuals; each individual has randomly selected cells marked as water, with probability equal to FILLING.
2. Calculating the fitness of each individual based on the number of 2x2 water blocks, connections to the border, and the number of separate water regions.
3. Rank-based selection - individuals are selected for the next generation; the better the fitness, the higher the chance of being selected.
4. A certain number of elite individuals remains unchanged and proceeds directly to the next generation.
5. Crossover - the remaining individuals are created by crossing selected parents. The child inherits half of the rows from one parent and the other half from the second parent.
6. Mutation - a random cell is changed from water to island or from island to water, with a probability defined by MUTATION_RATE.
7. Repeating the algorithm for a set number of generations until the best-fitting individual is found.
8. Identifying the islands, calculating their sizes, storing the size in one of the island's cells, and clearing the rest.
9. Using a solver to find a valid solution. If the puzzle is valid but unsolvable, a new one is generated.

'''
import random
import copy
import subprocess
import matplotlib.pyplot as plt
import os

PUZZLE_SIZE = 7
FILLING = 0.8 #The lower the number, the harder is nurikabe. At certain point, usually lower than 0.3 it's hard or impossible to generate valid nurikabe.
POPULATION_SIZE = 100
GENERATIONS = 100
ELITE_PERCENT = 0.1
MUTATION_RATE = 0.1
FITNESS = "easy" #easy/hard
PUZZLE_FILE_PATH = "nurikabe_genetic.csv"
PUZZLE_IMAGE_PATH = "nurikabe.png"

def create_random_grid():
    '''Creates an individual by filling random elements with water with probability FILLING'''
    return [['.' if random.random() < FILLING  else 0 for _ in range(PUZZLE_SIZE)] for _ in range(PUZZLE_SIZE)]

def generate_population():
    return [create_random_grid() for _ in range(POPULATION_SIZE)]

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

def count_disconnected_water_regions(grid):
    '''Counts unique water paths'''
    visited = set()

    def dfs(x, y):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if is_valid(nx, ny) and (nx, ny) not in visited and grid[nx][ny] == '.':
                    visited.add((nx, ny))
                    stack.append((nx, ny))

    regions = 0
    for i in range(PUZZLE_SIZE):
        for j in range(PUZZLE_SIZE):
            if grid[i][j] == '.' and (i, j) not in visited:
                visited.add((i, j))
                dfs(i, j)
                regions += 1
    return max(0, regions - 1)

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

def check_max_island_size(grid):
    '''Checks the size of the largest island'''
    max_allowed_size = (PUZZLE_SIZE * PUZZLE_SIZE)//4
    islands = find_islands(grid)
    for island in islands:
        if len(island) > max_allowed_size:
            return 5
    return 0

def fitness_easy(grid):
    '''
    Fitness function that checks how many 2x2 blocks exist, whether there are separate water paths, 
    and if there is an island whose size is larger than half of all cells.
    '''
    fitness = 0
    fitness += has_2x2_block(grid)
    fitness += count_disconnected_water_regions(grid)
    fitness += check_max_island_size(grid)

    return fitness

def fitness_hard(grid):
    '''
    Fitness function that checks how many 2x2 blocks exist, whether there are separate water paths,
    whether water touches the puzzle edge, and if there is an island whose size is larger than half of all cells. Additionally,
    it checks if there are islands of size 1. A perfect fit allows for a maximum of 5% of islands of size 1.
    '''
    fitness = 0
    fitness += has_2x2_block(grid)
    fitness += count_disconnected_water_regions(grid)
    fitness += check_max_island_size(grid)
 
    islands = find_islands(grid)
    one_cell_islands = sum(1 for island in islands if len(island) == 1)
    max_allowed = int(0.05 * PUZZLE_SIZE * PUZZLE_SIZE)

    if one_cell_islands > max_allowed:
        fitness += 5

    return fitness

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
    '''Crossover of two parents, the child receives half of the rows from the first parent and the other half from the second parent'''
    split = PUZZLE_SIZE // 2
    child = [row[:] for row in parent1[:split]] + [row[:] for row in parent2[split:]]
    return child

def mutate(grid):
    '''Swaps one random cell from water to island or vice versa'''
    i = random.randint(0, PUZZLE_SIZE - 1)
    j = random.randint(0, PUZZLE_SIZE - 1)

    grid[i][j] = '.' if grid[i][j] == 0 else 0
    return grid

def genetic_algorithm():
    population = generate_population()
    if FITNESS== 'easy':
        fitness = fitness_easy
    elif FITNESS == 'hard':
        fitness = fitness_hard
        
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
    plt.savefig("nurikabe.png", dpi=300)
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

def generate_nurikabe():
    while True:
        nurikabe = genetic_algorithm()
        if nurikabe is not None:
            islands = find_islands(nurikabe)
            assign_island_numbers(nurikabe, islands)
            save_to_file(nurikabe)

            if run_solver():
                break
            else:
                print("Not unique, regenerating...")

generate_nurikabe()
