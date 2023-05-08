# -------------------------- Required Imports --------------------------

from z3 import *
import numpy as np


# -------------------------- Helper functions --------------------------

def get_mini_grid(array, i, j, size):
    return [ L[i * size + y][j * size + x] for y in range(size) for x in range(size) ]


def get_grid(puzzler):
    size = len(puzzler)
    m_size = int(size**(1/2))
    char_len = len(str(size))
    horizontal = '+' + ''.join([f'{"-"*(m_size * (char_len + 1) + 1)}+' for _ in range(m_size)])
    output = ''
    for r in range(size):
        if r % m_size == 0:
            output = output + f'{horizontal}\n'
        line = [' '.join([f'{_:>{char_len}}' for _ in puzzler[r][(c * m_size):(c * m_size+ m_size)]]) 
                for c in range(0, m_size)]
        line_with_divider = ' | '.join([_ for _ in line])
        output = output + f'| {line_with_divider} |\n'
    return f'{output}{horizontal}'.replace(f'{0:>{char_len}}',f'{"∙":>{char_len}}')


def get_solution(model, L):
    return [ [ model.eval(L[i][j]).as_long() for j in range(size) ] for i in range(size) ]


def side_by_side(grid1, grid2, label_1=None, label_2=None, spaces=5):
    s1, s2 = get_grid(puzzle), get_grid(solution)
    separator = ' ' * spaces
    output = [f'{l}{separator}{r}' for (l, r) in zip(s1.split('\n'), s2.split('\n'))]
    if label_1 is not None and label_2 is not None:
        s1_len = len(s1.split('\n')[0])
        out = f'{label_1:<{s1_len}}{separator}{label_2}'
        output.insert(0, out)
    return '\n'.join(output)


# -------------------------- Configuration --------------------------

# Set the puzzle
grid_9x9 = [
    [0,0,0,2,6,0,7,0,1],
    [6,8,0,0,7,0,0,9,0],
    [1,9,0,0,0,4,5,0,0],
    [8,2,0,1,0,0,0,4,0],
    [0,0,4,6,0,2,9,0,0],
    [0,5,0,0,0,3,0,2,8],
    [0,0,9,3,0,0,0,7,4],
    [0,4,0,0,5,0,0,3,6],
    [7,0,3,0,1,8,0,0,0]]

grid_16x16 = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,3,0,0,0,7,0,0,0,4],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,7,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,8],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,7,0,0,0,0,0,0,0,0,2,0,0,0]]

puzzle = grid_16x16

# Set the size and m_size
size = len(puzzle)
m_size = int(size**(0.5))



# -------------------------- Solution Setup --------------------------

L = []
for row in range(size):
    L.append( [Int(f'L_{row}{col}') for col in range(size)] )



# -------------------------- Solution Constraints --------------------------

# Constraints every cells to the number 1 - {size}, where size is the size of the Sudoku as in sizeXsize
constraint_1 = [ And( L[row][col] >= 1, L[row][col] <= size)  for col in range(size) for row in range(size) ]

# Constraint on columns, so that each column has to be distinct
constraint_2 = []
for col in range(size):
    constraint_2.append( Distinct( [L[row][col] for row in range(size)] ) )

# Constraint on rows, so that each row has to be distinct
constraint_3 = []
for row in range(size):
    constraint_3.append( Distinct( [L[row][col] for col in range(size)] ) )

# Constraint on mini grids, so that each mini grid has to be distinct
constraint_4 = [ Distinct( get_mini_grid(L, y, x, m_size) )  for y in range(m_size) for x in range(m_size) ]  



# -------------------------- Add known values --------------------------

known_values = [L[i][j] == puzzle[i][j] for i in range(size) for j in range(size) if puzzle[i][j]]



# -------------------------- Z3 Solver --------------------------

# Create Z3 solver
s = Solver()

# Add the constraints
s.add(constraint_1)
s.add(constraint_2)
s.add(constraint_3)
s.add(constraint_4)

# Add the known values
s.add(known_values)

# Ask the solver for a solution
status = str(s.check())

if status == 'unsat':
    print('Model is UNSAT')
else:
    solution = get_solution(s.model(), L)
    print(side_by_side(puzzle, solution, 'Original puzzle:', 'Solved puzzle:'))



# -------------------------- Verification --------------------------

def sudoku_distinct(puzzle, which='row'):
    if which not in ['col', 'row']:
        raise Exception(f'Invalid value for parameter which: "{which}"')
    grid = puzzle if isinstance(puzzle, np.ndarray) else np.array(puzzle)
    size = grid.shape[1]
    grid = grid if which == 'row' else grid.T
    distinct_row = np.arange(1, size + 1)
    condition_1 = np.logical_and(grid >= 1, grid <= size).all()
    condition_3 = np.all(np.sort(grid, axis=1, kind=None, order=None) == distinct_row)
    return np.all([condition_1, condition_3])

def sudoku_minigrids(puzzle):
    grid = puzzle if isinstance(puzzle, np.ndarray) else np.array(puzzle)
    size = grid.shape[0]
    m_size = int(size**(0.5))
    minigrids = []
    for i in range(m_size):
        for j in range(m_size):
            minigrids.append(grid[(i*m_size):(i*m_size + m_size), (j*m_size):(j*m_size + m_size)].reshape((-1,size)))
    return np.all([sudoku_distinct(_) for _ in minigrids])


if 'solution' not in locals():
    print('Model was UNSAT, therefore, there\'s no need for verification')
else:     
    rows = sudoku_distinct(solution, which = 'row')
    cols = sudoku_distinct(solution, which = 'col')
    mini_grids = sudoku_minigrids(solution)
    verification = rows & cols & mini_grids
    print(f'This sudoku solution follows all the constraints: {verification}')

