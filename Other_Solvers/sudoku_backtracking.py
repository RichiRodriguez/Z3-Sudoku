
# A Utility Function to print the Grid
def print_grid(arr):
	for i in range(N):
		for j in range(N):
			print (arr[i][j], end = " "),
		print ()

		
# Function to Find the entry in
# the Grid that is still not used
# Searches the grid to find an
# entry that is still unassigned. If
# found, the reference parameters
# row, col will be set the location
# that is unassigned, and true is
# returned. If no unassigned entries
# remains, false is returned.
# 'l' is a list variable that has
# been passed from the solve_sudoku function
# to keep track of incrementation
# of Rows and Columns
def find_empty_location(arr, l):
	for row in range(N):
		for col in range(N):
			if(arr[row][col]== 0):
				l[0]= row
				l[1]= col
				return True
	return False

# Returns a boolean which indicates
# whether any assigned entry
# in the specified row matches
# the given number.
def used_in_row(arr, row, num):
	for i in range(N):
		if(arr[row][i] == num):
			return True
	return False

# Returns a boolean which indicates
# whether any assigned entry
# in the specified column matches
# the given number.
def used_in_col(arr, col, num):
	for i in range(N):
		if(arr[i][col] == num):
			return True
	return False

# Returns a boolean which indicates
# whether any assigned entry
# within the specified 3x3 box
# matches the given number
def used_in_box(arr, row, col, num):
	for i in range(Nsqrt):
		for j in range(Nsqrt):
			if(arr[i + row][j + col] == num):
				return True
	return False

# Checks whether it will be legal
# to assign num to the given row, col
# Returns a boolean which indicates
# whether it will be legal to assign
# num to the given row, col location.
def check_location_is_safe(arr, row, col, num):
	
	# Check if 'num' is not already
	# placed in current row,
	# current column and current 3x3 box
	return (not used_in_row(arr, row, num) and
		(not used_in_col(arr, col, num) and
		(not used_in_box(arr, row - row % Nsqrt,
						col - col % Nsqrt, num))))

# Takes a partially filled-in grid
# and attempts to assign values to
# all unassigned locations in such a
# way to meet the requirements
# for Sudoku solution (non-duplication
# across rows, columns, and boxes)
def solve_sudoku(arr):
	
	# 'l' is a list variable that keeps the
	# record of row and col in
	# find_empty_location Function	
	l =[0, 0]
	
	# If there is no unassigned
	# location, we are done	
	if(not find_empty_location(arr, l)):
		return True
	
	# Assigning list values to row and col
	# that we got from the above Function
	row = l[0]
	col = l[1]
	
	# consider digits 1 to 9
	for num in range(1, N + 1):
		
		# if looks promising
		if(check_location_is_safe(arr,
						row, col, num)):
			
			# make tentative assignment
			arr[row][col]= num

			# return, if success,
			# ya !
			if(solve_sudoku(arr)):
				return True

			# failure, unmake & try again
			arr[row][col] = 0
			
	# this triggers backtracking		
	return False

# Driver main function to test above functions
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


grid = grid_16x16
N = len(grid)
Nsqrt = int(N**(0.5))
	
# if success print the grid
if(solve_sudoku(grid)):
    print_grid(grid)
else:
    print ("No solution exists")

# The above code has been contributed by Harshit Sidhwa.
