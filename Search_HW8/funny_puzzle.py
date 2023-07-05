import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    dist = 0
    for i in range(len(from_state)):
        if(from_state[i] == 0):
            continue
        ideal_state = to_state.index(from_state[i]);
        ideal_x =ideal_state%3 
        ideal_y = ideal_state//3
        dist += abs(i%3 - ideal_x) + abs(i//3 - ideal_y);
        #print(dist,i,from_state[i],abs(i%3 - ideal_x) + abs(i//3 - ideal_y));
        
    return dist




def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    successors = []
    pos1 = state.index(0)
    pos2 = state.index(0, pos1+1)
    row, col = pos1 // 3, pos1 % 3
    if(row > 0):
        new_state = state[:]
        new_state[pos1], new_state[pos1-3] = new_state[pos1-3], new_state[pos1]
        if(new_state != state):
            successors.append(new_state)

         
    if row < 2:
        # Move tile up
        new_state = state[:]
        new_state[pos1], new_state[pos1+3] = new_state[pos1+3], new_state[pos1]
        if(new_state != state):
            successors.append(new_state)

          
    if col > 0:
        # Move tile right
        new_state = state[:]
        new_state[pos1], new_state[pos1-1] = new_state[pos1-1], new_state[pos1]
        if(new_state != state):
            successors.append(new_state)
    
    if col < 2:
        # Move tile left
        new_state = state[:]
        new_state[pos1], new_state[pos1+1] = new_state[pos1+1], new_state[pos1]
        if(new_state != state):
            successors.append(new_state)
        
    row, col = pos2 // 3, pos2% 3

    if(row > 0):
        new_state = state[:]
        new_state[pos2], new_state[pos2-3] = new_state[pos2-3], new_state[pos2]
        if(new_state != state):
            successors.append(new_state)
        
        
    if row < 2:
        # Move tile up
        new_state = state[:]
        new_state[pos2], new_state[pos2+3] = new_state[pos2+3], new_state[pos2]
        if(new_state != state):
            successors.append(new_state)
       
    if col > 0:
        # Move tile right
        new_state = state[:]
        new_state[pos2], new_state[pos2-1] = new_state[pos2-1], new_state[pos2]
        if(new_state != state):
            successors.append(new_state)
        
    if col < 2:
        # Move tile left
        new_state = state[:]
        new_state[pos2], new_state[pos2+1] = new_state[pos2+1], new_state[pos2]
        if(new_state != state):
            successors.append(new_state)
        
    list_of_lists = successors


    unique_lists = list(set(map(tuple, list_of_lists)))
    unique_lists = [list(l) for l in unique_lists]

        
    return sorted(unique_lists)




def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    pq = []
    moves = 0
    h = get_manhattan_distance(state)
    explored = set()
    heapq.heappush(pq,(h, state, (0, h, -1)))
    parent_pointer = {} 
    max_queue_len = 0
    fin_state = None
    fin_state_info = None
    while pq:
        g, new_state, dist = heapq.heappop(pq)
        
        if(new_state == goal_state):
           fin_state, fin_state_info =  new_state, dist
           break
        
        explored.add(tuple(new_state))
        for successor in get_succ(new_state):
            
            if tuple(successor) not in explored:
                next_heuristic = get_manhattan_distance(successor)
                next_g = dist[0] + 1
                next_cost = next_g + next_heuristic
                heapq.heappush(pq,(next_cost, successor, (next_g, next_heuristic, moves)))
                
        parent_pointer[moves] = {"state": new_state, "info": dist}
        moves +=  1
        max_queue_len = max(len(pq), max_queue_len)
        
    sequence = []
    state, state_info = fin_state, fin_state_info
    while state_info[2] != -1:
        sequence.append((state, state_info))
        parent_entry = parent_pointer[state_info[2]]
        state, state_info = parent_entry["state"], parent_entry["info"]
    sequence.append((state, state_info))

    for state, state_info in sequence[::-1]:
        print(state, "h={}".format(state_info[1]), "moves: {}".format(state_info[0]))
    print("Max queue length: {}".format(max_queue_len))


if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2,5,1,4,0,6,7,0,3])
    print()

    print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    solve([5, 2, 3, 0, 6, 4, 7, 1, 0])
    print()
