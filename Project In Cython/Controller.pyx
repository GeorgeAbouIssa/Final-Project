# cython: language_level=3
import time

cdef class Controller:
    """
    Main controller for the connected matter agent.
    Coordinates different phases of problem solving.
    """
    cdef public:
        object agent  # Reference to the main agent
    
    def __init__(self, agent):
        """Initialize the controller with a reference to the agent"""
        self.agent = agent
    
    cpdef list execute_search(self, double time_limit=100):
        """
        Execute the full search process with the given time limit.
        
        Args:
            time_limit: Maximum time in seconds for the search
            
        Returns:
            List of states forming a path from start to goal
        """
        cdef double start_time, elapsed_time, remaining_time
        cdef double phase1_ratio, phase2_ratio
        cdef double phase1_time, phase2_time
        cdef list block_path, morph_path, sequential_path
        cdef object block_final_state
        cdef bint is_disconnected
        cdef int goal_components_count
        
        print(f"Starting search with time limit {time_limit} seconds")
        print(f"Current time: 2025-05-06 06:20:04, User: GeorgeAbouIssa")
        
        # Record start time
        start_time = time.time()
        
        # Check if we're dealing with a disconnected goal state
        goal_components_count = len(self.agent.find_disconnected_components(self.agent.goal_state))
        is_disconnected = goal_components_count > 1
        
        # If we have a disconnected goal, use that specialized approach
        if is_disconnected:
            print(f"Detected disconnected goal with {goal_components_count} components. Using specialized approach.")
            self.agent.is_goal_disconnected = True
            return self.agent.disconnected_goal.search_disconnected_goal(time_limit)
        
        # Standard two-phase approach for connected goals
        
        # Time allocation: use more time for block movement in obstacle-heavy environments,
        # otherwise prioritize morphing which is more complex
        if len(self.agent.obstacles) > 20:
            phase1_ratio = 0.4  # 40% of time for block movement
            phase2_ratio = 0.6  # 60% of time for morphing
        else:
            phase1_ratio = 0.2  # 20% of time for block movement
            phase2_ratio = 0.8  # 80% of time for morphing
        
        phase1_time = time_limit * phase1_ratio
        
        # Phase 1: Block Movement
        print(f"Starting Phase 1: Block Movement (time limit: {phase1_time:.2f}s)")
        block_path = self.agent.movement.block_movement_phase(phase1_time)
        
        if not block_path:
            print("Block movement phase failed!")
            return None
        
        # Calculate elapsed time and adjust phase 2 time if needed
        elapsed_time = time.time() - start_time
        remaining_time = time_limit - elapsed_time
        
        # If we used less time than allocated for phase 1, add the surplus to phase 2
        if elapsed_time < phase1_time:
            phase2_time = remaining_time
        else:
            # If we used more than allocated, adjust phase 2 time but ensure a minimum
            phase2_time = max(10.0, time_limit * phase2_ratio)
        
        print(f"Phase 1 completed in {elapsed_time:.2f}s, {len(block_path)-1} moves")
        print(f"Starting Phase 2: Morphing (time limit: {phase2_time:.2f}s)")
        
        # Get the final state from phase 1
        block_final_state = block_path[-1]
        
        # Phase 2: Morphing
        morph_path = self.agent.movement.smarter_morphing_phase(block_final_state, phase2_time)
        
        if not morph_path:
            print("Morphing phase failed!")
            return block_path
        
        # Check if goal was reached
        if morph_path[-1] == self.agent.goal_state:
            print("Goal reached!")
        else:
            print("Goal not exactly reached but returning best solution")
        
        # Combine paths (except first state of morph_path which is last state of block_path)
        morph_path = morph_path[1:]  # Remove first state to avoid duplication
        full_path = block_path + morph_path
        
        print(f"Full search completed in {time.time() - start_time:.2f}s, {len(full_path)-1} moves total")
        
        # Final verification to ensure no gaps in the path
        self.verify_path(full_path)
        
        return full_path
    
    cpdef bint verify_path(self, list path):
        """
        Verify that a path has no gaps (all consecutive states differ by valid moves)
        
        Args:
            path: List of states to verify
            
        Returns:
            True if path is valid, False otherwise
        """
        cdef int i
        cdef object state, next_state
        cdef list valid_next_states
        
        print("Verifying path consistency...")
        
        for i in range(len(path) - 1):
            state = path[i]
            next_state = path[i + 1]
            
            # Check if next_state is reachable from state
            valid_next_states = self.agent.movement.get_all_valid_moves(state)
            
            if next_state not in valid_next_states:
                print(f"WARNING: Invalid transition between state {i} and {i+1}")
                return False
        
        print("Path verification complete: all transitions are valid")
        return True
    
    cpdef double path_cost(self, list path):
        """
        Calculate the cost of a path (number of moves)
        
        Args:
            path: List of states
            
        Returns:
            Cost of the path
        """
        cdef int i
        cdef object state, next_state
        cdef int total_cost = 0
        
        for i in range(len(path) - 1):
            state = path[i]
            next_state = path[i + 1]
            
            # Count how many blocks moved
            state_set = set(state)
            next_state_set = set(next_state)
            
            # Blocks that moved must have either left their position or new blocks appeared
            changed_positions = len(state_set.symmetric_difference(next_state_set)) // 2
            total_cost += changed_positions
        
        return total_cost
    
    cpdef list optimize_path(self, list path):
        """
        Optimize a path by removing unnecessary moves
        
        Args:
            path: List of states
            
        Returns:
            Optimized path
        """
        cdef list optimized_path
        cdef int i, j
        
        if not path or len(path) <= 2:
            return path  # Nothing to optimize
        
        print("Optimizing path...")
        
        # Start with first state
        optimized_path = [path[0]]
        
        # Skip states that don't make progress toward the goal
        i = 0
        while i < len(path) - 1:
            current_state = path[i]
            
            # Look for the furthest state we can directly reach
            j = len(path) - 1
            while j > i + 1:
                # Check if state j is directly reachable from state i
                next_state = path[j]
                valid_next_states = self.agent.movement.get_all_valid_moves(current_state)
                
                if next_state in valid_next_states:
                    # Can reach state j directly, so add it and skip to j
                    optimized_path.append(next_state)
                    i = j
                    break
                
                j -= 1
            
            if j == i + 1:
                # Could not find a shortcut, so just add the next state
                optimized_path.append(path[i + 1])
                i += 1
        
        print(f"Path optimization: {len(path)} -> {len(optimized_path)} states")
        return optimized_path
    
    cpdef list check_connectivity(self, list path):
        """
        Verify that all states in the path maintain connectivity
        
        Args:
            path: List of states
            
        Returns:
            List of indices where connectivity is broken
        """
        cdef list issues = []
        cdef int i
        cdef set state_set
        
        print("Checking connectivity throughout path...")
        
        for i, state in enumerate(path):
            state_set = set(state)
            
            # Check if this state is connected
            if not self.agent.is_connected(state_set):
                print(f"WARNING: State {i} is not connected!")
                issues.append(i)
        
        if not issues:
            print("All states maintain connectivity")
        
        return issues
    
    cpdef list check_obstacle_overlap(self, list path):
        """
        Verify that no states in the path overlap with obstacles
        
        Args:
            path: List of states
            
        Returns:
            List of indices where obstacles are overlapped
        """
        cdef list issues = []
        cdef int i
        cdef tuple pos
        
        print("Checking for obstacle overlaps...")
        
        for i, state in enumerate(path):
            for pos in state:
                if pos in self.agent.obstacles:
                    print(f"ERROR: State {i} overlaps with obstacle at {pos}!")
                    issues.append(i)
                    break
        
        if not issues:
            print("No states overlap with obstacles")
        
        return issues
    
    cpdef list fix_path_issues(self, list path):
        """
        Fix issues in the path (connectivity issues, obstacle overlaps)
        
        Args:
            path: List of states
            
        Returns:
            Fixed path or None if no valid path can be found
        """
        cdef list connectivity_issues, obstacle_issues
        cdef int issue_idx, i
        cdef list new_path, repaired_path, valid_next_states
        cdef object current_state, next_state, best_state
        cdef double best_heuristic
        
        # Check for problems
        connectivity_issues = self.check_connectivity(path)
        obstacle_issues = self.check_obstacle_overlap(path)
        
        if not connectivity_issues and not obstacle_issues:
            print("No issues to fix in path")
            return path
        
        print(f"Fixing {len(connectivity_issues)} connectivity issues and {len(obstacle_issues)} obstacle issues...")
        
        # Start with the original path
        new_path = list(path)
        
        # Fix connectivity issues
        for issue_idx in connectivity_issues:
            # Try to repair the path at this point
            if issue_idx == 0:
                print("ERROR: Start state is not connected!")
                return None
            
            # Get the previous valid state
            prev_idx = issue_idx - 1
            while prev_idx > 0 and prev_idx in connectivity_issues:
                prev_idx -= 1
            
            current_state = new_path[prev_idx]
            
            # Try to find a valid next state
            valid_next_states = self.agent.movement.get_all_valid_moves(current_state)
            
            if not valid_next_states:
                print(f"No valid moves from state {prev_idx}")
                return None
            
            # Find the best next state using heuristic
            best_state = None
            best_heuristic = float('inf')
            
            for next_state in valid_next_states:
                h = self.agent.movement.improved_morphing_heuristic(next_state)
                if h < best_heuristic:
                    best_heuristic = h
                    best_state = next_state
            
            if best_state:
                new_path[issue_idx] = best_state
                print(f"Repaired connectivity issue at state {issue_idx}")
            else:
                print(f"Failed to repair connectivity issue at state {issue_idx}")
                return None
        
        # Fix obstacle issues
        for issue_idx in obstacle_issues:
            if issue_idx == 0:
                print("ERROR: Start state overlaps with obstacles!")
                return None
            
            # Get the previous valid state
            prev_idx = issue_idx - 1
            while prev_idx > 0 and (prev_idx in obstacle_issues or prev_idx in connectivity_issues):
                prev_idx -= 1
            
            current_state = new_path[prev_idx]
            
            # Try to find a valid next state
            valid_next_states = self.agent.movement.get_all_valid_moves(current_state)
            
            if not valid_next_states:
                print(f"No valid moves from state {prev_idx}")
                return None
            
            # Find the best next state using heuristic
            best_state = None
            best_heuristic = float('inf')
            
            for next_state in valid_next_states:
                # Skip if this state overlaps with obstacles
                overlap = False
                for pos in next_state:
                    if pos in self.agent.obstacles:
                        overlap = True
                        break
                
                if overlap:
                    continue
                
                h = self.agent.movement.improved_morphing_heuristic(next_state)
                if h < best_heuristic:
                    best_heuristic = h
                    best_state = next_state
            
            if best_state:
                new_path[issue_idx] = best_state
                print(f"Repaired obstacle issue at state {issue_idx}")
            else:
                print(f"Failed to repair obstacle issue at state {issue_idx}")
                return None
        
        # Verify that the path is now valid
        if self.check_connectivity(new_path) or self.check_obstacle_overlap(new_path):
            print("Failed to fix all issues in the path")
            return None
        
        print("All issues fixed successfully")
        
        # Make sure all transitions are valid
        repaired_path = [new_path[0]]
        for i in range(1, len(new_path)):
            current_state = repaired_path[-1]
            next_state = new_path[i]
            
            valid_next_states = self.agent.movement.get_all_valid_moves(current_state)
            
            if next_state in valid_next_states:
                repaired_path.append(next_state)
            else:
                # Find a valid path between current_state and next_state
                print(f"Finding valid transition between states {i-1} and {i}")
                
                # Run a small search to find a path
                from collections import deque
                
                frontier = deque([(current_state, [current_state])])
                visited = {current_state}
                max_depth = 5  # Limit search depth
                
                path_found = False
                while frontier and not path_found:
                    state, path_so_far = frontier.popleft()
                    
                    if len(path_so_far) > max_depth:
                        continue
                    
                    for neighbor in self.agent.movement.get_all_valid_moves(state):
                        if neighbor in visited:
                            continue
                        
                        new_path_so_far = path_so_far + [neighbor]
                        
                        if neighbor == next_state:
                            # Found a path to the next state
                            # Add all states in the path except the first one (already in repaired_path)
                            repaired_path.extend(new_path_so_far[1:])
                            path_found = True
                            break
                        
                        visited.add(neighbor)
                        frontier.append((neighbor, new_path_so_far))
                
                if not path_found:
                    print(f"Failed to find valid transition between states {i-1} and {i}")
                    # Just add the next state directly - it might be fixed in post-processing
                    repaired_path.append(next_state)
        
        return repaired_path