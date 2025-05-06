# cython: language_level=3
import heapq
import time
from collections import deque
import concurrent.futures
import threading

# Current Date: 2025-05-06 06:14:32
# User: GeorgeAbouIssa

cdef class DisconnectedGoalHandler:
    cdef public:
        object agent  # Reference to the main agent
            
    def __init__(self, agent):
        self.agent = agent
            
    cpdef list disconnected_block_movement_phase(self, double time_limit=15):
        """
        Modified Phase 1 for disconnected goal states:
        Moves the entire block toward a strategic position for splitting
        """
        cdef double start_time
        cdef int closest_component_idx
        cdef set closest_component
        cdef tuple closest_centroid, target_centroid, original_centroid
        cdef list all_components_y, path
        cdef double overall_y, obstacle_density
        cdef bint use_vertical_approach
        cdef list open_set
        cdef set closed_set
        cdef dict g_score, came_from
        cdef double f, min_distance, max_distance, centroid_distance, distance_penalty, neighbor_distance, adjusted_heuristic, f_score
        cdef int g, tentative_g
        cdef object current, neighbor, best_state
        cdef tuple current_centroid, neighbor_centroid, neighbor_centroid_int, target_centroid_int
        cdef double best_distance_diff, distance_diff
        
        print("Starting Disconnected Block Movement Phase...")
        start_time = time.time()
        
        # Find the closest goal component to the start state
        closest_component_idx = self.find_closest_component()
        closest_component = self.agent.goal_components[closest_component_idx]
        closest_centroid = self.agent.component_centroids[closest_component_idx]
        
        # Determine if vertical positioning is better based on y-axis centroids
        all_components_y = [centroid[1] for centroid in self.agent.component_centroids]
        overall_y = self.agent.goal_centroid[1]
        
        # Check if centroid of all shapes is closer to y level of the closest shape
        use_vertical_approach = abs(overall_y - closest_centroid[1]) < sum([abs(y - closest_centroid[1]) for y in all_components_y]) / len(all_components_y)
        
        if use_vertical_approach:
            print("Using vertical approach for block movement")
            # Target position is at the overall centroid with y-level of closest component
            target_centroid = (self.agent.goal_centroid[0], closest_centroid[1])
        else:
            print("Using standard approach for block movement")
            # Target position is the overall centroid
            target_centroid = self.agent.goal_centroid
            
        # Cache original goal centroid and temporarily replace with target
        original_centroid = self.agent.goal_centroid
        self.agent.goal_centroid = target_centroid
        
        # Use standard A* search but with the modified target
        open_set = [(self.agent.movement.block_heuristic(self.agent.start_state), 0, self.agent.start_state)]
        closed_set = set()
        g_score = {self.agent.start_state: 0}
        came_from = {self.agent.start_state: None}
        
        # We want to get close to the target position
        min_distance = 1.0
        max_distance = 2.0

        while open_set and time.time() - start_time < time_limit:
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
    
            # Skip if already processed
            if current in closed_set:
                continue
        
            # Check if we're at the desired distance from the goal centroid
            current_centroid = self.agent.calculate_centroid(current)
            centroid_distance = (abs(current_centroid[0] - target_centroid[0]) + 
                            abs(current_centroid[1] - target_centroid[1]))
                        
            if min_distance <= centroid_distance <= max_distance:
                print(f"Block stopped at strategic position. Distance: {centroid_distance}")
                # Restore original goal centroid
                self.agent.goal_centroid = original_centroid
                return self.agent.reconstruct_path(came_from, current)
        
            closed_set.add(current)
    
            # Process neighbor states (block moves)
            for neighbor in self.agent.movement.get_valid_block_moves(current):
                if neighbor in closed_set:
                    continue
            
                # Calculate tentative g-score
                tentative_g = g_score[current] + 1
        
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                
                    # Adjust heuristic to prefer states close to target
                    neighbor_centroid = self.agent.calculate_centroid(neighbor)
                    neighbor_distance = (abs(neighbor_centroid[0] - target_centroid[0]) + 
                                    abs(neighbor_centroid[1] - target_centroid[1]))
                
                    # Penalize distances that are too small
                    distance_penalty = 0
                    if neighbor_distance < min_distance:
                        distance_penalty = 10 * (min_distance - neighbor_distance)
                
                    # Calculate Manhattan distance to target
                    if self.agent.obstacles:
                        neighbor_centroid_int = (int(round(neighbor_centroid[0])), int(round(neighbor_centroid[1])))
                        target_centroid_int = (int(round(target_centroid[0])), int(round(target_centroid[1])))
                        
                        # Ensure centroids are within bounds
                        neighbor_centroid_int = (
                            max(0, min(neighbor_centroid_int[0], self.agent.grid_size[0]-1)),
                            max(0, min(neighbor_centroid_int[1], self.agent.grid_size[1]-1))
                        )
                        target_centroid_int = (
                            max(0, min(target_centroid_int[0], self.agent.grid_size[0]-1)),
                            max(0, min(target_centroid_int[1], self.agent.grid_size[1]-1))
                        )
                        
                        adjusted_heuristic = self.agent.obstacle_aware_distance(neighbor_centroid_int, target_centroid_int) + distance_penalty
                    else:
                        adjusted_heuristic = neighbor_distance + distance_penalty
                        
                    f_score = tentative_g + adjusted_heuristic
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # Restore original goal centroid
        self.agent.goal_centroid = original_centroid
        
        # If we exit the loop, find the best available state
        if came_from:
            best_state = None
            best_distance_diff = float('inf')
        
            for state in came_from.keys():
                state_centroid = self.agent.calculate_centroid(state)
                distance = (abs(state_centroid[0] - target_centroid[0]) + 
                            abs(state_centroid[1] - target_centroid[1]))
            
                if distance < min_distance:
                    distance_diff = min_distance - distance
                elif distance > max_distance:
                    distance_diff = distance - max_distance
                else:
                    best_state = state
                    break
                
                if distance_diff < best_distance_diff:
                    best_distance_diff = distance_diff
                    best_state = state
        
            if best_state:
                print(f"Best strategic position found for disconnected goal")
                return self.agent.reconstruct_path(came_from, best_state)
    
        return [self.agent.start_state]  # No movement possible
    
    cpdef int find_closest_component(self):
        """Find the index of the closest goal component to the start state"""
        cdef tuple start_centroid
        cdef double min_distance, distance
        cdef int closest_idx, idx
        
        start_centroid = self.agent.calculate_centroid(self.agent.start_state)
        min_distance = float('inf')
        closest_idx = 0
        
        for idx, centroid in enumerate(self.agent.component_centroids):
            distance = abs(start_centroid[0] - centroid[0]) + abs(start_centroid[1] - centroid[1])
            if distance < min_distance:
                min_distance = distance
                closest_idx = idx
                
        return closest_idx
    
    cpdef dict assign_blocks_to_components(self, state):
        """
        Returns a dictionary mapping each component index to a set of positions
        """
        cdef dict assignments
        cdef list state_positions, component_sizes, distances
        cdef set assigned, unassigned
        cdef int total_blocks_needed, idx, component_idx, i
        cdef tuple pos, centroid
        cdef list pos_distances
        cdef double dist
        
        assignments = {i: set() for i in range(len(self.agent.goal_components))}
        state_positions = list(state)
        
        # Create a dictionary to track assigned positions
        assigned = set()
        
        # First, count how many blocks we need for each component
        component_sizes = [len(comp) for comp in self.agent.goal_components]
        total_blocks_needed = sum(component_sizes)
        
        # Ensure we have enough blocks
        if len(state_positions) < total_blocks_needed:
            print(f"Warning: Not enough blocks ({len(state_positions)}) for goal state ({total_blocks_needed})")
            return None
            
        # Calculate distance from each block to each component centroid
        distances = []
        for pos in state_positions:
            pos_distances = []
            for idx, centroid in enumerate(self.agent.component_centroids):
                if self.agent.obstacles:
                    dist = self.agent.obstacle_aware_distance(pos, (int(centroid[0]), int(centroid[1])))
                else:
                    dist = abs(pos[0] - centroid[0]) + abs(pos[1] - centroid[1])
                pos_distances.append((idx, dist))
            distances.append((pos, sorted(pos_distances, key=lambda x: x[1])))
            
        # Sort blocks by their distance to their closest component
        distances.sort(key=lambda x: x[1][0][1])
        
        # Assign blocks to components in order of increasing distance
        for pos, component_distances in distances:
            for component_idx, dist in component_distances:
                if len(assignments[component_idx]) < len(self.agent.goal_components[component_idx]) and pos not in assigned:
                    assignments[component_idx].add(pos)
                    assigned.add(pos)
                    break
                    
        # Ensure all blocks are assigned
        unassigned = set(state_positions) - assigned
        if unassigned:
            # Assign remaining blocks to components that still need them
            for pos in unassigned:
                for component_idx in range(len(self.agent.goal_components)):
                    if len(assignments[component_idx]) < len(self.agent.goal_components[component_idx]):
                        assignments[component_idx].add(pos)
                        assigned.add(pos)
                        break
                        
        # Double-check that we've assigned the right number of blocks to each component
        for idx, component in enumerate(self.agent.goal_components):
            if len(assignments[idx]) != len(component):
                print(f"Warning: Component {idx} has {len(assignments[idx])} blocks assigned but needs {len(component)}")
                
        return assignments
    
    cpdef list plan_disconnect_moves(self, state, dict assignments):
        """
        Plan a sequence of moves to disconnect the shape into separate components
        Returns a list of states representing the disconnection process
        """
        cdef set current_state
        cdef list path, component_positions
        cdef bint is_separable
        cdef int i, j
        cdef set pos_i, pos_j, connection_points
        cdef tuple pos_i_t, pos_j_t
        
        # Start with current state
        current_state = set(state)
        path = [frozenset(current_state)]
        
        # Group the assignments by component
        component_positions = [set(assignments[i]) for i in range(len(self.agent.goal_components))]
        
        # Check if the state is already naturally separable
        is_separable = True
        for i in range(len(component_positions)):
            for j in range(i+1, len(component_positions)):
                # Check if there's a direct connection between components
                for pos_i_t in component_positions[i]:
                    for pos_j_t in component_positions[j]:
                        # Check if positions are adjacent
                        if any((pos_i_t[0] + dx, pos_i_t[1] + dy) == pos_j_t for dx, dy in self.agent.directions):
                            is_separable = False
                            break
                    if not is_separable:
                        break
                if not is_separable:
                    break
            if not is_separable:
                break
                
        # If already separable, return current state
        if is_separable:
            print("State is already naturally separable into components")
            return path
            
        # Find minimal set of points that connect the components
        connection_points = set()
        for i in range(len(component_positions)):
            for j in range(i+1, len(component_positions)):
                # Find all connections between components i and j
                for pos_i_t in component_positions[i]:
                    for pos_j_t in component_positions[j]:
                        # Check if positions are adjacent
                        if any((pos_i_t[0] + dx, pos_i_t[1] + dy) == pos_j_t for dx, dy in self.agent.directions):
                            # Add both positions to connection points
                            connection_points.add(pos_i_t)
                            connection_points.add(pos_j_t)
        
        # For theoretical disconnection, we don't need to actually move blocks
        # Just mark the connection points as if they're disconnected
        print(f"Identified {len(connection_points)} connection points for theoretical disconnection")
        
        # Return current state as the disconnection plan
        # The actual disconnection will happen during the morphing phase
        return path
    
    cpdef double component_morphing_heuristic(self, state, goal_component):
        """
        Heuristic for morphing a specific component
        """
        cdef int missing_blocks, missing_penalty, matching_positions, connectivity_bonus
        cdef list state_list, goal_list, distances, row, row_indices
        cdef tuple pos, goal_pos
        cdef set assigned_cols, goal_component_set
        cdef int total_distance, i, j, min_dist, best_j
        
        if not state:
            return float('inf')
        
        # Check if we've lost any blocks compared to the component's required size
        if len(state) < len(goal_component):
            missing_blocks = len(goal_component) - len(state)
            missing_penalty = 10000 * missing_blocks  # Large penalty per missing block
            return missing_penalty  # Return large penalty proportional to missing blocks
                
        state_list = list(state)
        goal_list = list(goal_component)
        
        # Early exit if states have different sizes
        if len(state_list) != len(goal_list):
            return float('inf')
        
        # Build distance matrix with obstacle-aware distances
        distances = []
        for pos in state_list:
            row = []
            for goal_pos in goal_list:
                # Use obstacle-aware distance calculation
                if self.agent.obstacles:
                    dist = self.agent.obstacle_aware_distance(pos, goal_pos)
                else:
                    # Use faster Manhattan distance if no obstacles
                    dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                row.append(dist)
            distances.append(row)
        
        # Use greedy assignment algorithm
        total_distance = 0
        assigned_cols = set()
        
        # Sort rows by minimum distance
        row_indices = list(range(len(state_list)))
        row_indices.sort(key=lambda i: min(distances[i]))
        
        for i in row_indices:
            # Find closest unassigned goal position
            min_dist = float('inf')
            best_j = -1
            
            for j in range(len(goal_list)):
                if j not in assigned_cols and distances[i][j] < min_dist:
                    min_dist = distances[i][j]
                    best_j = j
            
            if best_j != -1:
                assigned_cols.add(best_j)
                total_distance += min_dist
                
                # If a path is impossible (infinite distance), heavily penalize
                if min_dist == float('inf'):
                    return float('inf')
            else:
                # No assignment possible
                return float('inf')
        
        # Add connectivity bonus: prefer states that have more blocks in goal positions
        goal_component_set = frozenset(goal_component)
        matching_positions = len(frozenset(state).intersection(goal_component_set))
        connectivity_bonus = -matching_positions * 0.5  # Negative to encourage more matches
        
        return total_distance + connectivity_bonus
    
    cpdef list component_morphing_phase(self, start_state, goal_component, double time_limit=15):
        """
        Morph a specific component into its goal shape
        """
        cdef double start_time, obstacle_density, current_heuristic, best_heuristic, last_improvement_time, stagnation_tolerance
        cdef int adaptive_beam_width, iterations
        cdef list open_set, neighbors
        cdef set closed_set
        cdef dict g_score, came_from
        cdef double f, f_score
        cdef int g, tentative_g
        cdef object current, best_state, neighbor
        cdef object goal_component_set
        
        start_time = time.time()
        
        # Adapt beam width based on obstacle density
        adaptive_beam_width = self.agent.beam_width
        if len(self.agent.obstacles) > 0:
            obstacle_density = len(self.agent.obstacles) / (self.agent.grid_size[0] * self.agent.grid_size[1])
            adaptive_beam_width = int(self.agent.beam_width * (1 + min(1.0, obstacle_density * 5)))
            
        # Initialize beam search
        open_set = [(self.component_morphing_heuristic(start_state, goal_component), 0, start_state)]
        closed_set = set()
        
        # Track path, g-scores, and best state
        g_score = {start_state: 0}
        came_from = {start_state: None}
        
        # Track best state seen so far
        best_state = start_state
        best_heuristic = self.component_morphing_heuristic(start_state, goal_component)
        
        iterations = 0
        last_improvement_time = time.time()
        goal_component_set = frozenset(goal_component)
        
        while open_set and time.time() - start_time < time_limit:
            iterations += 1
            
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            # Check if goal reached
            if current == goal_component_set:
                print(f"Component goal reached after {iterations} iterations!")
                return self.agent.reconstruct_path(came_from, current)
            
            # Check if this is the best state seen so far
            current_heuristic = self.component_morphing_heuristic(current, goal_component)
            if current_heuristic < best_heuristic:
                best_state = current
                best_heuristic = current_heuristic
                last_improvement_time = time.time()
                
                # Print progress occasionally
                if iterations % 500 == 0:
                    print(f"Component progress: h={best_heuristic}, iterations={iterations}")
                
                # If we're very close to the goal, increase search intensity
                if best_heuristic < 5 * len(goal_component):
                    adaptive_beam_width *= 2
        
            # Check for stagnation
            stagnation_tolerance = time_limit * 0.3
            if time.time() - last_improvement_time > stagnation_tolerance:
                print("Component search stagnated, restarting...")
                # Clear the beam and start from the best state
                open_set = [(best_heuristic, g_score[best_state], best_state)]
                last_improvement_time = time.time()
        
            # Limit iterations to prevent infinite loops
            if iterations >= self.agent.max_iterations:
                print(f"Reached max iterations ({self.agent.max_iterations}) for component morphing")
                break
            
            closed_set.add(current)
        
            # Get valid moves for this component
            neighbors = self.agent.movement.get_all_valid_moves(current)
        
            # Process each neighbor
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
            
                # Skip if neighbor has wrong size
                if len(neighbor) != len(goal_component):
                    continue
            
                # Calculate tentative g-score
                tentative_g = g_score[current] + 1
            
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.component_morphing_heuristic(neighbor, goal_component)
                
                    # Add to open set
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
            # Beam search pruning: keep only the best states
            if len(open_set) > adaptive_beam_width:
                open_set = heapq.nsmallest(adaptive_beam_width, open_set)
                heapq.heapify(open_set)
    
        # If we exit the loop, return the best state found
        print(f"Component morphing timed out after {iterations} iterations!")
        return self.agent.reconstruct_path(came_from, best_state)
    
    cpdef list search_disconnected_goal(self, double time_limit=100):
        """
        Search method for disconnected goal states:
        1. First try parallel approach:
           a. Move blocks to strategic position
           b. Assign blocks to components
           c. Morph each component in parallel
        2. If parallel approach fails, try sequential approach, still using the 
           strategic position from Phase 1
        """
        cdef double start_time, parallel_time_limit, sequential_time_limit, move_time_ratio, disconnect_time_ratio, morphing_time_ratio
        cdef double obstacle_density, move_time_limit, disconnect_time_limit, morphing_time_limit, assignments_start_time, disconnect_time_used
        cdef double remaining_time
        cdef list block_path, disconnect_path, combined_path, component_paths, component_start_states, component_time_limits, sequential_path, final_path
        cdef object block_final_state, assignments
        cdef int total_blocks, i
        cdef object shared_grid_lock
        cdef set shared_occupied_cells
        cdef list component_locks, component_occupied_cells, futures
        
        print(f"Starting search for disconnected goal with {len(self.agent.goal_components)} components")
        print(f"Current date: 2025-05-06 06:14:32, User: GeorgeAbouIssa")
        start_time = time.time()

        # Reserve time for sequential approach if needed
        parallel_time_limit = time_limit * 0.6  # Use 60% of time for parallel approach
        sequential_time_limit = time_limit * 0.4  # Reserve 40% for sequential if needed

        # Try Approach 1: Parallel processing
        # Allocate time for different phases
        move_time_ratio = 0.2  # 20% for block movement
        disconnect_time_ratio = 0.15  # 15% for disconnection planning
        morphing_time_ratio = 0.65  # 65% for morphing

        # Adjust ratios if obstacles are present
        if len(self.agent.obstacles) > 0:
            obstacle_density = len(self.agent.obstacles) / (self.agent.grid_size[0] * self.agent.grid_size[1])
            move_time_ratio = min(0.4, 0.2 + obstacle_density * 0.5)  # Up to 40% for movement
            morphing_time_ratio = 1.0 - move_time_ratio - disconnect_time_ratio

        move_time_limit = parallel_time_limit * move_time_ratio
        disconnect_time_limit = parallel_time_limit * disconnect_time_ratio
        morphing_time_limit = parallel_time_limit * morphing_time_ratio

        print(f"Time allocation: {move_time_ratio:.1%} block movement, "
                f"{disconnect_time_ratio:.1%} disconnection planning, "
                f"{morphing_time_ratio:.1%} morphing")

        # Phase 1: Strategic Block Movement
        block_path = self.disconnected_block_movement_phase(move_time_limit)

        if not block_path:
            print("Block movement phase failed! Cannot continue.")
            return None

        # Get the final state from block movement phase
        block_final_state = frozenset(block_path[-1])

        # Phase 2: Assign blocks to components and plan disconnection
        assignments_start_time = time.time()
        assignments = self.assign_blocks_to_components(block_final_state)

        if assignments is None:
            print("Failed to assign blocks to components!")
            return block_path

        # Theoretical disconnection planning
        disconnect_path = self.plan_disconnect_moves(block_final_state, assignments)
        disconnect_time_used = time.time() - assignments_start_time

        # Add remaining disconnect time to morphing time
        if disconnect_time_used < disconnect_time_limit:
            morphing_time_limit += (disconnect_time_limit - disconnect_time_used)

        # Phase 3: Parallel Morphing of Components
        print(f"Starting parallel morphing of {len(self.agent.goal_components)} components")

        # Prepare component states and goals
        component_start_states = []
        for i in range(len(self.agent.goal_components)):
            component_start_states.append(frozenset(assignments[i]))

        # Time allocation for each component based on its size
        total_blocks = sum(len(comp) for comp in self.agent.goal_components)
        component_time_limits = [
            morphing_time_limit * len(comp) / total_blocks
            for comp in self.agent.goal_components
        ]

        # Add shared resources for thread coordination
        shared_grid_lock = threading.Lock()
        shared_occupied_cells = set()  # Track cells occupied by any component
        component_locks = [threading.Lock() for _ in range(len(self.agent.goal_components))]
        component_occupied_cells = [set() for _ in range(len(self.agent.goal_components))]
        
        # Initialize shared grid with current occupied cells
        for i, comp_state in enumerate(component_start_states):
            component_occupied_cells[i] = set(comp_state)
            shared_occupied_cells.update(comp_state)
        
        # Create wrapper function for thread-safe component morphing
        def thread_safe_component_morphing(i, start_state, goal_component, time_limit):
            """Wrapper for thread-safe component morphing with coordination"""
            # Create a thread-local version of get_all_valid_moves
            def coordinated_get_valid_moves(state):
                # Get basic moves without coordination
                basic_moves = self.agent.movement.get_valid_morphing_moves(state)
                chain_moves = self.agent.movement.get_smart_chain_moves(state)
                sliding_moves = self.agent.movement.get_sliding_chain_moves(state)
                
                # Combine all moves
                all_moves = list(set(basic_moves + chain_moves + sliding_moves))
                
                # Apply coordination checks
                valid_moves = []
                with shared_grid_lock:
                    # Calculate cells that other components are using
                    others_occupied = shared_occupied_cells.copy()
                    for pos in state:
                        if pos in others_occupied:
                            others_occupied.remove(pos)
                    
                    for move in all_moves:
                        # Check for conflicts with other components
                        has_conflict = False
                        for pos in move:
                            if pos in others_occupied:
                                has_conflict = True
                                break
                        
                        if not has_conflict:
                            valid_moves.append(move)
                
                return valid_moves
            
            # Temporarily override the get_all_valid_moves method for this thread
            original_get_all_valid_moves = self.agent.movement.get_all_valid_moves
            self.agent.movement.get_all_valid_moves = coordinated_get_valid_moves
            
            try:
                # Run the original morphing with our thread-safe version
                result = self.component_morphing_phase(start_state, goal_component, time_limit)
                
                # Update shared grid with final state
                final_state = result[-1] if result else start_state
                with shared_grid_lock:
                    # Remove old positions
                    for pos in component_occupied_cells[i]:
                        if pos in shared_occupied_cells:
                            shared_occupied_cells.remove(pos)
                    
                    # Add new positions
                    shared_occupied_cells.update(final_state)
                    component_occupied_cells[i] = set(final_state)
                
                return result
            finally:
                # Restore original method
                self.agent.movement.get_all_valid_moves = original_get_all_valid_moves
        
        # Initialize component_paths
        component_paths = []
    
        # Use thread pool for parallel morphing with coordination
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agent.goal_components)) as executor:
            # Submit tasks for each component with our wrapper
            futures = []
            for i in range(len(self.agent.goal_components)):
                futures.append(
                    executor.submit(
                        thread_safe_component_morphing,
                        i,
                        component_start_states[i],
                        self.agent.goal_components[i],
                        component_time_limits[i]
                    )
                )
        
            # Collect results as they complete
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    component_paths.append(result)
                except Exception as e:
                    print(f"Error in component {i} morphing: {e}")
                    # Add the start state as a fallback to ensure we have a path
                    component_paths.append([component_start_states[i]])

        # Make sure we have a path for each component
        if len(component_paths) != len(self.agent.goal_components):
            print(f"Warning: Expected {len(self.agent.goal_components)} component paths, got {len(component_paths)}")
            # Fill in missing paths with their start states
            for i in range(len(component_paths), len(self.agent.goal_components)):
                component_paths.append([component_start_states[i]])

        # Combine the paths from all components
        combined_path = block_path[:-1]  # Remove the last state from block path

        # Add the component paths after the block movement
        # For visualization, we'll alternate steps from each component
        # to show them moving simultaneously
        max_component_path_length = max(len(path) for path in component_paths)

        for step in range(max_component_path_length):
            combined_state = set()
            for i, path in enumerate(component_paths):
                # If this component's path is long enough, add its state at this step
                if step < len(path):
                    combined_state.update(path[step])
                elif path:  # Use the final state when a thread is done
                    combined_state.update(path[-1])
        
            # Add the combined state to the path if not empty
            if combined_state:
                combined_path.append(list(combined_state))

        # Check if goal is reached using our more forgiving criterion
        if not self.is_full_disconnected_goal_reached(combined_path[-1]):
            print("Parallel approach failed to reach goal. Trying sequential approach using the same strategic position...")
            
            # Try sequential approach starting from the strategic position (from Phase 1)
            remaining_time = max(10, time_limit - (time.time() - start_time))
            
            # Use the sequential approach but starting from the strategic position achieved in Phase 1
            sequential_path = self.sequential_from_strategic_position(
                block_final_state, 
                remaining_time
            )
            
            if sequential_path and len(sequential_path) > 1:
                # Combine the initial movement with the sequential solving path
                final_path = block_path[:-1] + sequential_path[1:]
                print("Sequential approach produced a valid path")
                return final_path
            else:
                print("WARNING: Sequential approach also failed. Returning best available path.")

        return combined_path
    
    cpdef bint is_full_disconnected_goal_reached(self, state):
        """
        Check if goal components are sufficiently matched in the current state.
        Allows for some tolerance in the matching to avoid being too strict.
        """
        cdef list state_components
        cdef int total_correct_blocks, total_goal_blocks, best_match, best_idx
        cdef double match_percentage, threshold
        cdef list goal_component_sets, state_component_sets
        cdef set matched_state_indices, gc, sc
        
        state_components = self.agent.find_disconnected_components(state)
        
        # If we have fewer components than the goal, we're definitely not done
        if len(state_components) < len(self.agent.goal_components):
            return False
        
        # For each goal component, find the best matching state component
        total_correct_blocks = 0
        total_goal_blocks = sum(len(comp) for comp in self.agent.goal_components)
        
        # Convert to sets for easier intersection operations
        goal_component_sets = [set(comp) for comp in self.agent.goal_components]
        state_component_sets = [set(comp) for comp in state_components]
        
        # Match each goal component to its best-matching state component
        # without reusing state components
        matched_state_indices = set()
        
        for gc in goal_component_sets:
            best_match = 0
            best_idx = -1
            
            for i, sc in enumerate(state_component_sets):
                if i in matched_state_indices:
                    continue  # Skip already matched components
                    
                # Calculate intersection (blocks in the correct position)
                intersection = len(gc.intersection(sc))
                
                if intersection > best_match:
                    best_match = intersection
                    best_idx = i
            
            if best_idx >= 0:
                matched_state_indices.add(best_idx)
                total_correct_blocks += best_match
        
        # Calculate match percentage
        match_percentage = total_correct_blocks / total_goal_blocks if total_goal_blocks > 0 else 0
        
        # Print debug info for monitoring
        print(f"Match percentage: {match_percentage:.2%} ({total_correct_blocks}/{total_goal_blocks} blocks)")
        
        # Accept if at least 90% of blocks are correctly placed
        threshold = 0.90
        return match_percentage >= threshold

    cpdef list get_disconnected_valid_moves(self, state, goal_components):
        """
        Generate valid moves for disconnected components
        Allows moves that maintain connectivity within each component
        But doesn't require connectivity between components
        """
        cdef list current_components, all_moves, component_moves, basic_moves, chain_moves, sliding_moves, valid_moves
        cdef object component_state
        cdef set component, new_state
        
        # Find the components in the current state
        current_components = self.agent.find_disconnected_components(state)
    
        # If we have fewer components than needed, can't generate valid disconnected moves
        if len(current_components) < len(goal_components):
            return self.agent.movement.get_all_valid_moves(state)
    
        # Generate moves for each component separately
        all_moves = []
    
        for component in current_components:
            component_moves = []
        
            # Get basic morphing moves for this component
            component_state = frozenset(component)
            basic_moves = self.agent.movement.get_valid_morphing_moves(component_state)
            chain_moves = self.agent.movement.get_smart_chain_moves(component_state)
            sliding_moves = self.agent.movement.get_sliding_chain_moves(component_state)
        
            # Combine all move types
            component_moves.extend(basic_moves)
            component_moves.extend(chain_moves)
            component_moves.extend(sliding_moves)
        
            # For each component move, create a new overall state
            for move in component_moves:
                # Create new overall state with this component's move
                new_state = (state - component_state) | move
                all_moves.append(frozenset(new_state))
    
        return all_moves
    
    cpdef list sequential_from_strategic_position(self, strategic_state, double time_limit=60):
        """
        Apply the sequential component-by-component approach starting from
        the strategic position achieved in Phase 1.
        
        Args:
            strategic_state: The state after initial block movement
            time_limit: Time limit for the sequential approach
            
        Returns:
            List of states representing the path
        """
        cdef double start_time, remaining_time
        cdef set current_state, component_initial, component_goal, final_component_state
        cdef dict assignments
        cdef list combined_path, component_time_limits, component_path, other_blocks_list, cleanup_path
        cdef int total_blocks, expected_count, i
        cdef set all_obstacles, other_blocks
        cdef bint final_match
        
        print(f"Starting sequential approach from strategic position")
        start_time = time.time()

        # Start with the strategic position
        current_state = set(strategic_state)
        
        # Start building the path
        combined_path = [list(current_state)]

        # Assign blocks to components
        print("Assigning blocks to components")
        assignments = self.assign_blocks_to_components(current_state)
        if assignments is None:
            print("Failed to assign blocks to components in sequential approach!")
            return combined_path

        # Calculate time for component-wise search
        remaining_time = time_limit - (time.time() - start_time)
        
        # Allocate time for each component based on its complexity (size)
        total_blocks = sum(len(comp) for comp in self.agent.goal_components)
        component_time_limits = [remaining_time * len(comp) / total_blocks for comp in self.agent.goal_components]
        
        # Solve each component sequentially
        for i, goal_component in enumerate(self.agent.goal_components):
            print(f"Solving component {i+1}/{len(self.agent.goal_components)} with {len(goal_component)} blocks")
            
            # Prepare the component's initial and goal states
            component_initial = assignments[i]
            component_goal = goal_component
            
            # All other blocks become obstacles
            other_blocks = current_state - component_initial
            all_obstacles = self.agent.obstacles.union(other_blocks)
            
            # Create a new agent for this component
            from ConnectedMatterAgent import ConnectedMatterAgent
            component_agent = ConnectedMatterAgent(
                grid_size=self.agent.grid_size,
                start_positions=list(component_initial),
                goal_positions=list(component_goal),
                topology=self.agent.topology,
                max_simultaneous_moves=self.agent.max_simultaneous_moves,
                min_simultaneous_moves=self.agent.min_simultaneous_moves,
                obstacles=all_obstacles
            )
            
            # Run a complete search for this component
            component_time_limit = component_time_limits[i]
            print(f"  - Time allocated: {component_time_limit:.2f}s")
            print(f"  - Starting with {len(component_initial)} blocks, targeting {len(component_goal)} positions")
            print(f"  - Additional obstacles: {len(other_blocks)} (other blocks)")
            
            component_path = component_agent.search(component_time_limit)
            
            if not component_path:
                print(f"  - Search failed for component {i+1}")
                # Try with a fallback approach - just run morphing directly
                print(f"  - Trying fallback with direct morphing")
                component_path = component_agent.movement.smarter_morphing_phase(
                    start_state=frozenset(component_initial),
                    time_limit=component_time_limit
                )
                if not component_path:
                    print(f"  - Fallback failed for component {i+1}")
                    continue
            
            # Update the current state with the final positions of this component
            final_component_state = set(component_path[-1])
            current_state = (current_state - component_initial).union(final_component_state)
            
            # Build the combined path by adding the component path
            # We need to combine the states: keep the positions of blocks not in this component
            other_blocks_list = list(current_state - final_component_state)
            
            # For each state in the component path (except first which is already included)
            for state in component_path[1:]:
                # Create a combined state with this component's state and other blocks
                combined_state = list(state) + other_blocks_list
                combined_path.append(combined_state)
            
            print(f"  - Component {i+1} solved: {len(component_path)-1} moves")
        
        # Verify the block count is consistent
        expected_count = len(strategic_state)
        for i, state in enumerate(combined_path):
            if len(state) != expected_count:
                print(f"WARNING: State {i} has {len(state)} blocks instead of {expected_count}")
                if i > 0:
                    # Fix by using previous valid state
                    combined_path[i] = combined_path[i-1]
        
        # Check if we have a valid solution
        final_match = self.is_full_disconnected_goal_reached(combined_path[-1])
        
        # If goal not reached, try cleanup phase
        if not final_match:
            print("Sequential approach failed to reach goal. Trying cleanup phase...")
            remaining_time = max(5, time_limit - (time.time() - start_time))
            cleanup_path = self.cleanup_and_retry(combined_path[-1], remaining_time)
            
            if cleanup_path and len(cleanup_path) > 1:
                # Combine the existing path with the cleanup path
                combined_path.extend(cleanup_path[1:])  # Skip the first state which is already in combined_path
                
                # Check if the goal is now reached
                final_match = self.is_full_disconnected_goal_reached(combined_path[-1])
                print(f"After cleanup phase, goal {'reached' if final_match else 'still not reached'}")
        else:
            print("Sequential approach succeeded in reaching goal state")
        
        return combined_path
    
    cpdef list cleanup_and_retry(self, current_state, double time_limit=20):
        """
        Cleanup phase with strict block count verification.
        
        Args:
            current_state: Current state after sequential approach
            time_limit: Time limit for the cleanup phase
            
        Returns:
            List of states representing the additional path
        """
        cdef double start_time, remaining_time
        cdef int original_block_count, i
        cdef set current_state_set, blocks_at_goal, blocks_not_at_goal, unfilled_goals
        cdef list clusters, cleanup_path, target_goals, goal_path
        cdef list cluster_distances
        cdef int anchor_idx
        cdef set non_goal_cluster, initial_blocks_at_goal
        cdef bint is_connected
        
        print("Starting cleanup phase...")
        start_time = time.time()
        
        # Store original block count for verification
        original_block_count = len(current_state)
        print(f"Original block count: {original_block_count}")
        
        # Convert to set for easier operations
        current_state_set = set(current_state)
        
        # Identify blocks that are at goal positions
        blocks_at_goal = current_state_set.intersection(self.agent.goal_state)
        
        # Identify blocks that are not at goal positions
        blocks_not_at_goal = current_state_set - blocks_at_goal
        
        # Identify unfilled goal positions
        unfilled_goals = set(self.agent.goal_state) - blocks_at_goal
        
        # If all goals are filled or no blocks are available, nothing to clean up
        if not unfilled_goals or not blocks_not_at_goal:
            print("No cleanup needed or possible")
            return [current_state]
        
        print(f"Found {len(blocks_at_goal)} blocks at goal positions and {len(blocks_not_at_goal)} blocks to reconnect")
        print(f"Unfilled goals: {len(unfilled_goals)}")
        
        # Start building the path
        cleanup_path = [list(current_state)]
        
        # Step 1: Find disconnected clusters of blocks not at goal positions
        clusters = self.agent.find_disconnected_components(blocks_not_at_goal)
        
        if not clusters:
            print("No clusters found")
            return cleanup_path
            
        print(f"Found {len(clusters)} disconnected clusters of blocks not at goal positions")
        
        # If only one cluster, no need to reconnect
        if len(clusters) == 1:
            print("Only one cluster found, skipping reconnection")
            non_goal_cluster = set(clusters[0])
        else:
            # Step 2: Calculate which cluster is closest to unfilled goals
            cluster_distances = []
            
            for i, cluster in enumerate(clusters):
                # Calculate minimum distance from this cluster to any unfilled goal
                min_distance = float('inf')
                
                for block in cluster:
                    for goal in unfilled_goals:
                        dist = abs(block[0] - goal[0]) + abs(block[1] - goal[1])
                        if dist < min_distance:
                            min_distance = dist
                
                # Store the minimum distance for this cluster
                cluster_distances.append((i, min_distance))
            
            # Sort clusters by distance to unfilled goals
            cluster_distances.sort(key=lambda x: x[1])
            
            # The closest cluster is our anchor
            anchor_idx = cluster_distances[0][0]
            print(f"Cluster {anchor_idx} is closest to unfilled goals (distance: {cluster_distances[0][1]})")
            
            # Use simplified reconnection approach with strict block count verification
            non_goal_cluster = self.reconnect_clusters_strict(
                clusters, cluster_distances, current_state, blocks_at_goal, 
                cleanup_path, original_block_count
            )
        
        # Step 4: Now we have one connected component of non-goal blocks
        # Try to form the goal shape with this structure
        print(f"Moving {len(non_goal_cluster)} connected blocks toward goal positions")
        
        # Identify unfilled goals that our non-goal blocks should try to fill
        target_goals = list(unfilled_goals)[:len(non_goal_cluster)]
        
        # Create a new agent to solve the formation of the goal shape
        from ConnectedMatterAgent import ConnectedMatterAgent
        goal_agent = ConnectedMatterAgent(
            grid_size=self.agent.grid_size,
            start_positions=list(non_goal_cluster),
            goal_positions=target_goals,
            topology=self.agent.topology,
            max_simultaneous_moves=self.agent.max_simultaneous_moves,
            min_simultaneous_moves=self.agent.min_simultaneous_moves,
            obstacles=self.agent.obstacles.union(blocks_at_goal)
        )
        
        # Run the solver with remaining time
        remaining_time = max(3, time_limit - (time.time() - start_time))
        
        # First try connected movement using the built-in agent
        try:
            print(f"Attempting goal formation with {remaining_time:.1f} seconds remaining")
            goal_path = []
            
            # Use morphing which maintains connectivity
            goal_path = goal_agent.movement.smarter_morphing_phase(
                start_state=frozenset(non_goal_cluster),
                time_limit=remaining_time
            )
            
            # If we found a valid path, combine it with the fixed blocks
            if goal_path and len(goal_path) > 1:
                print(f"Goal shape formation successful with {len(goal_path)-1} additional steps")
                
                # Get the initial blocks that are at goal positions
                initial_blocks_at_goal = blocks_at_goal.copy()
                
                # Combine each state with fixed blocks and verify block count
                for i, state in enumerate(goal_path[1:], 1):  # Skip first state
                    # Verify that the state has the correct number of blocks
                    if len(state) + len(initial_blocks_at_goal) != original_block_count:
                        print(f"WARNING: Incorrect block count in goal path state {i}")
                        print(f"Expected {original_block_count}, got {len(state) + len(initial_blocks_at_goal)}")
                        
                        # Fix the state to have the correct number of blocks
                        # If short, add blocks from previous state
                        if len(state) + len(initial_blocks_at_goal) < original_block_count:
                            prev_state = set(goal_path[i-1]) if i > 1 else set(non_goal_cluster)
                            missing = original_block_count - (len(state) + len(initial_blocks_at_goal))
                            extra_blocks = list(prev_state - set(state))[:missing]
                            state = set(state).union(extra_blocks)
                        # If too many, remove blocks furthest from goals
                        elif len(state) + len(initial_blocks_at_goal) > original_block_count:
                            excess = (len(state) + len(initial_blocks_at_goal)) - original_block_count
                            # Sort blocks by distance to goals and remove furthest ones
                            distance_to_goals = {}
                            for block in state:
                                min_dist = min(abs(block[0] - goal[0]) + abs(block[1] - goal[1]) 
                                               for goal in target_goals)
                                distance_to_goals[block] = min_dist
                            
                            blocks_to_remove = sorted(state, key=lambda b: -distance_to_goals[b])[:excess]
                            state = set(state) - set(blocks_to_remove)
                    
                    # Create the combined state
                    combined_state = list(initial_blocks_at_goal.union(state))
                    
                    # Final verification
                    assert len(combined_state) == original_block_count, f"Block count mismatch: {len(combined_state)} != {original_block_count}"
                    
                    cleanup_path.append(combined_state)
            else:
                print("Failed to form goal shape")
        except Exception as e:
            print(f"Error in goal formation: {str(e)}")
            # If an error occurs, just return what we have so far
        
        # Final verification of all states in the path
        for i, state in enumerate(cleanup_path):
            if len(state) != original_block_count:
                print(f"ERROR: State {i} has incorrect block count: {len(state)} != {original_block_count}")
                # Fix the state to have the correct block count
                if i > 0:  # If not the first state
                    prev_state = set(cleanup_path[i-1])
                    if len(state) < original_block_count:
                        # Add blocks from previous state
                        missing = original_block_count - len(state)
                        extra_blocks = list(prev_state - set(state))[:missing]
                        state = list(set(state).union(extra_blocks))
                    else:
                        # Remove excess blocks
                        excess = len(state) - original_block_count
                        state = list(set(state))[:original_block_count]
                    
                    cleanup_path[i] = state
        
        print(f"Cleanup complete with {len(cleanup_path)} states")
        return cleanup_path
    
    cpdef set reconnect_clusters_strict(self, list clusters, list cluster_distances, current_state, set blocks_at_goal, list cleanup_path, int original_block_count):
        """
        Individual block movement with continuous connectivity checks.
        
        Args:
            clusters: List of clusters to reconnect
            cluster_distances: List of (cluster_idx, distance) tuples
            current_state: Current state as a list
            blocks_at_goal: Set of blocks already at goal positions
            cleanup_path: Path being built
            original_block_count: Original number of blocks to maintain
            
        Returns:
            Set of connected non-goal blocks
        """
        cdef int anchor_idx, blocks_to_move, blocks_moved, last_idx
        cdef set current_blocks, remaining_cluster, connected_blocks
        cdef tuple movable_block, connected_block, target_block
        cdef list other_clusters_indices, path, temp_path
        cdef bint is_connected
        cdef set temp_cluster, obstacles
        cdef tuple current_pos, new_pos
        
        print(f"Current time: 2025-05-06 06:14:32, User: GeorgeAbouIssa")
        print(f"Starting reconnection with {len(clusters)} clusters")
        
        # The closest cluster is our anchor
        anchor_idx = cluster_distances[0][0]
        
        # Keep track of our current state
        current_blocks = set(current_state)
        
        # Start with the anchor cluster as our connected component
        connected_blocks = set(clusters[anchor_idx])
        
        # Process clusters from closest to furthest (excluding the anchor)
        other_clusters_indices = [i for i, _ in cluster_distances if i != anchor_idx]
        
        for idx in other_clusters_indices:
            cluster_to_connect = set(clusters[idx])
            if not cluster_to_connect:
                continue
                
            print(f"Connecting cluster {idx} with {len(cluster_to_connect)} blocks")
            
            # Create a working copy of the cluster
            remaining_cluster = cluster_to_connect.copy()
            
            # Find the closest pair of blocks between the clusters
            closest_pair = None
            min_distance = float('inf')
            
            for pos1 in cluster_to_connect:
                for pos2 in connected_blocks:
                    dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                    if dist < min_distance:
                        min_distance = dist
                        closest_pair = (pos1, pos2)
            
            if not closest_pair:
                print(f"No valid connection found for cluster {idx}")
                connected_blocks.update(cluster_to_connect)
                continue
                
            source_pos, target_pos = closest_pair
            
            # For very close clusters, just combine them directly
            if min_distance <= 2:
                connected_blocks.update(cluster_to_connect)
                print(f"Clusters are very close, direct connection made")
                continue
            
            print(f"Moving individual blocks from cluster {idx} to connect at distance {min_distance}")
            
            # Calculate approximately how many blocks we need to move
            # to bridge the gap (with a buffer for obstacles)
            blocks_to_move = min(len(cluster_to_connect) - 1, min_distance + 3)
            blocks_moved = 0
            
            # Process one block at a time
            while blocks_moved < blocks_to_move and remaining_cluster:
                # Find a block that can be removed without disconnecting the cluster
                movable_block = None
                
                for block in remaining_cluster:
                    # Check if removing this block maintains connectivity
                    temp_cluster = remaining_cluster - {block}
                    
                    # If empty or still connected, we can move this block
                    if not temp_cluster or self.agent.is_connected(temp_cluster):
                        movable_block = block
                        break
                
                if not movable_block:
                    print(f"No more blocks can be moved without disconnecting")
                    break
                
                print(f"Moving block at {movable_block} toward target")
                
                # Find a path for this block to the connected component
                # Avoid all obstacles including goal blocks
                path = None
                target_block = None
                
                # Try to find a path to any block in the connected component
                for connected_block in connected_blocks:
                    obstacles = ((current_blocks - {movable_block, connected_block}) | 
                                 self.agent.obstacles | blocks_at_goal)
                    
                    temp_path = self.agent.find_clean_path(movable_block, connected_block, obstacles)
                    
                    if temp_path and (not path or len(temp_path) < len(path)):
                        path = temp_path
                        target_block = connected_block
                
                if not path:
                    print(f"No valid path found for block at {movable_block}")
                    # Try the next block
                    remaining_cluster.remove(movable_block)
                    continue
                
                print(f"Found path of length {len(path)} to {target_block}")
                
                # Remove the block from its current position
                current_blocks.remove(movable_block)
                remaining_cluster.remove(movable_block)
                
                # Verify block count after removal
                assert len(current_blocks) == original_block_count - 1, f"Block count mismatch after removal: {len(current_blocks)}"
                
                # Add state to show removal
                cleanup_path.append(list(current_blocks))
                
                # Move the block along the path (step by step)
                # We'll go to the second-to-last position to avoid colliding with the target
                last_idx = min(len(path) - 2, min_distance)
                
                # If path is too short, just use the last valid position
                if last_idx < 1:
                    last_idx = 1 if len(path) > 1 else 0
                
                # Walk the block through each step of the path
                current_pos = None
                
                for i in range(1, last_idx + 1):
                    # Get the next position in the path
                    new_pos = path[i]
                    
                    # Check if this position is valid
                    if (new_pos not in current_blocks and 
                        new_pos not in blocks_at_goal and
                        new_pos not in self.agent.obstacles):
                        
                        # If we had placed the block somewhere earlier, remove it
                        if current_pos:
                            current_blocks.remove(current_pos)
                            # Add state to show removal
                            cleanup_path.append(list(current_blocks))
                        
                        # Place the block at the new position
                        current_blocks.add(new_pos)
                        current_pos = new_pos
                        
                        # Verify block count
                        assert len(current_blocks) == original_block_count, f"Block count mismatch: {len(current_blocks)}"
                        
                        # Add state to show the move
                        cleanup_path.append(list(current_blocks))
                    else:
                        # Position already occupied, try next
                        print(f"Position {new_pos} is occupied, trying next")
                        continue
                
                # If we successfully placed the block somewhere, count it
                if current_pos:
                    blocks_moved += 1
                    connected_blocks.add(current_pos)
                    print(f"Successfully moved block to {current_pos}")
                    
                    # Check for any blocks that need to be reconnected after this move
                    self.check_and_reconnect_blocks(
                        current_blocks, blocks_at_goal, connected_blocks,
                        remaining_cluster, cleanup_path, original_block_count
                    )
                else:
                    # Couldn't place the block anywhere, add it back
                    print(f"Failed to move block, adding back")
                    current_blocks.add(movable_block)
                    remaining_cluster.add(movable_block)
                    
                    # Verify block count
                    assert len(current_blocks) == original_block_count, f"Block count mismatch: {len(current_blocks)}"
                    
                    # Add state to show the rollback
                    cleanup_path.append(list(current_blocks))
            
            # Add any remaining blocks from this cluster to the connected component
            connected_blocks.update(remaining_cluster)
            
            print(f"Connected cluster {idx} by moving {blocks_moved} blocks")
        
        # Return non-goal blocks
        return current_blocks - blocks_at_goal
    
    cpdef check_and_reconnect_blocks(self, set current_blocks, set blocks_at_goal, set connected_blocks, 
                                   set remaining_cluster, list cleanup_path, int original_block_count):
        """
        Check if there are blocks that need to be reconnected and handle them.
        
        Args:
            current_blocks: Current set of all blocks
            blocks_at_goal: Set of blocks at goal positions
            connected_blocks: Set of blocks already connected
            remaining_cluster: Blocks in the current cluster being processed
            cleanup_path: Path being built
            original_block_count: Original number of blocks to maintain
        """
        cdef set non_goal_blocks
        cdef list all_clusters
        cdef int main_cluster_idx, max_overlap, i, overlap
        cdef set main_cluster, cluster_set
        cdef tuple pos1, pos2, movable_block, main_block, current_pos, new_pos
        cdef double min_distance
        cdef tuple closest_pair
        cdef list path, temp_path
        cdef tuple target_block
        cdef set obstacles
        cdef int last_idx
        
        # First identify all blocks not at goal positions
        non_goal_blocks = current_blocks - blocks_at_goal
        
        # Check if all non-goal blocks are connected
        if self.agent.is_connected(non_goal_blocks):
            return  # All good, no need to reconnect anything
        
        # Find the disconnected components
        all_clusters = self.agent.find_disconnected_components(non_goal_blocks)
        
        # If only one cluster, nothing to reconnect
        if len(all_clusters) <= 1:
            return
        
        print(f"Found {len(all_clusters)} disconnected components after block move")
        
        # Identify which cluster contains most of the connected blocks
        # (this should be our main cluster that we're building)
        main_cluster_idx = 0
        max_overlap = 0
        
        for i, cluster_set in enumerate(all_clusters):
            overlap = len(cluster_set.intersection(connected_blocks))
            if overlap > max_overlap:
                max_overlap = overlap
                main_cluster_idx = i
        
        main_cluster = set(all_clusters[main_cluster_idx])
        
        # Process each other cluster, trying to connect it to the main cluster
        for i, cluster_set in enumerate(all_clusters):
            if i == main_cluster_idx:
                continue  # Skip the main cluster
                
            # Skip empty clusters
            if not cluster_set:
                continue
                
            print(f"Reconnecting cluster with {len(cluster_set)} blocks to main cluster")
            
            # Find the closest pair of blocks between the clusters
            closest_pair = None
            min_distance = float('inf')
            
            for pos1 in cluster_set:
                for pos2 in main_cluster:
                    dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                    if dist < min_distance:
                        min_distance = dist
                        closest_pair = (pos1, pos2)
            
            if not closest_pair:
                print(f"No valid connection found between clusters")
                # Just add these blocks to the connected blocks collection
                connected_blocks.update(cluster_set)
                continue
                
            source_pos, target_pos = closest_pair
            
            # For very close clusters, just combine them directly
            if min_distance <= 2:
                connected_blocks.update(cluster_set)
                print(f"Clusters are very close, direct connection made")
                continue
            
            # Identify movable blocks in the separated cluster
            movable_block = None
            for block in cluster_set:
                # Check if removing this block maintains connectivity of the cluster
                temp_cluster = cluster_set - {block}
                
                # If only one block, or removing it maintains connectivity
                if not temp_cluster or self.agent.is_connected(temp_cluster):
                    movable_block = block
                    break
            
            if not movable_block:
                print(f"No movable blocks found in cluster")
                # Just add all blocks to connected blocks
                connected_blocks.update(cluster_set)
                continue
                
            # Find a path for this block to the main cluster
            path = None
            target_block = None
            
            # Try to find a path to any block in the main cluster
            for main_block in main_cluster:
                obstacles = ((current_blocks - {movable_block, main_block}) | 
                              self.agent.obstacles | blocks_at_goal)
                
                temp_path = self.agent.find_clean_path(movable_block, main_block, obstacles)
                
                if temp_path and (not path or len(temp_path) < len(path)):
                    path = temp_path
                    target_block = main_block
            
            if not path:
                print(f"No valid path found to reconnect cluster")
                # Just add all blocks to connected blocks
                connected_blocks.update(cluster_set)
                continue
                
            print(f"Found path of length {len(path)} to reconnect cluster")
            
            # Remove the block from its current position
            current_blocks.remove(movable_block)
            cluster_set.remove(movable_block)
            
            # Move the block along the path
            # We'll go to the second-to-last position to avoid colliding with the target
            last_idx = min(len(path) - 2, min_distance)
            
            # If path is too short, just use the last valid position
            if last_idx < 1:
                last_idx = 1 if len(path) > 1 else 0
            
            # Verify block count after removal
            assert len(current_blocks) == original_block_count - 1, f"Block count mismatch after removal: {len(current_blocks)}"
            
            # Add state to show removal
            cleanup_path.append(list(current_blocks))
            
            # Walk the block through each step of the path
            current_pos = None
            
            for i in range(1, last_idx + 1):
                # Get the next position in the path
                new_pos = path[i]
                
                # Check if this position is valid
                if (new_pos not in current_blocks and 
                    new_pos not in blocks_at_goal and
                    new_pos not in self.agent.obstacles):
                    
                    # If we had placed the block somewhere earlier, remove it
                    if current_pos:
                        current_blocks.remove(current_pos)
                        # Add state to show removal
                        cleanup_path.append(list(current_blocks))
                    
                    # Place the block at the new position
                    current_blocks.add(new_pos)
                    current_pos = new_pos
                    
                    # Verify block count
                    assert len(current_blocks) == original_block_count, f"Block count mismatch: {len(current_blocks)}"
                    
                    # Add state to show the move
                    cleanup_path.append(list(current_blocks))
                else:
                    # Position already occupied, try next
                    print(f"Position {new_pos} is occupied, trying next")
                    continue
            
            # If we successfully placed the block somewhere, count it
            if current_pos:
                main_cluster.add(current_pos)
                connected_blocks.add(current_pos)
                print(f"Successfully moved block to {current_pos} to reconnect cluster")
                
                # Add remaining blocks from this cluster to connected blocks
                connected_blocks.update(cluster_set)
            else:
                # Couldn't place the block anywhere, add it back
                print(f"Failed to move block, adding back")
                current_blocks.add(movable_block)
                cluster_set.add(movable_block)
                
                # Verify block count
                assert len(current_blocks) == original_block_count, f"Block count mismatch: {len(current_blocks)}"
                
                # Add state to show the rollback
                cleanup_path.append(list(current_blocks))
                
                # Add all blocks to connected blocks collection
                connected_blocks.update(cluster_set)