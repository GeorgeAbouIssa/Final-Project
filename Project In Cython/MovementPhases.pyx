import heapq
import time

class MovementPhases:
    def __init__(self, agent):
        # Store reference to the parent agent
        self.agent = agent
        
    def block_heuristic(self, state):
        """
        Heuristic for block movement phase:
        Accounts for obstacles between current position and goal
        """
        agent = self.agent
        
        if not state:
            return float('inf')
        
        # Check if we've lost any blocks and apply proportional penalty
        if len(state) < len(agent.start_state):
            missing_blocks = len(agent.start_state) - len(state)
            missing_penalty = 10000 * missing_blocks  # Large penalty per missing block
            return missing_penalty  # Return large penalty proportional to missing blocks
                
        current_centroid = agent.calculate_centroid(state)
        goal_centroid_int = (int(agent.goal_centroid[0]), int(agent.goal_centroid[1]))
        
        # If no obstacles, use simple Manhattan distance
        if not agent.obstacles:
            return abs(current_centroid[0] - agent.goal_centroid[0]) + abs(current_centroid[1] - agent.goal_centroid[1])
        
        # With obstacles, calculate path distance to goal centroid
        # Round centroid to nearest grid cell for distance calculation
        current_centroid_int = (int(round(current_centroid[0])), int(round(current_centroid[1])))
        
        # Ensure centroid is within bounds
        current_centroid_int = (
            max(0, min(current_centroid_int[0], agent.grid_size[0]-1)),
            max(0, min(current_centroid_int[1], agent.grid_size[1]-1))
        )
        goal_centroid_int = (
            max(0, min(goal_centroid_int[0], agent.grid_size[0]-1)),
            max(0, min(goal_centroid_int[1], agent.grid_size[1]-1))
        )
        
        # Get obstacle-aware distance
        return agent.obstacle_handler.obstacle_aware_distance(current_centroid_int, goal_centroid_int)
    
    def improved_morphing_heuristic(self, state):
        """
        Improved heuristic for morphing phase:
        Uses bipartite matching to find optimal assignment of blocks to goal positions,
        now accounts for obstacles
        """
        agent = self.agent
        
        if not state:
            return float('inf')
        
        # Check if we've lost any blocks and apply proportional penalty
        if len(state) < len(agent.start_state):
            missing_blocks = len(agent.start_state) - len(state)
            missing_penalty = 10000 * missing_blocks  # Large penalty per missing block
            return missing_penalty  # Return large penalty proportional to missing blocks
                
        state_list = list(state)
        goal_list = list(agent.goal_state)
        
        # Early exit if states have different sizes
        if len(state_list) != len(goal_list):
            return float('inf')
        
        # Reward blocks that are already in goal positions
        blocks_at_goal = set(state).intersection(agent.goal_state)
        goal_bonus = -len(blocks_at_goal) * 10  # Large negative value (bonus) for blocks in place
        
        # Build distance matrix with obstacle-aware distances
        distances = []
        for pos in state_list:
            # If block is already at a goal position, give it maximum preference
            # to stay where it is by assigning 0 distance to its current position
            # and high distance to all other positions
            if pos in agent.goal_state:
                row = [0 if goal_pos == pos else 1000 for goal_pos in goal_list]
            else:
                row = []
                for goal_pos in goal_list:
                    # Use obstacle-aware distance calculation
                    if agent.obstacles:
                        dist = agent.obstacle_handler.obstacle_aware_distance(pos, goal_pos)
                    else:
                        # Use faster Manhattan distance if no obstacles
                        dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                        
                    # If this goal position is already filled, make it less attractive
                    if goal_pos in blocks_at_goal and goal_pos != pos:
                        dist += 100  # Discourage moving to goals that are already filled
                        
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
        
        # Return total distance plus goal bonus
        return total_distance + goal_bonus
    
    def block_movement_phase(self, time_limit=15):
        """
        Phase 1: Move the entire block toward the goal centroid
        Returns the path of states to get near the goal area
        Modified to stop 1 grid cell before reaching the goal centroid
        """
        agent = self.agent
        
        print("Starting Block Movement Phase...")
        start_time = time.time()

        # For disconnected goals, use the modified approach
        if agent.is_goal_disconnected:
            return self.disconnected_block_movement_phase(time_limit)

        # Initialize A* search
        open_set = [(self.block_heuristic(agent.start_state), 0, agent.start_state)]
        closed_set = set()

        # Track path and g-scores
        g_score = {agent.start_state: 0}
        came_from = {agent.start_state: None}

        # Modified: We want to stop 1 grid cell before reaching the centroid
        # Instead of using a small threshold, we'll check if distance is between 1.0 and 2.0
        # This ensures we're approximately 1 grid cell away from the goal centroid
        min_distance = 1.0
        max_distance = 1.0

        while open_set and time.time() - start_time < time_limit:
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
    
            # Skip if already processed
            if current in closed_set:
                continue
        
            # Check if we're at the desired distance from the goal centroid
            current_centroid = agent.calculate_centroid(current)
            centroid_distance = (abs(current_centroid[0] - agent.goal_centroid[0]) + 
                            abs(current_centroid[1] - agent.goal_centroid[1]))
                        
            if min_distance <= centroid_distance <= max_distance:
                print(f"Block stopped 1 grid cell before goal centroid. Distance: {centroid_distance}")
                return agent.reconstruct_path(came_from, current)
        
            closed_set.add(current)
    
            # Process neighbor states (block moves)
            for neighbor in agent.get_valid_block_moves(current):
                if neighbor in closed_set:
                    continue
            
                # Calculate tentative g-score
                tentative_g = g_score[current] + 1
        
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                
                    # Modified: Adjust heuristic to prefer states that are close to but not at the centroid
                    neighbor_centroid = agent.calculate_centroid(neighbor)
                    neighbor_distance = (abs(neighbor_centroid[0] - agent.goal_centroid[0]) + 
                                    abs(neighbor_centroid[1] - agent.goal_centroid[1]))
                
                    # Penalize distances that are too small (< 1.0)
                    distance_penalty = 0
                    if neighbor_distance < min_distance:
                        distance_penalty = 10 * (min_distance - neighbor_distance)
                
                    adjusted_heuristic = self.block_heuristic(neighbor) + distance_penalty
                    f_score = tentative_g + adjusted_heuristic
            
                    # Add to open set
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # If we exit the loop, either no path was found or time limit reached
        if time.time() - start_time >= time_limit:
            print("Block movement phase timed out!")
    
        # Return the best state we found
        if came_from:
            # Find state with appropriate distance to centroid
            best_state = None
            best_distance_diff = float('inf')
        
            for state in came_from.keys():
                state_centroid = agent.calculate_centroid(state)
                distance = (abs(state_centroid[0] - agent.goal_centroid[0]) + 
                            abs(state_centroid[1] - agent.goal_centroid[1]))
            
                # We want a state that's as close as possible to our target distance range
                if distance < min_distance:
                    distance_diff = min_distance - distance
                elif distance > max_distance:
                    distance_diff = distance - max_distance
                else:
                    # Distance is within our desired range
                    best_state = state
                    break
                
                if distance_diff < best_distance_diff:
                    best_distance_diff = distance_diff
                    best_state = state
        
            if best_state:
                best_centroid = agent.calculate_centroid(best_state)
                best_distance = (abs(best_centroid[0] - agent.goal_centroid[0]) + 
                                abs(best_centroid[1] - agent.goal_centroid[1]))
                print(f"Best block position found with centroid distance: {best_distance}")
                return agent.reconstruct_path(came_from, best_state)
    
        return [agent.start_state]  # No movement possible
    
    def smarter_morphing_phase(self, start_state, time_limit=15):
        """
        Improved Phase 2: Morph the block into the goal shape while maintaining connectivity
        Uses beam search and intelligent move generation with support for simultaneous moves
        Now with adaptive beam width based on obstacle density
        """
        agent = self.agent
        
        print(f"Starting Smarter Morphing Phase with {agent.min_simultaneous_moves}-{agent.max_simultaneous_moves} simultaneous moves...")
        start_time = time.time()
        
        # Identify blocks already at goal positions in the start state
        agent.blocks_at_goal = set(start_state).intersection(agent.goal_state)
        print(f"Starting morphing with {len(agent.blocks_at_goal)} blocks already at goal positions")
        
        # Adapt beam width based on obstacle density
        adaptive_beam_width = agent.beam_width
        if len(agent.obstacles) > 0:
            # Increase beam width for environments with obstacles
            obstacle_density = len(agent.obstacles) / (agent.grid_size[0] * agent.grid_size[1])
            adaptive_beam_width = int(agent.beam_width * (1 + min(1.0, obstacle_density * 5)))
            print(f"Adjusted beam width to {adaptive_beam_width} based on obstacle density")
            
        # Initialize beam search
        open_set = [(self.improved_morphing_heuristic(start_state), 0, start_state)]
        closed_set = set()
        
        # Track path, g-scores, and best state
        g_score = {start_state: 0}
        came_from = {start_state: None}
        
        # Track best state seen so far
        best_state = start_state
        best_heuristic = self.improved_morphing_heuristic(start_state)
        
        # Track the maximum number of blocks at goal positions seen so far
        max_blocks_at_goal = len(agent.blocks_at_goal)
        
        iterations = 0
        last_improvement_time = time.time()
        
        while open_set and time.time() - start_time < time_limit:
            iterations += 1
            
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            # Check if goal reached
            if current == agent.goal_state:
                print(f"Goal reached after {iterations} iterations!")
                return agent.reconstruct_path(came_from, current)
            
            # Check how many blocks are at goal positions in current state
            blocks_at_goal_current = len(set(current).intersection(agent.goal_state))
            
            # Update max_blocks_at_goal if we found a better state
            if blocks_at_goal_current > max_blocks_at_goal:
                max_blocks_at_goal = blocks_at_goal_current
                print(f"New maximum blocks at goal: {max_blocks_at_goal}/{len(agent.goal_state)}")
            
            # Check if this is the best state seen so far
            current_heuristic = self.improved_morphing_heuristic(current)
            if current_heuristic < best_heuristic or blocks_at_goal_current > len(agent.blocks_at_goal):
                best_state = current
                best_heuristic = current_heuristic
                agent.blocks_at_goal = set(current).intersection(agent.goal_state)
                last_improvement_time = time.time()
                
                # Print progress occasionally
                if iterations % 500 == 0:
                    print(f"Progress: h={best_heuristic}, blocks at goal={len(agent.blocks_at_goal)}/{len(agent.goal_state)}, iterations={iterations}")
                    
                # If we're very close to the goal, increase search intensity
                if best_heuristic < 5 * len(agent.goal_state):
                    adaptive_beam_width *= 2
            
            # Check for stagnation - more patient in obstacle-heavy environments
            stagnation_tolerance = time_limit * (0.3 + min(0.3, len(agent.obstacles) / 100))
            if time.time() - last_improvement_time > stagnation_tolerance:
                print("Search stagnated, restarting...")
                # Clear the beam and start from the best state
                open_set = [(best_heuristic, g_score[best_state], best_state)]
                last_improvement_time = time.time()
            
            # Limit iterations to prevent infinite loops
            if iterations >= agent.max_iterations:
                print(f"Reached max iterations ({agent.max_iterations})")
                break
                
            closed_set.add(current)
            
            # Get all valid moves
            neighbors = agent.get_all_valid_moves(current)
            
            # Process each neighbor
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g-score
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Check if the move causes blocks to disappear
                    if len(neighbor) != len(current):
                        continue  # Skip this move
                        
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # Calculate blocks at goal in this neighbor
                    blocks_at_goal_neighbor = len(set(neighbor).intersection(agent.goal_state))
                    
                    # Prioritize states with more blocks at goal positions by giving them better f-scores
                    goal_position_bonus = max(0, blocks_at_goal_neighbor - blocks_at_goal_current) * 10
                    
                    f_score = tentative_g + self.improved_morphing_heuristic(neighbor) - goal_position_bonus
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
            
            # Beam search pruning: keep only the best states
            if len(open_set) > adaptive_beam_width:
                open_set = heapq.nsmallest(adaptive_beam_width, open_set)
                heapq.heapify(open_set)
        
        # If we exit the loop, either no path was found or time limit reached
        if time.time() - start_time >= time_limit:
            print(f"Morphing phase timed out after {iterations} iterations!")
        
        # Return the best state found
        return agent.reconstruct_path(came_from, best_state)
    
    def disconnected_block_movement_phase(self, time_limit=15):
        """
        Modified Phase 1 for disconnected goal states:
        Moves the entire block toward a strategic position for splitting
        """
        agent = self.agent
        
        print("Starting Disconnected Block Movement Phase...")
        start_time = time.time()
        
        # Find the closest goal component to the start state
        closest_component_idx = agent.disconnected_handler.find_closest_component()
        closest_component = agent.goal_components[closest_component_idx]
        closest_centroid = agent.component_centroids[closest_component_idx]
        
        # Determine if vertical positioning is better based on y-axis centroids
        all_components_y = [centroid[1] for centroid in agent.component_centroids]
        overall_y = agent.goal_centroid[1]
        
        # Check if centroid of all shapes is closer to y level of the closest shape
        use_vertical_approach = abs(overall_y - closest_centroid[1]) < sum([abs(y - closest_centroid[1]) for y in all_components_y]) / len(all_components_y)
        
        if use_vertical_approach:
            print("Using vertical approach for block movement")
            # Target position is at the overall centroid with y-level of closest component
            target_centroid = (agent.goal_centroid[0], closest_centroid[1])
        else:
            print("Using standard approach for block movement")
            # Target position is the overall centroid
            target_centroid = agent.goal_centroid
            
        # Cache original goal centroid and temporarily replace with target
        original_centroid = agent.goal_centroid
        agent.goal_centroid = target_centroid
        
        # Use standard A* search but with the modified target
        open_set = [(self.block_heuristic(agent.start_state), 0, agent.start_state)]
        closed_set = set()
        g_score = {agent.start_state: 0}
        came_from = {agent.start_state: None}
        
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
            current_centroid = agent.calculate_centroid(current)
            centroid_distance = (abs(current_centroid[0] - target_centroid[0]) + 
                            abs(current_centroid[1] - target_centroid[1]))
                        
            if min_distance <= centroid_distance <= max_distance:
                print(f"Block stopped at strategic position. Distance: {centroid_distance}")
                # Restore original goal centroid
                agent.goal_centroid = original_centroid
                return agent.reconstruct_path(came_from, current)
        
            closed_set.add(current)
    
            # Process neighbor states (block moves)
            for neighbor in agent.get_valid_block_moves(current):
                if neighbor in closed_set:
                    continue
            
                # Calculate tentative g-score
                tentative_g = g_score[current] + 1
        
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                
                    # Adjust heuristic to prefer states close to target
                    neighbor_centroid = agent.calculate_centroid(neighbor)
                    neighbor_distance = (abs(neighbor_centroid[0] - target_centroid[0]) + 
                                    abs(neighbor_centroid[1] - target_centroid[1]))
                
                    # Penalize distances that are too small
                    distance_penalty = 0
                    if neighbor_distance < min_distance:
                        distance_penalty = 10 * (min_distance - neighbor_distance)
                
                    # Calculate Manhattan distance to target
                    if agent.obstacles:
                        neighbor_centroid_int = (int(round(neighbor_centroid[0])), int(round(neighbor_centroid[1])))
                        target_centroid_int = (int(round(target_centroid[0])), int(round(target_centroid[1])))
                        
                        # Ensure centroids are within bounds
                        neighbor_centroid_int = (
                            max(0, min(neighbor_centroid_int[0], agent.grid_size[0]-1)),
                            max(0, min(neighbor_centroid_int[1], agent.grid_size[1]-1))
                        )
                        target_centroid_int = (
                            max(0, min(target_centroid_int[0], agent.grid_size[0]-1)),
                            max(0, min(target_centroid_int[1], agent.grid_size[1]-1))
                        )
                        
                        adjusted_heuristic = agent.obstacle_handler.obstacle_aware_distance(neighbor_centroid_int, target_centroid_int) + distance_penalty
                    else:
                        adjusted_heuristic = neighbor_distance + distance_penalty
                        
                    f_score = tentative_g + adjusted_heuristic
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # Restore original goal centroid
        agent.goal_centroid = original_centroid
        
        # If we exit the loop, find the best available state
        if came_from:
            best_state = None
            best_distance_diff = float('inf')
        
            for state in came_from.keys():
                state_centroid = agent.calculate_centroid(state)
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
                return agent.reconstruct_path(came_from, best_state)
    
        return [agent.start_state]  # No movement possible