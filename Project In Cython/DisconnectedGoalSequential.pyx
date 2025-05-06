import heapq
import time

class DisconnectedGoalSequential:
    def __init__(self, agent):
        # Store reference to the parent agent
        self.agent = agent
    
    def component_morphing_heuristic(self, state, goal_component):
        """
        Heuristic for morphing a specific component
        """
        agent = self.agent
        
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
                if agent.obstacles:
                    dist = agent.obstacle_handler.obstacle_aware_distance(pos, goal_pos)
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
    
    def component_morphing_phase(self, start_state, goal_component, time_limit=15):
        """
        Morph a specific component into its goal shape
        """
        agent = self.agent
        start_time = time.time()
        
        # Adapt beam width based on obstacle density
        adaptive_beam_width = agent.beam_width
        if len(agent.obstacles) > 0:
            obstacle_density = len(agent.obstacles) / (agent.grid_size[0] * agent.grid_size[1])
            adaptive_beam_width = int(agent.beam_width * (1 + min(1.0, obstacle_density * 5)))
            
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
                return agent.reconstruct_path(came_from, current)
            
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
            if iterations >= agent.max_iterations:
                print(f"Reached max iterations ({agent.max_iterations}) for component morphing")
                break
            
            closed_set.add(current)
        
            # Get valid moves for this component
            neighbors = agent.get_all_valid_moves(current)
        
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
        return agent.reconstruct_path(came_from, best_state)
        
    def search_disconnected_goal_sequential(self, time_limit=30):
        """
        Sequential search method for disconnected goal states:
        1. Move blocks to strategic position
        2. Assign blocks to components
        3. Morph each component one at a time
        """
        agent = self.agent
        print(f"Starting sequential search for disconnected goal with {len(agent.goal_components)} components")
        start_time = time.time()

        # Allocate time for different phases
        move_time_ratio = 0.2  # 20% for block movement
        disconnect_time_ratio = 0.1  # 10% for disconnection planning
        morphing_time_ratio = 0.7  # 70% for morphing

        # Adjust ratios if obstacles are present
        if len(agent.obstacles) > 0:
            obstacle_density = len(agent.obstacles) / (agent.grid_size[0] * agent.grid_size[1])
            move_time_ratio = min(0.4, 0.2 + obstacle_density * 0.5)  # Up to 40% for movement
            morphing_time_ratio = 1.0 - move_time_ratio - disconnect_time_ratio

        move_time_limit = time_limit * move_time_ratio
        disconnect_time_limit = time_limit * disconnect_time_ratio
        morphing_time_limit = time_limit * morphing_time_ratio

        print(f"Time allocation: {move_time_ratio:.1%} block movement, "
              f"{disconnect_time_ratio:.1%} disconnection planning, "
              f"{morphing_time_ratio:.1%} morphing")

        # Phase 1: Strategic Block Movement
        block_path = agent.movement_phases.disconnected_block_movement_phase(move_time_limit)

        if not block_path:
            print("Block movement phase failed!")
            return None

        # Get the final state from block movement phase
        block_final_state = frozenset(block_path[-1])

        # Phase 2: Assign blocks to components and plan disconnection
        assignments_start_time = time.time()
        assignments = agent.disconnected_handler.assign_blocks_to_components(block_final_state)

        if assignments is None:
            print("Failed to assign blocks to components!")
            return block_path

        # Theoretical disconnection planning
        disconnect_path = agent.disconnected_handler.plan_disconnect_moves(block_final_state, assignments)
        disconnect_time_used = time.time() - assignments_start_time

        # Add remaining disconnect time to morphing time
        if disconnect_time_used < disconnect_time_limit:
            morphing_time_limit += (disconnect_time_limit - disconnect_time_used)

        # Phase 3: Sequential Morphing of Components
        print(f"Starting sequential morphing of {len(agent.goal_components)} components")

        # Prepare component states and goals
        component_start_states = []
        for i in range(len(agent.goal_components)):
            component_start_states.append(frozenset(assignments[i]))

        # Time allocation for each component based on its size
        total_blocks = sum(len(comp) for comp in agent.goal_components)
        component_time_limits = [
            morphing_time_limit * len(comp) / total_blocks 
            for comp in agent.goal_components
        ]

        # Morph components sequentially
        combined_path = block_path[:-1]  # Remove the last state from block path
        current_overall_state = frozenset(block_path[-1])
        
        for i in range(len(agent.goal_components)):
            print(f"Morphing component {i+1}/{len(agent.goal_components)}")
            
            # Extract the current state of this component from overall state
            component_blocks = component_start_states[i]
            
            # Morph this component
            component_path = self.component_morphing_phase(
                component_blocks, 
                agent.goal_components[i],
                component_time_limits[i]
            )
            
            if not component_path:
                print(f"Component {i+1} morphing failed!")
                continue
            
            # Update overall state with the morphed component
            # Remove the old component positions
            updated_state = set(current_overall_state) - set(component_blocks)
            # Add the new component positions
            updated_state.update(component_path[-1])
            current_overall_state = frozenset(updated_state)
            
            # Add this updated overall state to the combined path
            combined_path.append(list(current_overall_state))
        
        # Check if goal is reached
        if not agent.disconnected_handler.is_full_disconnected_goal_reached(combined_path[-1]):
            print("WARNING: Final state does not match all goal components. Returning incomplete solution.")
            return None
            
        return combined_path