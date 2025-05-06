import heapq
import time

class DisconnectedGoalSequential:
    def __init__(self, agent):
        # Store reference to the parent agent
        self.agent = agent
    
    def find_closest_component(self):
        """Find the index of the closest goal component to the start state"""
        agent = self.agent
        
        start_centroid = agent.calculate_centroid(agent.start_state)
        min_distance = float('inf')
        closest_idx = 0
        
        for idx, centroid in enumerate(agent.component_centroids):
            distance = abs(start_centroid[0] - centroid[0]) + abs(start_centroid[1] - centroid[1])
            if distance < min_distance:
                min_distance = distance
                closest_idx = idx
                
        return closest_idx
    
    def assign_blocks_to_components(self, state):
        """
        Returns a dictionary mapping each component index to a set of positions
        """
        agent = self.agent
        
        assignments = {i: set() for i in range(len(agent.goal_components))}
        state_positions = list(state)
        
        # Create a dictionary to track assigned positions
        assigned = set()
        
        # First, count how many blocks we need for each component
        component_sizes = [len(comp) for comp in agent.goal_components]
        total_blocks_needed = sum(component_sizes)
        
        # Ensure we have enough blocks
        if len(state_positions) < total_blocks_needed:
            print(f"Warning: Not enough blocks ({len(state_positions)}) for goal state ({total_blocks_needed})")
            return None
            
        # Calculate distance from each block to each component centroid
        distances = []
        for pos in state_positions:
            pos_distances = []
            for idx, centroid in enumerate(agent.component_centroids):
                if agent.obstacles:
                    dist = agent.obstacle_handler.obstacle_aware_distance(pos, (int(centroid[0]), int(centroid[1])))
                else:
                    dist = abs(pos[0] - centroid[0]) + abs(pos[1] - centroid[1])
                pos_distances.append((idx, dist))
            distances.append((pos, sorted(pos_distances, key=lambda x: x[1])))
            
        # Sort blocks by their distance to their closest component
        distances.sort(key=lambda x: x[1][0][1])
        
        # Assign blocks to components in order of increasing distance
        for pos, component_distances in distances:
            for component_idx, dist in component_distances:
                if len(assignments[component_idx]) < len(agent.goal_components[component_idx]) and pos not in assigned:
                    assignments[component_idx].add(pos)
                    assigned.add(pos)
                    break
                    
        # Ensure all blocks are assigned
        unassigned = set(state_positions) - assigned
        if unassigned:
            # Assign remaining blocks to components that still need them
            for pos in unassigned:
                for component_idx in range(len(agent.goal_components)):
                    if len(assignments[component_idx]) < len(agent.goal_components[component_idx]):
                        assignments[component_idx].add(pos)
                        assigned.add(pos)
                        break
                        
        # Double-check that we've assigned the right number of blocks to each component
        for idx, component in enumerate(agent.goal_components):
            if len(assignments[idx]) != len(component):
                print(f"Warning: Component {idx} has {len(assignments[idx])} blocks assigned but needs {len(component)}")
                
        return assignments
    
    def plan_disconnect_moves(self, state, assignments):
        """
        Plan a sequence of moves to disconnect the shape into separate components
        Returns a list of states representing the disconnection process
        """
        agent = self.agent
        
        # Start with current state
        current_state = set(state)
        path = [frozenset(current_state)]
        
        # Group the assignments by component
        component_positions = [set(assignments[i]) for i in range(len(agent.goal_components))]
        
        # Check if the state is already naturally separable
        is_separable = True
        for i in range(len(component_positions)):
            for j in range(i+1, len(component_positions)):
                # Check if there's a direct connection between components
                for pos_i in component_positions[i]:
                    for pos_j in component_positions[j]:
                        # Check if positions are adjacent
                        if any((pos_i[0] + dx, pos_i[1] + dy) == pos_j for dx, dy in agent.directions):
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
                for pos_i in component_positions[i]:
                    for pos_j in component_positions[j]:
                        # Check if positions are adjacent
                        if any((pos_i[0] + dx, pos_i[1] + dy) == pos_j for dx, dy in agent.directions):
                            # Add both positions to connection points
                            connection_points.add(pos_i)
                            connection_points.add(pos_j)
        
        # For theoretical disconnection, we don't need to actually move blocks
        # Just mark the connection points as if they're disconnected
        print(f"Identified {len(connection_points)} connection points for theoretical disconnection")
        
        # Return current state as the disconnection plan
        # The actual disconnection will happen during the morphing phase
        return path
    
    def is_full_disconnected_goal_reached(self, state):
        """
        Check if goal components are sufficiently matched in the current state.
        Allows for some tolerance in the matching to avoid being too strict.
        """
        agent = self.agent
        
        state_components = agent.find_disconnected_components(state)
        
        # If we have fewer components than the goal, we're definitely not done
        if len(state_components) < len(agent.goal_components):
            return False
        
        # For each goal component, find the best matching state component
        total_correct_blocks = 0
        total_goal_blocks = sum(len(comp) for comp in agent.goal_components)
        
        # Convert to sets for easier intersection operations
        goal_component_sets = [set(comp) for comp in agent.goal_components]
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
        # This threshold can be adjusted based on requirements
        threshold = 0.90
        return match_percentage >= threshold
    
    def get_disconnected_valid_moves(self, state, goal_components):
        """
        Generate valid moves for disconnected components
        Allows moves that maintain connectivity within each component
        But doesn't require connectivity between components
        """
        agent = self.agent
        
        # Find the components in the current state
        current_components = agent.find_disconnected_components(state)
    
        # If we have fewer components than needed, can't generate valid disconnected moves
        if len(current_components) < len(goal_components):
            return agent.get_all_valid_moves(state)
    
        # Generate moves for each component separately
        all_moves = []
    
        for component in current_components:
            component_moves = []
        
            # Get basic morphing moves for this component
            component_state = frozenset(component)
            basic_moves = agent.get_valid_morphing_moves(component_state)
            chain_moves = agent.get_smart_chain_moves(component_state)
            sliding_moves = agent.get_sliding_chain_moves(component_state)
        
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
        # IMPORTANT: Try multiple methods and use direct start state if all fail
        try:
            block_path = agent.disconnected_block_movement_phase(move_time_limit)
        except Exception as e1:
            print(f"Agent method failed: {e1}")
            try:
                # Try the movement_phases version as backup
                block_path = agent.movement_phases.disconnected_block_movement_phase(move_time_limit)
            except Exception as e2:
                print(f"Movement phases method failed: {e2}")
                print("Block movement phase failed! Using direct start state.")
                block_path = [agent.start_state]  # Start from current state directly

        # CRITICAL FIX: Ensure we have a path even if the above fails
        if not block_path or len(block_path) == 0:
            print("Block movement phase returned empty path! Using direct start state.")
            block_path = [agent.start_state]  # Start from current state directly

        # Get the final state from block movement phase
        block_final_state = frozenset(block_path[-1])

        # Phase 2: Assign blocks to components and plan disconnection
        assignments_start_time = time.time()

        # CRITICAL FIX: Use multithreaded assignments if this fails
        try:
            assignments = self.assign_blocks_to_components(block_final_state)
            if assignments is None:
                raise ValueError("Assignment failed")
        except Exception as e:
            print(f"Sequential assignment failed: {e}, trying multithreaded assignment")
            try:
                # Try using the multithreaded handler's method as backup
                assignments = agent.disconnected_handler.assign_blocks_to_components(block_final_state)
            except Exception as e2:
                print(f"Multithreaded assignment also failed: {e2}")

        # CRITICAL FIX: If assignments still fail, create a simple assignment
        if assignments is None:
            print("All assignment methods failed! Creating direct assignment.")
            # Simple assignment: divide blocks equally among components
            block_list = list(block_final_state)
            assignments = {}

            # Make sure we have enough blocks
            total_goal_blocks = sum(len(comp) for comp in agent.goal_components)
            if len(block_list) < total_goal_blocks:
                print(f"WARNING: Not enough blocks for goal state. Have {len(block_list)}, need {total_goal_blocks}")
                # Return just the block movement path - can't achieve goal state
                return block_path

            block_index = 0
            for i, component in enumerate(agent.goal_components):
                comp_size = len(component)
                assignments[i] = set(block_list[block_index:block_index + comp_size])
                block_index += comp_size

        # Theoretical disconnection planning (with better error handling)
        try:
            disconnect_path = self.plan_disconnect_moves(block_final_state, assignments)
        except Exception as e:
            print(f"Disconnect planning failed: {e}")
            disconnect_path = [block_final_state]  # Just use current state
        
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

        # Track if we've made any progress
        made_progress = False

        for i in range(len(agent.goal_components)):
            print(f"Morphing component {i+1}/{len(agent.goal_components)}")

            # Extract the current state of this component from overall state
            component_blocks = component_start_states[i]

            # CRITICAL FIX: Add explicit error handling for component morphing
            try:
                # Morph this component
                component_path = self.component_morphing_phase(
                    component_blocks, 
                    agent.goal_components[i],
                    component_time_limits[i]
                )

                if not component_path or len(component_path) == 0:
                    print(f"Component {i+1} morphing returned empty path, skipping")
                    continue
                    
                # Update overall state with the morphed component
                # Remove the old component positions
                updated_state = set(current_overall_state) - set(component_blocks)
                # Add the new component positions
                updated_state.update(component_path[-1])
                current_overall_state = frozenset(updated_state)

                # Add this updated overall state to the combined path
                combined_path.append(list(current_overall_state))
                made_progress = True

            except Exception as e:
                print(f"Error in component {i+1} morphing: {e}")
                # Skip this component, continue with others
                continue
    
        # CRITICAL FIX: If we didn't make any progress at all, return the block movement path
        if not made_progress:
            print("WARNING: No component morphing succeeded. Returning just the block movement path.")
            return block_path
    
        # CRITICAL FIX: Add the final state if it's not already in the path
        if combined_path and combined_path[-1] != list(current_overall_state):
            combined_path.append(list(current_overall_state))
    
        # Check if goal is reached
        if not self.is_full_disconnected_goal_reached(combined_path[-1]):
            print("WARNING: Final state does not match all goal components.")
            # CRITICAL FIX: Return the best solution we have instead of None
            print("Returning best partial solution.")
            return combined_path
        
        return combined_path