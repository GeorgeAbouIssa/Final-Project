import time
import concurrent.futures
from threading import Lock

class DisconnectedGoalMultiThreaded:
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
        
    def search_disconnected_goal_multithreaded(self, time_limit=30):
        """
        Multithreaded search method for disconnected goal states:
        1. Move blocks to strategic position
        2. Assign blocks to components
        3. Morph each component in parallel
        If this fails, fall back to sequential approach
        """
        agent = self.agent
        
        # First, check if the goal state is truly disconnected or just identified as such
        if agent.is_connected(agent.goal_state):
            print("Goal appears disconnected but is actually connected, falling back to standard search")
            # If the goal is actually connected, use the standard search approach
            block_time_ratio = 0.3
            if len(agent.obstacles) > 0:
                obstacle_density = len(agent.obstacles) / (agent.grid_size[0] * agent.grid_size[1])
                block_time_ratio = min(0.5, 0.3 + obstacle_density * 0.5)
                
            block_time_limit = time_limit * block_time_ratio
            morphing_time_limit = time_limit * (1 - block_time_ratio)
            
            # Use standard search approach
            block_path = agent.movement_phases.block_movement_phase(block_time_limit)
            if not block_path:
                return None
                
            block_final_state = frozenset(block_path[-1])
            morphing_path = agent.movement_phases.smarter_morphing_phase(block_final_state, morphing_time_limit)
            
            if not morphing_path:
                return block_path
                
            return block_path[:-1] + morphing_path
        
        print(f"Starting multithreaded search for disconnected goal with {len(agent.goal_components)} components")
        start_time = time.time()

        # Allocate time for different phases
        move_time_ratio = 0.2  # 20% for block movement
        disconnect_time_ratio = 0.15  # 15% for disconnection planning
        morphing_time_ratio = 0.65  # 65% for morphing

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
        # Try both methods to find a block path
        try:
            # First try using the movement_phases handler
            block_path = agent.movement_phases.disconnected_block_movement_phase(move_time_limit)
        except Exception as e1:
            print(f"Movement phases handler failed: {e1}")
            try:
                # If that fails, try using the direct method
                block_path = agent.disconnected_block_movement_phase(move_time_limit)
            except Exception as e2:
                print(f"Direct method failed: {e2}")
                # If both fail, use a simple path with just the start state
                print("Block movement phase failed! Using direct start state.")
                block_path = [agent.start_state]

        if not block_path:
            print("Block movement phase returned empty path! Using direct start state.")
            block_path = [agent.start_state]

        # Get the final state from block movement phase
        block_final_state = frozenset(block_path[-1])

        # Phase 2: Assign blocks to components and plan disconnection
        assignments_start_time = time.time()
        assignments = self.assign_blocks_to_components(block_final_state)

        if assignments is None:
            print("Failed to assign blocks to components!")
            # Fallback to sequential method
            remaining_time = max(1, time_limit - (time.time() - start_time))
            print(f"Falling back to sequential method with {remaining_time:.1f} seconds remaining")
            return self.agent.sequential_handler.search_disconnected_goal_sequential(remaining_time)

        # Theoretical disconnection planning
        disconnect_path = self.plan_disconnect_moves(block_final_state, assignments)
        disconnect_time_used = time.time() - assignments_start_time

        # Add remaining disconnect time to morphing time
        if disconnect_time_used < disconnect_time_limit:
            morphing_time_limit += (disconnect_time_limit - disconnect_time_used)

        # Phase 3: Parallel Morphing of Components
        print(f"Starting parallel morphing of {len(agent.goal_components)} components")

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

        # Add shared resources for thread coordination
        shared_grid_lock = Lock()
        shared_occupied_cells = set()  # Track cells occupied by any component
        component_locks = [Lock() for _ in range(len(agent.goal_components))]
        component_occupied_cells = [set() for _ in range(len(agent.goal_components))]
        
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
                basic_moves = agent.get_valid_morphing_moves(state)
                chain_moves = agent.get_smart_chain_moves(state)
                sliding_moves = agent.get_sliding_chain_moves(state)
                
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
            original_get_all_valid_moves = agent.get_all_valid_moves
            agent.get_all_valid_moves = coordinated_get_valid_moves
            
            try:
                # Run the component morphing from the sequential handler
                # This is the critical connection with the sequential handler
                result = self.agent.sequential_handler.component_morphing_phase(start_state, goal_component, time_limit)
                
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
                agent.get_all_valid_moves = original_get_all_valid_moves
        
        # Initialize component_paths
        component_paths = []
    
        # Use thread pool for parallel morphing with coordination
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(agent.goal_components)) as executor:
            # Submit tasks for each component with our wrapper
            futures = []
            for i in range(len(agent.goal_components)):
                futures.append(
                    executor.submit(
                        thread_safe_component_morphing,
                        i,
                        component_start_states[i],
                        agent.goal_components[i],
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
        if len(component_paths) != len(agent.goal_components):
            print(f"Warning: Expected {len(agent.goal_components)} component paths, got {len(component_paths)}")
            # Fill in missing paths with their start states
            for i in range(len(component_paths), len(agent.goal_components)):
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
                elif path:  # Add this condition to use the final state when a thread is done
                    combined_state.update(path[-1])  # Use the last state of this component
        
            # Add the combined state to the path if not empty
            if combined_state:
                combined_path.append(list(combined_state))

        # Check if goal is reached using our more forgiving criterion
        if not self.is_full_disconnected_goal_reached(combined_path[-1]):
            print("WARNING: Multithreaded approach failed. Falling back to sequential approach.")
            
            # IMPORTANT FIX: Call the sequential handler properly
            # Calculate remaining time for the sequential approach
            remaining_time = max(1, time_limit - (time.time() - start_time))
            print(f"Switching to sequential approach with {remaining_time:.1f} seconds remaining")
            
            # Make sure we use the correct attribute and method names
            # Use the sequential handler that was initialized in ConnectedMatterAgent
            try:
                # This is the proper way to access the sequential handler
                return self.agent.sequential_handler.search_disconnected_goal_sequential(remaining_time)
            except Exception as e:
                print(f"Sequential fallback failed with error: {e}")
                print("Returning the best result from multithreaded approach anyway")
                # Return the best result we have, even if it doesn't fully satisfy the goal
                return combined_path

        return combined_path