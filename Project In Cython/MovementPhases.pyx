import heapq
import time
from collections import deque

class MovementPhases:
    def __init__(self, agent):
        self.agent = agent
        
    def get_valid_block_moves(self, state):
        """
        Generate valid moves for the entire block of elements
        A valid block move shifts all elements in the same direction while maintaining connectivity
        """
        valid_moves = []
        state_list = list(state)
        
        # Try moving the entire block in each direction
        for dx, dy in self.agent.directions:
            # Calculate new positions after moving
            new_positions = [(pos[0] + dx, pos[1] + dy) for pos in state_list]
            
            # Check if all new positions are valid (within bounds, not occupied by obstacles)
            all_valid = all(
                0 <= pos[0] < self.agent.grid_size[0] and 
                0 <= pos[1] < self.agent.grid_size[1] and
                pos not in self.agent.obstacles 
                for pos in new_positions
            )
            
            # Only consider moves that keep all positions within bounds and not overlapping obstacles
            if all_valid:
                # Ensure no positions overlap - each position must be unique
                if len(set(new_positions)) == len(new_positions):
                    new_state = frozenset(new_positions)
                    valid_moves.append(new_state)
        
        return valid_moves
    
    def get_valid_morphing_moves(self, state):
        """
        Generate valid morphing moves that maintain connectivity
        Supports multiple simultaneous block movements with minimum requirement
        """
        state_key = hash(state)
        if state_key in self.agent.valid_moves_cache:
            return self.agent.valid_moves_cache[state_key]
            
        # Get single block moves first
        single_moves = []
        state_set = set(state)
        
        # Identify blocks that have reached their goal positions
        blocks_at_goal = state_set.intersection(self.agent.goal_state)
        self.agent.blocks_at_goal = blocks_at_goal
        
        # Find non-critical points that can move without breaking connectivity
        articulation_points = self.agent.get_articulation_points(state_set)
        
        # Don't move blocks that have reached their goal positions
        # unless they're the only blocks we can move (to avoid deadlock)
        movable_points = state_set - articulation_points - blocks_at_goal
        
        # If all points are critical or at goal, try moving critical points that aren't at goals
        if not movable_points:
            for point in articulation_points - blocks_at_goal:
                # Try removing and see if structure remains connected
                temp_state = state_set.copy()
                temp_state.remove(point)
                if self.agent.is_connected(temp_state):
                    movable_points.add(point)
                    
        # If still no movable points and we have blocks at goal,
        # allow minimal movement of goal blocks as last resort
        if not movable_points and blocks_at_goal:
            # Try moving goal blocks that aren't critical articulation points first
            non_critical_goal_blocks = blocks_at_goal - articulation_points
            if non_critical_goal_blocks:
                for point in non_critical_goal_blocks:
                    temp_state = state_set.copy()
                    temp_state.remove(point)
                    if self.agent.is_connected(temp_state):
                        movable_points.add(point)
            
            # If still stuck, try critical goal blocks as absolute last resort
            if not movable_points:
                for point in blocks_at_goal.intersection(articulation_points):
                    temp_state = state_set.copy()
                    temp_state.remove(point)
                    if self.agent.is_connected(temp_state):
                        movable_points.add(point)
        
        # Generate single block moves, prioritizing moves toward the goal
        for point in movable_points:
            # Find closest goal position for this point
            closest_goal = None
            min_dist = float('inf')
            
            for goal_pos in self.agent.goal_state:
                if goal_pos not in state_set:  # Only consider unoccupied goals
                    if self.agent.obstacles:
                        dist = self.agent.obstacle_aware_distance(point, goal_pos)
                    else:
                        dist = abs(point[0] - goal_pos[0]) + abs(point[1] - goal_pos[1])
                        
                    if dist < min_dist:
                        min_dist = dist
                        closest_goal = goal_pos
                        
            # Prioritize directions toward the closest goal
            ordered_directions = self.agent.directions.copy()
            if closest_goal:
                # Calculate direction vectors
                dx = 1 if closest_goal[0] > point[0] else -1 if closest_goal[0] < point[0] else 0
                dy = 1 if closest_goal[1] > point[1] else -1 if closest_goal[1] < point[1] else 0
                
                # Put preferred direction first
                preferred_dir = (dx, dy)
                if preferred_dir in ordered_directions:
                    ordered_directions.remove(preferred_dir)
                    ordered_directions.insert(0, preferred_dir)
                
                # Also prioritize partial matches (just x or just y component)
                for i, dir in enumerate(ordered_directions.copy()):
                    if dir[0] == dx or dir[1] == dy:
                        ordered_directions.remove(dir)
                        ordered_directions.insert(1, dir)
                
            # Try moving in each direction, starting with preferred ones
            for dx, dy in ordered_directions:
                new_pos = (point[0] + dx, point[1] + dy)
                
                # Skip if out of bounds or is an obstacle - strict validation
                if not self.agent.is_valid_position(new_pos):
                    continue
                
                # Skip if already occupied
                if new_pos in state_set:
                    continue
                
                # Create new state by moving the point
                new_state_set = state_set.copy()
                new_state_set.remove(point)
                new_state_set.add(new_pos)
                
                # Check if new position is adjacent to at least one other point
                has_adjacent = False
                for adj_dx, adj_dy in self.agent.directions:
                    adj_pos = (new_pos[0] + adj_dx, new_pos[1] + adj_dy)
                    if adj_pos in new_state_set and adj_pos != new_pos:
                        has_adjacent = True
                        break
                
                # Only consider moves that maintain connectivity
                if has_adjacent and self.agent.is_connected(new_state_set):
                    single_moves.append((point, new_pos))
                    
        # Start with empty valid moves list
        valid_moves = []
        
        # In dense obstacle environments, more simultaneous moves could be better
        # Increase minimum number of simultaneous moves based on obstacle density
        local_min_moves = self.agent.min_simultaneous_moves
        if len(self.agent.obstacles) > 20 and local_min_moves == 1:
            local_min_moves = min(2, self.agent.max_simultaneous_moves)
            
        # Generate multi-block moves
        for k in range(local_min_moves, min(self.agent.max_simultaneous_moves + 1, len(single_moves) + 1)):
            # Generate combinations of k moves
            for combo in self._generate_move_combinations(single_moves, k):
                # Check if the combination is valid (no conflicts)
                if self._is_valid_move_combination(combo, state_set):
                    # Apply the combination and check connectivity
                    new_state = self._apply_moves(state_set, combo)
                    if new_state != state_set and self.agent.is_connected(new_state):
                        valid_moves.append(frozenset(new_state))
        
        # If no valid moves with min_simultaneous_moves, fallback to single moves if allowed
        if not valid_moves and self.agent.min_simultaneous_moves == 1:
            for move in single_moves:
                new_state = self._apply_moves(state_set, [move])
                if new_state != state_set:  # Only add if the move was successful
                    valid_moves.append(frozenset(new_state))
        
        # Cache results
        self.agent.valid_moves_cache[state_key] = valid_moves
        return valid_moves
    
    def _generate_move_combinations(self, single_moves, k):
        """Generate all combinations of k moves from the list of single moves"""
        if k == 1:
            return [[move] for move in single_moves]
        
        result = []
        for i in range(len(single_moves) - k + 1):
            move = single_moves[i]
            for combo in self._generate_move_combinations(single_moves[i+1:], k-1):
                result.append([move] + combo)
        
        return result
    
    def _is_valid_move_combination(self, moves, state_set):
        """Check if a combination of moves is valid (no conflicts)"""
        # Extract source and target positions
        sources = set()
        targets = set()
        
        for src, tgt in moves:
            # Check for overlapping sources or targets
            if src in sources or tgt in targets:
                return False
            sources.add(src)
            targets.add(tgt)
            
            # Check that no target is also a source for another move
            if tgt in sources or src in targets:
                return False
                
            # Check that target doesn't overlap with any non-moving block
            non_moving_blocks = state_set - sources
            if tgt in non_moving_blocks:
                return False
                
            # CRITICAL: Ensure no target is an obstacle
            if tgt in self.agent.obstacles:
                print(f"CRITICAL: Attempted to move to obstacle at {tgt} - move rejected")
                return False
        
        return True
    
    def _apply_moves(self, state_set, moves):
        """Apply a list of moves to the state"""
        new_state = state_set.copy()
        
        # First validate that we won't have any overlaps
        sources = set()
        targets = set()
        
        for src, tgt in moves:
            sources.add(src)
            targets.add(tgt)
            
        # CRITICAL: Check for obstacle collisions
        for tgt in targets:
            if tgt in self.agent.obstacles:
                print(f"CRITICAL: Attempted to move to obstacle at {tgt} - move rejected")
                return state_set  # Return original state if obstacle overlap detected
            
        # Ensure we're not creating duplicate positions
        # 1. No target should overlap with a non-moving block
        non_moving_blocks = state_set - sources
        if targets.intersection(non_moving_blocks):
            return state_set  # Return original state if overlap detected
            
        # 2. No duplicate targets
        if len(targets) != len(moves):
            return state_set  # Return original state if duplicate targets
        
        # Apply moves only if valid
        for src, tgt in moves:
            new_state.remove(src)
            new_state.add(tgt)
            
        # Verify we haven't lost any blocks
        if len(new_state) != len(state_set):
            print(f"WARNING: Block count changed from {len(state_set)} to {len(new_state)}")
            return state_set  # Return original state if blocks were lost
            
        return new_state
    
    def get_smart_chain_moves(self, state):
        """
        Generate chain moves where one block moves into the space of another
        while that block moves elsewhere, maintaining connectivity.
        Enhanced to prioritize clearing paths in tight spaces.
        """
        state_set = set(state)
        valid_moves = []
        
        # For each block, try to move it toward a goal position
        for pos in state_set:
            # Find closest goal position using obstacle-aware pathfinding
            min_dist = float('inf')
            closest_goal = None
            
            for goal_pos in self.agent.goal_state:
                if goal_pos not in state_set:  # Only consider unoccupied goals
                    dist = self.agent.obstacle_aware_distance(pos, goal_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_goal = goal_pos
            
            if not closest_goal:
                continue
                
            # Calculate direction toward goal
            dx = 1 if closest_goal[0] > pos[0] else -1 if closest_goal[0] < pos[0] else 0
            dy = 1 if closest_goal[1] > pos[1] else -1 if closest_goal[1] < pos[1] else 0
            
            # Try moving in that direction
            next_pos = (pos[0] + dx, pos[1] + dy)
            
            # Skip if next position is invalid (out of bounds or obstacle)
            if not self.agent.is_valid_position(next_pos):
                continue
            
            # If next position is occupied, try chain move
            if next_pos in state_set:
                chain_pos = (next_pos[0] + dx, next_pos[1] + dy)
                
                # Check if chain position is valid
                if chain_pos not in state_set and self.agent.is_valid_position(chain_pos):
                    # Create new state by moving both blocks
                    new_state_set = state_set.copy()
                    new_state_set.remove(pos)
                    new_state_set.remove(next_pos)
                    new_state_set.add(next_pos)
                    new_state_set.add(chain_pos)
                    
                    # Check if new state is connected
                    if self.agent.is_connected(new_state_set):
                        valid_moves.append(frozenset(new_state_set))
            else:
                # Direct move if next position is unoccupied
                new_state_set = state_set.copy()
                new_state_set.remove(pos)
                new_state_set.add(next_pos)
                
                # Check if new state is connected
                if self.agent.is_connected(new_state_set):
                    valid_moves.append(frozenset(new_state_set))
        
        return valid_moves
    
    def get_sliding_chain_moves(self, state):
        """
        Generate sliding chain moves where multiple blocks move in sequence
        to navigate tight spaces
        """
        state_set = set(state)
        valid_moves = []
        
        # Identify blocks that have reached their goal positions
        blocks_at_goal = state_set.intersection(self.agent.goal_state)
        
        # For each block, try to initiate a sliding chain
        for pos in state_set:
            # Skip if it's at a goal position
            if pos in blocks_at_goal:
                continue
                
            # Skip if it's a critical articulation point
            articulation_points = self.agent.get_articulation_points(state_set)
            if pos in articulation_points and len(articulation_points) <= 20:
                continue
                
            # Try sliding in each direction
            for dx, dy in self.agent.directions:
                # Only consider diagonal moves for sliding chains
                if dx != 0 and dy != 0:
                    # Define the sliding path (up to 20 steps)
                    path = []
                    current_pos = pos
                    for _ in range(20):  # Maximum chain length
                        next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                        # Stop if position is invalid (out of bounds or obstacle)
                        if not self.agent.is_valid_position(next_pos):
                            break
                        path.append(next_pos)
                        current_pos = next_pos
                    
                    # Try sliding the block along the path
                    for i, target_pos in enumerate(path):
                        # Skip if target is occupied
                        if target_pos in state_set:
                            continue
                            
                        # Create new state by moving the block
                        new_state_set = state_set.copy()
                        new_state_set.remove(pos)
                        new_state_set.add(target_pos)
                        
                        # Check if new state is connected and has the correct number of blocks
                        if self.agent.is_connected(new_state_set) and len(new_state_set) == len(state_set):
                            valid_moves.append(frozenset(new_state_set))
                        
                        # No need to continue if we can't reach this position
                        break
        
        return valid_moves
    
    def get_all_valid_moves(self, state):
        """
        Combine all move generation methods to maximize options
        """
        # Start with basic morphing moves
        basic_moves = self.get_valid_morphing_moves(state)
        
        # Add chain moves
        chain_moves = self.get_smart_chain_moves(state)
        
        # Add sliding chain moves
        sliding_moves = self.get_sliding_chain_moves(state)
        
        # Combine all moves (frozensets automatically handle duplicates)
        all_moves = list(set(basic_moves + chain_moves + sliding_moves))
        
        # Verify all moves have the correct number of blocks
        valid_moves = []
        for move in all_moves:
            if len(move) == len(state):
                valid_moves.append(move)
            else:
                print(f"WARNING: Invalid move with {len(move)} blocks instead of {len(state)}")
        
        return valid_moves
    
    def block_heuristic(self, state):
        """
        Heuristic for block movement phase:
        Accounts for obstacles between current position and goal
        """
        if not state:
            return float('inf')
        
        # Check if we've lost any blocks and apply proportional penalty
        if len(state) < len(self.agent.start_state):
            missing_blocks = len(self.agent.start_state) - len(state)
            missing_penalty = 10000 * missing_blocks  # Large penalty per missing block
            return missing_penalty  # Return large penalty proportional to missing blocks
                
        current_centroid = self.agent.calculate_centroid(state)
        goal_centroid_int = (int(self.agent.goal_centroid[0]), int(self.agent.goal_centroid[1]))
        
        # If no obstacles, use simple Manhattan distance
        if not self.agent.obstacles:
            return abs(current_centroid[0] - self.agent.goal_centroid[0]) + abs(current_centroid[1] - self.agent.goal_centroid[1])
        
        # With obstacles, calculate path distance to goal centroid
        # Round centroid to nearest grid cell for distance calculation
        current_centroid_int = (int(round(current_centroid[0])), int(round(current_centroid[1])))
        
        # Ensure centroid is within bounds
        current_centroid_int = (
            max(0, min(current_centroid_int[0], self.agent.grid_size[0]-1)),
            max(0, min(current_centroid_int[1], self.agent.grid_size[1]-1))
        )
        goal_centroid_int = (
            max(0, min(goal_centroid_int[0], self.agent.grid_size[0]-1)),
            max(0, min(goal_centroid_int[1], self.agent.grid_size[1]-1))
        )
        
        # Get obstacle-aware distance
        return self.agent.obstacle_aware_distance(current_centroid_int, goal_centroid_int)
    
    def improved_morphing_heuristic(self, state):
        """
        Improved heuristic for morphing phase:
        Uses bipartite matching to find optimal assignment of blocks to goal positions,
        now accounts for obstacles
        """
        if not state:
            return float('inf')
        
        # Check if we've lost any blocks and apply proportional penalty
        if len(state) < len(self.agent.start_state):
            missing_blocks = len(self.agent.start_state) - len(state)
            missing_penalty = 10000 * missing_blocks  # Large penalty per missing block
            return missing_penalty  # Return large penalty proportional to missing blocks
                
        state_list = list(state)
        goal_list = list(self.agent.goal_state)
        
        # Early exit if states have different sizes
        if len(state_list) != len(goal_list):
            return float('inf')
        
        # Reward blocks that are already in goal positions
        blocks_at_goal = set(state).intersection(self.agent.goal_state)
        goal_bonus = -len(blocks_at_goal) * 10  # Large negative value (bonus) for blocks in place
        
        # Build distance matrix with obstacle-aware distances
        distances = []
        for pos in state_list:
            # If block is already at a goal position, give it maximum preference
            # to stay where it is by assigning 0 distance to its current position
            # and high distance to all other positions
            if pos in self.agent.goal_state:
                row = [0 if goal_pos == pos else 1000 for goal_pos in goal_list]
            else:
                row = []
                for goal_pos in goal_list:
                    # Use obstacle-aware distance calculation
                    if self.agent.obstacles:
                        dist = self.agent.obstacle_aware_distance(pos, goal_pos)
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
        print("Starting Block Movement Phase...")
        start_time = time.time()

        # For disconnected goals, use the modified approach
        if self.agent.is_goal_disconnected:
            return self.agent.disconnected_block_movement_phase(time_limit)

        # Initialize A* search
        open_set = [(self.block_heuristic(self.agent.start_state), 0, self.agent.start_state)]
        closed_set = set()

        # Track path and g-scores
        g_score = {self.agent.start_state: 0}
        came_from = {self.agent.start_state: None}

        # We want to stop 1 grid cell before reaching the centroid
        min_distance = 1.0
        max_distance = 1.0

        while open_set and time.time() - start_time < time_limit:
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
    
            # Skip if already processed
            if current in closed_set:
                continue
        
            # Check if we're at the desired distance from the goal centroid
            current_centroid = self.agent.calculate_centroid(current)
            centroid_distance = (abs(current_centroid[0] - self.agent.goal_centroid[0]) + 
                            abs(current_centroid[1] - self.agent.goal_centroid[1]))
                        
            if min_distance <= centroid_distance <= max_distance:
                print(f"Block stopped 1 grid cell before goal centroid. Distance: {centroid_distance}")
                return self.agent.reconstruct_path(came_from, current)
        
            closed_set.add(current)
    
            # Process neighbor states (block moves)
            for neighbor in self.get_valid_block_moves(current):
                if neighbor in closed_set:
                    continue
            
                # Calculate tentative g-score
                tentative_g = g_score[current] + 1
        
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                
                    # Adjust heuristic to prefer states close to but not at the centroid
                    neighbor_centroid = self.agent.calculate_centroid(neighbor)
                    neighbor_distance = (abs(neighbor_centroid[0] - self.agent.goal_centroid[0]) + 
                                    abs(neighbor_centroid[1] - self.agent.goal_centroid[1]))
                
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
                state_centroid = self.agent.calculate_centroid(state)
                distance = (abs(state_centroid[0] - self.agent.goal_centroid[0]) + 
                            abs(state_centroid[1] - self.agent.goal_centroid[1]))
            
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
                best_centroid = self.agent.calculate_centroid(best_state)
                best_distance = (abs(best_centroid[0] - self.agent.goal_centroid[0]) + 
                                abs(best_centroid[1] - self.agent.goal_centroid[1]))
                print(f"Best block position found with centroid distance: {best_distance}")
                return self.agent.reconstruct_path(came_from, best_state)
    
        return [self.agent.start_state]  # No movement possible
    
    def smarter_morphing_phase(self, start_state, time_limit=15):
        """
        Improved Phase 2: Morph the block into the goal shape while maintaining connectivity
        Uses beam search and intelligent move generation with support for simultaneous moves
        Now with adaptive beam width based on obstacle density
        """
        print(f"Starting Smarter Morphing Phase with {self.agent.min_simultaneous_moves}-{self.agent.max_simultaneous_moves} simultaneous moves...")
        start_time = time.time()
        
        # Identify blocks already at goal positions in the start state
        self.agent.blocks_at_goal = set(start_state).intersection(self.agent.goal_state)
        print(f"Starting morphing with {len(self.agent.blocks_at_goal)} blocks already at goal positions")
        
        # Adapt beam width based on obstacle density
        adaptive_beam_width = self.agent.beam_width
        if len(self.agent.obstacles) > 0:
            # Increase beam width for environments with obstacles
            obstacle_density = len(self.agent.obstacles) / (self.agent.grid_size[0] * self.agent.grid_size[1])
            adaptive_beam_width = int(self.agent.beam_width * (1 + min(1.0, obstacle_density * 5)))
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
        max_blocks_at_goal = len(self.agent.blocks_at_goal)
        
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
            if current == self.agent.goal_state:
                print(f"Goal reached after {iterations} iterations!")
                return self.agent.reconstruct_path(came_from, current)
            
            # Check how many blocks are at goal positions in current state
            blocks_at_goal_current = len(set(current).intersection(self.agent.goal_state))
            
            # Update max_blocks_at_goal if we found a better state
            if blocks_at_goal_current > max_blocks_at_goal:
                max_blocks_at_goal = blocks_at_goal_current
                print(f"New maximum blocks at goal: {max_blocks_at_goal}/{len(self.agent.goal_state)}")
            
            # Check if this is the best state seen so far
            current_heuristic = self.improved_morphing_heuristic(current)
            if current_heuristic < best_heuristic or blocks_at_goal_current > len(self.agent.blocks_at_goal):
                best_state = current
                best_heuristic = current_heuristic
                self.agent.blocks_at_goal = set(current).intersection(self.agent.goal_state)
                last_improvement_time = time.time()
                
                # Print progress occasionally
                if iterations % 500 == 0:
                    print(f"Progress: h={best_heuristic}, blocks at goal={len(self.agent.blocks_at_goal)}/{len(self.agent.goal_state)}, iterations={iterations}")
                    
                # If we're very close to the goal, increase search intensity
                if best_heuristic < 5 * len(self.agent.goal_state):
                    adaptive_beam_width *= 2
            
            # Check for stagnation - more patient in obstacle-heavy environments
            stagnation_tolerance = time_limit * (0.3 + min(0.3, len(self.agent.obstacles) / 100))
            if time.time() - last_improvement_time > stagnation_tolerance:
                print("Search stagnated, restarting...")
                # Clear the beam and start from the best state
                open_set = [(best_heuristic, g_score[best_state], best_state)]
                last_improvement_time = time.time()
            
            # Limit iterations to prevent infinite loops
            if iterations >= self.agent.max_iterations:
                print(f"Reached max iterations ({self.agent.max_iterations})")
                break
                
            closed_set.add(current)
            
            # Get all valid moves
            neighbors = self.get_all_valid_moves(current)
            
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
                    blocks_at_goal_neighbor = len(set(neighbor).intersection(self.agent.goal_state))
                    
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
        return self.agent.reconstruct_path(came_from, best_state)