import heapq
import time
import matplotlib.pyplot as plt
from collections import deque
from threading import Lock
import concurrent.futures
from functools import partial

# Import the modules we've created
from ObstacleHandler import ObstacleHandler
from MovementPhases import MovementPhases
from DisconnectedGoalSequential import DisconnectedGoalSequential
from DisconnectedGoalMultiThreaded import DisconnectedGoalMultiThreaded

class ConnectedMatterAgent:
    def __init__(self, grid_size, start_positions, goal_positions, topology="moore", max_simultaneous_moves=1, min_simultaneous_moves=1, obstacles=None):
        self.grid_size = grid_size
        self.start_positions = list(start_positions)
        self.goal_positions = list(goal_positions)
        self.topology = topology
        self.max_simultaneous_moves = max_simultaneous_moves
        self.min_simultaneous_moves = min(min_simultaneous_moves, max_simultaneous_moves)  # Ensure min <= max
        self.obstacles = set(obstacles) if obstacles else set()
        
        # Set moves based on topology
        if self.topology == "moore":
            self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:  # Von Neumann
            self.directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            
        # Initialize the start and goal states
        self.start_state = frozenset((x, y) for x, y in start_positions)
        self.goal_state = frozenset((x, y) for x, y in goal_positions)
        
        # Check if goal state is disconnected and find components
        self.goal_components = self.find_disconnected_components(self.goal_state)
        self.is_goal_disconnected = len(self.goal_components) > 1
        
        if self.is_goal_disconnected:
            print(f"Goal state has {len(self.goal_components)} disconnected components")
            # Calculate centroids for each component
            self.component_centroids = [self.calculate_centroid(comp) for comp in self.goal_components]
            # Calculate the overall goal centroid
            self.goal_centroid = self.calculate_centroid(self.goal_positions)
        else:
            # Calculate the centroid of the goal positions for block movement phase
            # Using exact position calculation instead of average to ensure precise positioning
            self.goal_centroid = self.calculate_centroid(self.goal_positions)
        
        # Cache for valid moves to avoid recomputation
        self.valid_moves_cache = {}
        
        # For optimizing the search
        self.articulation_points_cache = {}
        self.connectivity_check_cache = {}
        
        # Enhanced parameters for improved search
        self.beam_width = 800  # Increased beam width for better exploration
        self.max_iterations = 200000  # Limit iterations to prevent infinite loops
        
        # Track blocks that have reached their goal positions
        self.blocks_at_goal = set()
        
        # Initialize handlers
        self.obstacle_handler = ObstacleHandler(grid_size, self.obstacles, self.directions)
        if obstacles:
            self.obstacle_handler.build_obstacle_maze()
        
        # Initialize the movement phase handler
        self.movement_phases = MovementPhases(self)
        
        # Initialize disconnected goal handlers
        self.sequential_handler = DisconnectedGoalSequential(self)
        self.disconnected_handler = DisconnectedGoalMultiThreaded(self)
        
    def calculate_centroid(self, positions):
        """Calculate the centroid (average position) of a set of positions"""
        if not positions:
            return (0, 0)
        x_sum = sum(pos[0] for pos in positions)
        y_sum = sum(pos[1] for pos in positions)
        return (x_sum / len(positions), y_sum / len(positions))
    
    def is_connected(self, positions):
        """Check if all positions are connected using BFS"""
        if not positions:
            return True
            
        # Use cache if available
        positions_hash = hash(frozenset(positions))
        if positions_hash in self.connectivity_check_cache:
            return self.connectivity_check_cache[positions_hash]
            
        # Convert to set for O(1) lookup
        positions_set = set(positions)
        
        # Start BFS from first position
        start = next(iter(positions_set))
        visited = {start}
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            
            # Check all adjacent positions
            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in positions_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # All positions should be visited if connected
        is_connected = len(visited) == len(positions_set)
        
        # Cache the result
        self.connectivity_check_cache[positions_hash] = is_connected
        return is_connected
    
    def get_articulation_points(self, state_set):
        """
        Find articulation points (critical points that if removed would disconnect the structure)
        Uses a modified DFS algorithm
        """
        state_hash = hash(frozenset(state_set))
        if state_hash in self.articulation_points_cache:
            return self.articulation_points_cache[state_hash]
            
        if len(state_set) <= 2:  # All points are critical in structures of size 1-2
            self.articulation_points_cache[state_hash] = set(state_set)
            return set(state_set)
            
        articulation_points = set()
        visited = set()
        discovery = {}
        low = {}
        parent = {}
        time = [0]  # Using list to allow modification inside nested function
        
        def dfs(u, time):
            children = 0
            visited.add(u)
            discovery[u] = low[u] = time[0]
            time[0] += 1
            
            # Visit all neighbors
            for dx, dy in self.directions:
                v = (u[0] + dx, u[1] + dy)
                if v in state_set:
                    if v not in visited:
                        children += 1
                        parent[v] = u
                        dfs(v, time)
                        
                        # Check if subtree rooted with v has a connection to ancestors of u
                        low[u] = min(low[u], low[v])
                        
                        # u is an articulation point if:
                        # 1) u is root and has two or more children
                        # 2) u is not root and low value of one of its children >= discovery value of u
                        if parent.get(u) is None and children > 1:
                            articulation_points.add(u)
                        if parent.get(u) is not None and low[v] >= discovery[u]:
                            articulation_points.add(u)
                            
                    elif v != parent.get(u):  # Update low value of u for parent function calls
                        low[u] = min(low[u], discovery[v])
        
        # Call DFS for all vertices
        for point in state_set:
            if point not in visited:
                dfs(point, time)
                
        self.articulation_points_cache[state_hash] = articulation_points
        return articulation_points
    
    def get_valid_block_moves(self, state):
        """
        Generate valid moves for the entire block of elements
        A valid block move shifts all elements in the same direction while maintaining connectivity
        """
        valid_moves = []
        state_list = list(state)
        
        # Try moving the entire block in each direction
        for dx, dy in self.directions:
            # Calculate new positions after moving
            new_positions = [(pos[0] + dx, pos[1] + dy) for pos in state_list]
            
            # Check if all new positions are valid (within bounds, not occupied by obstacles)
            all_valid = all(0 <= pos[0] < self.grid_size[0] and 
                            0 <= pos[1] < self.grid_size[1] and
                            pos not in self.obstacles for pos in new_positions)
            
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
        Now optimized for obstacle environments
        """
        state_key = hash(state)
        if state_key in self.valid_moves_cache:
            return self.valid_moves_cache[state_key]
            
        # Get single block moves first
        single_moves = []
        state_set = set(state)
        
        # Identify blocks that have reached their goal positions
        blocks_at_goal = state_set.intersection(self.goal_state)
        self.blocks_at_goal = blocks_at_goal
        
        # Find non-critical points that can move without breaking connectivity
        articulation_points = self.get_articulation_points(state_set)
        
        # Don't move blocks that have reached their goal positions
        # unless they're the only blocks we can move (to avoid deadlock)
        movable_points = state_set - articulation_points - blocks_at_goal
        
        # If all points are critical or at goal, try moving critical points that aren't at goals
        if not movable_points:
            for point in articulation_points - blocks_at_goal:
                # Try removing and see if structure remains connected
                temp_state = state_set.copy()
                temp_state.remove(point)
                if self.is_connected(temp_state):
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
                    if self.is_connected(temp_state):
                        movable_points.add(point)
            
            # If still stuck, try critical goal blocks as absolute last resort
            if not movable_points:
                for point in blocks_at_goal.intersection(articulation_points):
                    temp_state = state_set.copy()
                    temp_state.remove(point)
                    if self.is_connected(temp_state):
                        movable_points.add(point)
        
        # Generate single block moves, prioritizing moves toward the goal
        for point in movable_points:
            # Find closest goal position for this point
            closest_goal = None
            min_dist = float('inf')
            
            for goal_pos in self.goal_state:
                if goal_pos not in state_set:  # Only consider unoccupied goals
                    if self.obstacles:
                        dist = self.obstacle_handler.obstacle_aware_distance(point, goal_pos)
                    else:
                        dist = abs(point[0] - goal_pos[0]) + abs(point[1] - goal_pos[1])
                        
                    if dist < min_dist:
                        min_dist = dist
                        closest_goal = goal_pos
                        
            # Prioritize directions toward the closest goal
            ordered_directions = self.directions.copy()
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
                
                # Skip if out of bounds or is an obstacle
                if not (0 <= new_pos[0] < self.grid_size[0] and 
                        0 <= new_pos[1] < self.grid_size[1]):
                    continue
                    
                # Skip if position is an obstacle
                if new_pos in self.obstacles:
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
                for adj_dx, adj_dy in self.directions:
                    adj_pos = (new_pos[0] + adj_dx, new_pos[1] + adj_dy)
                    if adj_pos in new_state_set and adj_pos != new_pos:
                        has_adjacent = True
                        break
                
                # Only consider moves that maintain connectivity
                if has_adjacent and self.is_connected(new_state_set):
                    single_moves.append((point, new_pos))
                    
        # Start with empty valid moves list
        valid_moves = []
        
        # In dense obstacle environments, more simultaneous moves could be better
        # Increase minimum number of simultaneous moves based on obstacle density
        local_min_moves = self.min_simultaneous_moves
        if len(self.obstacles) > 20 and local_min_moves == 1:
            local_min_moves = min(2, self.max_simultaneous_moves)
            
        # Generate multi-block moves
        for k in range(local_min_moves, min(self.max_simultaneous_moves + 1, len(single_moves) + 1)):
            # Generate combinations of k moves
            for combo in self._generate_move_combinations(single_moves, k):
                # Check if the combination is valid (no conflicts)
                if self._is_valid_move_combination(combo, state_set):
                    # Apply the combination and check connectivity
                    new_state = self._apply_moves(state_set, combo)
                    if self.is_connected(new_state):
                        valid_moves.append(frozenset(new_state))
        
        # If no valid moves with min_simultaneous_moves, fallback to single moves if allowed
        if not valid_moves and self.min_simultaneous_moves == 1:
            valid_moves = [frozenset(self._apply_moves(state_set, [move])) for move in single_moves]
        
        # Cache results
        self.valid_moves_cache[state_key] = valid_moves
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
            
            for goal_pos in self.goal_state:
                if goal_pos not in state_set:  # Only consider unoccupied goals
                    if self.obstacles:
                        dist = self.obstacle_handler.obstacle_aware_distance(pos, goal_pos)
                    else:
                        dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
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
            
            # If next position is occupied, try chain move
            if next_pos in state_set:
                chain_pos = (next_pos[0] + dx, next_pos[1] + dy)
                
                # Check if chain position is valid
                if chain_pos not in state_set and chain_pos not in self.obstacles:
                    # Create new state by moving both blocks
                    new_state_set = state_set.copy()
                    new_state_set.remove(pos)
                    new_state_set.remove(next_pos)
                    new_state_set.add(next_pos)
                    new_state_set.add(chain_pos)
                    
                    # Check if new state is connected
                    if self.is_connected(new_state_set):
                        valid_moves.append(frozenset(new_state_set))
            else:
                # Direct move if next position is unoccupied
                new_state_set = state_set.copy()
                new_state_set.remove(pos)
                new_state_set.add(next_pos)
                
                # Check if new state is connected
                if self.is_connected(new_state_set):
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
        blocks_at_goal = state_set.intersection(self.goal_state)
        
        # For each block, try to initiate a sliding chain
        for pos in state_set:
            # Skip if it's at a goal position
            if pos in blocks_at_goal:
                continue
                
            # Skip if it's a critical articulation point
            articulation_points = self.get_articulation_points(state_set)
            if pos in articulation_points and len(articulation_points) <= 20:
                continue
                
            # Try sliding in each direction
            for dx, dy in self.directions:
                # Only consider diagonal moves for sliding chains
                if dx != 0 and dy != 0:
                    # Define the sliding path (up to 3 steps)
                    path = []
                    current_pos = pos
                    for _ in range(20):  # Maximum chain length
                        next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                        # Stop if out of bounds or is an obstacle
                        if not (0 <= next_pos[0] < self.grid_size[0] and 
                                0 <= next_pos[1] < self.grid_size[1]):
                            break
                        if next_pos in self.obstacles:
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
                        if self.is_connected(new_state_set) and len(new_state_set) == len(state_set):
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
    
    def find_disconnected_components(self, positions):
        """
        Find all disconnected components in a set of positions using BFS
        Returns a list of sets, where each set contains positions in one component
        """
        if not positions:
            return []
            
        positions_set = set(positions)
        components = []
        
        while positions_set:
            # Start a new component
            component = set()
            start = next(iter(positions_set))
            
            # BFS to find all connected positions
            queue = deque([start])
            component.add(start)
            positions_set.remove(start)
            
            while queue:
                current = queue.popleft()
                
                # Check all adjacent positions
                for dx, dy in self.directions:
                    neighbor = (current[0] + dx, current[1] + dy)
                    if neighbor in positions_set:
                        component.add(neighbor)
                        positions_set.remove(neighbor)
                        queue.append(neighbor)
            
            # Add the component to the list
            components.append(component)
        
        return components
    
    def reconstruct_path(self, came_from, current):
        """
        Reconstruct the path from start to goal
        """
        path = []
        while current:
            path.append(list(current))
            current = came_from.get(current)
        
        path.reverse()
        return path
    
    def search(self, time_limit=30):
        """
        Main search method combining block movement and smarter morphing
        Now with dynamic time allocation based on obstacles
        """
        # Build obstacle maze representation if not already done
        if self.obstacles and not self.obstacle_handler.obstacle_maze:
            self.obstacle_handler.build_obstacle_maze()
            
        # Dynamically allocate time based on obstacle density
        block_time_ratio = 0.3  # Default 30% for block movement
        
        # If there are obstacles, allocate more time for movement phase
        if len(self.obstacles) > 0:
            obstacle_density = len(self.obstacles) / (self.grid_size[0] * self.grid_size[1])
            # Allocate up to 50% for block movement in dense obstacle environments
            block_time_ratio = min(0.5, 0.3 + obstacle_density * 0.5)
            
        # For disconnected goals, check if fully connected and choose approach accordingly
        if self.is_goal_disconnected:
            # Check if the goal state is connected
            if self.is_connected(self.goal_state):
                print("Goal appears disconnected but is actually connected, using standard search")
                # If connected, use standard search approach
            else:
                try:
                    # First try the multithreaded handler
                    return self.disconnected_handler.search_disconnected_goal_multithreaded(time_limit)
                except Exception as e:
                    print(f"Multithreaded search failed: {e}")
                    try:
                        # Fall back to sequential handler
                        return self.sequential_handler.search_disconnected_goal_sequential(time_limit)
                    except Exception as e:
                        print(f"Sequential search failed: {e}")
                        # If both fail, fall back to standard search
                        print("Falling back to standard search")
            
        block_time_limit = time_limit * block_time_ratio
        morphing_time_limit = time_limit * (1 - block_time_ratio)
        
        print(f"Time allocation: {block_time_ratio:.1%} block movement, {1-block_time_ratio:.1%} morphing")
        
        # Phase 1: Block Movement
        block_path = self.movement_phases.block_movement_phase(block_time_limit)
        
        if not block_path:
            print("Block movement phase failed!")
            return None
        
        # Get the final state from block movement phase
        block_final_state = frozenset(block_path[-1])
        
        # Phase 2: Smarter Morphing
        morphing_path = self.movement_phases.smarter_morphing_phase(block_final_state, morphing_time_limit)
        
        if not morphing_path:
            print("Morphing phase failed!")
            return block_path
        
        # Combine paths (remove duplicate state at transition)
        combined_path = block_path[:-1] + morphing_path
        
        # Verify block count is consistent throughout the path
        expected_count = len(self.start_state)
        for i, state in enumerate(combined_path):
            if len(state) != expected_count:
                print(f"WARNING: State {i} has {len(state)} blocks instead of {expected_count}")
                # Fix the state by using the previous valid state
                if i > 0:
                    combined_path[i] = combined_path[i-1]
        
        return combined_path
    
    def visualize_path(self, path, interval=0.5):
        """
        Visualize the path as an animation
        """
        if not path:
            print("No path to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.ion()  # Turn on interactive mode
    
        # Get bounds for plotting
        min_x, max_x = 0, self.grid_size[0] - 1
        min_y, max_y = 0, self.grid_size[1] - 1
    
        # Show initial state
        ax.clear()
        ax.set_xlim(min_x - 0.5, max_x + 0.5)
        ax.set_ylim(min_y - 0.5, max_y + 0.5)
        ax.grid(True)
    
        # Track and display blocks at goal positions differently
        # Draw goal positions (as outlines)
        goal_rects = []
        for pos in self.goal_positions:
            rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(rect)
            goal_rects.append(rect)
    
        # Draw current positions
        current_positions = path[0]
        
        # Determine which blocks are at goal positions
        blocks_at_goal = [pos for pos in current_positions if (pos[0], pos[1]) in self.goal_state]
        blocks_not_at_goal = [pos for pos in current_positions if (pos[0], pos[1]) not in self.goal_state]
        
        # Draw blocks at goal positions (green filled squares)
        goal_block_rects = []
        for pos in blocks_at_goal:
            rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, facecolor='green', alpha=0.7)
            ax.add_patch(rect)
            goal_block_rects.append(rect)
            
        # Draw other blocks (blue squares)
        non_goal_rects = []
        for pos in blocks_not_at_goal:
            rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, facecolor='blue', alpha=0.7)
            ax.add_patch(rect)
            non_goal_rects.append(rect)
            
        # Draw obstacles (black squares)
        obstacle_rects = []
        for pos in self.obstacles:
            rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, facecolor='black')
            ax.add_patch(rect)
            obstacle_rects.append(rect)
        
        ax.set_title(f"Step 0/{len(path)-1} - {len(blocks_at_goal)} blocks at goal")
        plt.draw()
        plt.pause(interval)
    
        # Animate the path
        for i in range(1, len(path)):
            # Verify block count is consistent
            if len(path[i]) != len(path[0]):
                print(f"Warning: State {i} has {len(path[i])} blocks instead of {len(path[0])}")
                # If block count is inconsistent, skip this frame
                continue
                
            # Update positions
            new_positions = path[i]
        
            # Clear previous blocks
            for rect in goal_block_rects + non_goal_rects:
                rect.remove()
            
            # Determine which blocks are at goal positions
            blocks_at_goal = [pos for pos in new_positions if (pos[0], pos[1]) in self.goal_state]
            blocks_not_at_goal = [pos for pos in new_positions if (pos[0], pos[1]) not in self.goal_state]
            
            # Draw blocks at goal positions (green filled squares)
            goal_block_rects = []
            for pos in blocks_at_goal:
                rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, facecolor='green', alpha=0.7)
                ax.add_patch(rect)
                goal_block_rects.append(rect)
                
            # Draw other blocks (blue squares)
            non_goal_rects = []
            for pos in blocks_not_at_goal:
                rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, facecolor='blue', alpha=0.7)
                ax.add_patch(rect)
                non_goal_rects.append(rect)
            
            ax.set_title(f"Step {i}/{len(path)-1} - {len(blocks_at_goal)} blocks at goal")
            plt.draw()
            plt.pause(interval)
    
        plt.ioff()  # Turn off interactive mode
        plt.show(block=True)
    
    def disconnected_block_movement_phase(self, time_limit=15):
        """
        Modified Phase 1 for disconnected goal states:
        Moves the entire block toward a strategic position for splitting
        """
        print("Starting Disconnected Block Movement Phase...")
        start_time = time.time()
        
        # Find the closest goal component to the start state
        closest_component_idx = self.find_closest_component()
        closest_component = self.goal_components[closest_component_idx]
        closest_centroid = self.component_centroids[closest_component_idx]
        
        # Determine if vertical positioning is better based on y-axis centroids
        all_components_y = [centroid[1] for centroid in self.component_centroids]
        overall_y = self.goal_centroid[1]
        
        # Check if centroid of all shapes is closer to y level of the closest shape
        use_vertical_approach = abs(overall_y - closest_centroid[1]) < sum([abs(y - closest_centroid[1]) for y in all_components_y]) / len(all_components_y)
        
        if use_vertical_approach:
            print("Using vertical approach for block movement")
            # Target position is at the overall centroid with y-level of closest component
            target_centroid = (self.goal_centroid[0], closest_centroid[1])
        else:
            print("Using standard approach for block movement")
            # Target position is the overall centroid
            target_centroid = self.goal_centroid
            
        # Cache original goal centroid and temporarily replace with target
        original_centroid = self.goal_centroid
        self.goal_centroid = target_centroid
        
        # Use standard A* search but with the modified target
        open_set = [(self.movement_phases.block_heuristic(self.start_state), 0, self.start_state)]
        closed_set = set()
        g_score = {self.start_state: 0}
        came_from = {self.start_state: None}
        
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
            current_centroid = self.calculate_centroid(current)
            centroid_distance = (abs(current_centroid[0] - target_centroid[0]) + 
                            abs(current_centroid[1] - target_centroid[1]))
                        
            if min_distance <= centroid_distance <= max_distance:
                print(f"Block stopped at strategic position. Distance: {centroid_distance}")
                # Restore original goal centroid
                self.goal_centroid = original_centroid
                return self.reconstruct_path(came_from, current)
        
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
                
                    # Adjust heuristic to prefer states close to target
                    neighbor_centroid = self.calculate_centroid(neighbor)
                    neighbor_distance = (abs(neighbor_centroid[0] - target_centroid[0]) + 
                                    abs(neighbor_centroid[1] - target_centroid[1]))
                
                    # Penalize distances that are too small
                    distance_penalty = 0
                    if neighbor_distance < min_distance:
                        distance_penalty = 10 * (min_distance - neighbor_distance)
                
                    # Calculate Manhattan distance to target
                    if self.obstacles:
                        neighbor_centroid_int = (int(round(neighbor_centroid[0])), int(round(neighbor_centroid[1])))
                        target_centroid_int = (int(round(target_centroid[0])), int(round(target_centroid[1])))
                        
                        # Ensure centroids are within bounds
                        neighbor_centroid_int = (
                            max(0, min(neighbor_centroid_int[0], self.grid_size[0]-1)),
                            max(0, min(neighbor_centroid_int[1], self.grid_size[1]-1))
                        )
                        target_centroid_int = (
                            max(0, min(target_centroid_int[0], self.grid_size[0]-1)),
                            max(0, min(target_centroid_int[1], self.grid_size[1]-1))
                        )
                        
                        adjusted_heuristic = self.obstacle_handler.obstacle_aware_distance(neighbor_centroid_int, target_centroid_int) + distance_penalty
                    else:
                        adjusted_heuristic = neighbor_distance + distance_penalty
                        
                    f_score = tentative_g + adjusted_heuristic
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # Restore original goal centroid
        self.goal_centroid = original_centroid
        
        # If we exit the loop, find the best available state
        if came_from:
            best_state = None
            best_distance_diff = float('inf')
        
            for state in came_from.keys():
                state_centroid = self.calculate_centroid(state)
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
                return self.reconstruct_path(came_from, best_state)

        return [self.start_state]  # No movement possible
        
    def find_closest_component(self):
        """Find the index of the closest goal component to the start state"""
        start_centroid = self.calculate_centroid(self.start_state)
        min_distance = float('inf')
        closest_idx = 0
        
        for idx, centroid in enumerate(self.component_centroids):
            distance = abs(start_centroid[0] - centroid[0]) + abs(start_centroid[1] - centroid[1])
            if distance < min_distance:
                min_distance = distance
                closest_idx = idx
                
        return closest_idx