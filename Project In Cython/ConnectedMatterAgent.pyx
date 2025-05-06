# cython: language_level=3
import heapq
import time
import matplotlib.pyplot as plt
from collections import deque
from libc.math cimport abs as c_abs

# Import component handlers
from MovementPhases cimport MovementPhases
from ObstacleHandler cimport ObstacleHandler
from DisconnectedGoal cimport DisconnectedGoalHandler

cdef class ConnectedMatterAgent:
    cdef public:
        tuple grid_size
        list start_positions
        list goal_positions
        str topology
        int max_simultaneous_moves
        int min_simultaneous_moves
        set obstacles
        list directions
        frozenset start_state
        frozenset goal_state
        dict valid_moves_cache
        dict articulation_points_cache
        dict connectivity_check_cache
        dict distance_map_cache
        int beam_width
        int max_iterations
        set blocks_at_goal
        object obstacle_maze
        MovementPhases movement
        ObstacleHandler obstacle_handler
        list goal_components
        bint is_goal_disconnected
        list component_centroids
        tuple goal_centroid
        DisconnectedGoalHandler disconnected_goal
        
    def __init__(self, grid_size, start_positions, goal_positions, str topology="moore", int max_simultaneous_moves=1, int min_simultaneous_moves=1, obstacles=None):
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
        
        # Cache for algorithmic optimization
        self.valid_moves_cache = {}
        self.articulation_points_cache = {}
        self.connectivity_check_cache = {}
        self.distance_map_cache = {}
        
        # Enhanced parameters for improved search
        self.beam_width = 800  # Increased beam width for better exploration
        self.max_iterations = 200000  # Limit iterations to prevent infinite loops
        
        # Track blocks that have reached their goal positions
        self.blocks_at_goal = set()
        
        # For obstacle pathfinding optimization
        self.obstacle_maze = None
        
        # Initialize component handlers
        self.movement = MovementPhases(self)
        self.obstacle_handler = ObstacleHandler(self)
        
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
            self.goal_centroid = self.calculate_centroid(self.goal_positions)
            
        # Initialize disconnected goal handler after centroids are calculated
        self.disconnected_goal = DisconnectedGoalHandler(self)
        
        # Build the obstacle maze after all handlers are initialized
        if obstacles:
            self.build_obstacle_maze()
            
    cpdef tuple calculate_centroid(self, positions):
        """Calculate the centroid (average position) of a set of positions"""
        cdef double x_sum, y_sum
        cdef int count
        
        if not positions:
            return (0, 0)
        x_sum = sum(pos[0] for pos in positions)
        y_sum = sum(pos[1] for pos in positions)
        count = len(positions)
        return (x_sum / count, y_sum / count)
    
    cpdef bint is_valid_position(self, tuple pos):
        """Check if a position is valid (in bounds and not an obstacle)"""
        return (0 <= pos[0] < self.grid_size[0] and 
                0 <= pos[1] < self.grid_size[1] and
                pos not in self.obstacles)
    
    cpdef bint is_connected(self, positions):
        """Check if all positions are connected using BFS"""
        cdef int positions_hash
        cdef set positions_set, visited
        cdef object queue
        cdef tuple current, neighbor, start
        cdef bint is_connected
        
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
    
    cpdef set get_articulation_points(self, set state_set):
        """
        Find articulation points (critical points that if removed would disconnect the structure)
        Uses a modified DFS algorithm
        """
        cdef int state_hash
        cdef set articulation_points
        cdef set visited
        cdef dict discovery, low, parent
        cdef list time
        cdef tuple u, v
        cdef int children
        
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
            cdef int children = 0
            cdef tuple v
            
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
    
    cpdef list find_disconnected_components(self, positions):
        """
        Find all disconnected components in a set of positions using BFS
        Returns a list of sets, where each set contains positions in one component
        """
        cdef set positions_set
        cdef list components
        cdef set component
        cdef tuple start, current, neighbor
        cdef object queue
        
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
    
    cpdef build_obstacle_maze(self):
        """Create a grid representation with obstacles for pathfinding"""
        self.obstacle_handler.build_obstacle_maze()
    
    cpdef list reconstruct_path(self, dict came_from, current):
        """
        Reconstruct the path from start to goal
        """
        cdef list path = []
        
        while current:
            path.append(list(current))
            current = came_from.get(current)
        
        path.reverse()
        return path
    
    cpdef list search(self, double time_limit=30):
        """
        Main search method combining block movement and smarter morphing
        Now with dynamic time allocation based on obstacles
        """
        cdef double block_time_ratio, obstacle_density
        cdef double block_time_limit, morphing_time_limit
        cdef list block_path, morphing_path, combined_path
        cdef object block_final_state
        cdef int expected_count, i
        
        # Build obstacle maze representation if not already done
        if self.obstacles and not self.obstacle_maze:
            self.build_obstacle_maze()
            
        # Dynamically allocate time based on obstacle density
        block_time_ratio = 0.3  # Default 30% for block movement
        
        # If there are obstacles, allocate more time for movement phase
        if len(self.obstacles) > 0:
            obstacle_density = len(self.obstacles) / (self.grid_size[0] * self.grid_size[1])
            # Allocate up to 50% for block movement in dense obstacle environments
            block_time_ratio = min(0.5, 0.3 + obstacle_density * 0.5)
            
        # For disconnected goals, adjust time allocation
        if self.is_goal_disconnected:
            return self.disconnected_goal.search_disconnected_goal(time_limit)
            
        block_time_limit = time_limit * block_time_ratio
        morphing_time_limit = time_limit * (1 - block_time_ratio)
        
        print(f"Time allocation: {block_time_ratio:.1%} block movement, {1-block_time_ratio:.1%} morphing")
        
        # Phase 1: Block Movement
        block_path = self.movement.block_movement_phase(block_time_limit)
        
        if not block_path:
            print("Block movement phase failed!")
            return None
        
        # Get the final state from block movement phase
        block_final_state = frozenset(block_path[-1])
        
        # Phase 2: Smarter Morphing
        morphing_path = self.movement.smarter_morphing_phase(block_final_state, morphing_time_limit)
        
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
    
    cpdef visualize_path(self, path, double interval=0.5):
        """
        Visualize the path as an animation
        """
        cdef int min_x, max_x, min_y, max_y, i
        cdef list goal_rects, obstacle_rects, goal_block_rects, non_goal_rects
        cdef list new_positions, blocks_at_goal, blocks_not_at_goal
        
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
        
        # Draw obstacles
        obstacle_rects = []
        for pos in self.obstacles:
            rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, facecolor='red', alpha=0.7)
            ax.add_patch(rect)
            obstacle_rects.append(rect)
    
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
    
    cpdef double obstacle_aware_distance(self, tuple pos, tuple target):
        """
        Calculate the distance between a position and a target,
        accounting for obstacles
        """
        return self.obstacle_handler.obstacle_aware_distance(pos, target)
    
    cpdef list find_clean_path(self, tuple start_pos, tuple end_pos, set obstacles):
        """
        Find a clean path between two positions, avoiding all obstacles.
        """
        return self.obstacle_handler.find_clean_path(start_pos, end_pos, obstacles)