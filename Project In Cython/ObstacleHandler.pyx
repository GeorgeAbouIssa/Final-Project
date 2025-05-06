# cython: language_level=3
from collections import deque
import heapq

cdef class ObstacleHandler:
    cdef public:
        object agent  # Reference to the main agent
    
    def __init__(self, agent):
        self.agent = agent
    
    cpdef build_obstacle_maze(self):
        """Create a grid representation with obstacles for pathfinding"""
        cdef int x, y
        
        # Initialize obstacle maze grid
        self.agent.obstacle_maze = [[0 for _ in range(self.agent.grid_size[1])] for _ in range(self.agent.grid_size[0])]
        
        # Mark obstacle cells
        for x, y in self.agent.obstacles:
            if 0 <= x < self.agent.grid_size[0] and 0 <= y < self.agent.grid_size[1]:
                self.agent.obstacle_maze[x][y] = 1  # Mark obstacle cells
        
        # Clear the distance map cache when obstacles change
        self.agent.distance_map_cache = {}
        
    cpdef list calculate_distance_map(self, tuple target):
        """
        Calculate distance map from all cells to the target,
        accounting for obstacles (using BFS for accurate distances)
        """
        cdef list dist_map
        cdef object queue
        cdef set visited
        cdef tuple pos
        cdef int x, y, nx, ny, dist
        
        # Check if we've already computed this map
        if target in self.agent.distance_map_cache:
            return self.agent.distance_map_cache[target]
            
        # Initialize distance map with infinity
        dist_map = [[float('inf') for _ in range(self.agent.grid_size[1])] for _ in range(self.agent.grid_size[0])]
        
        # BFS to calculate distances
        queue = deque([(target, 0)])  # (position, distance)
        visited = {target}
        
        while queue:
            (x, y), dist = queue.popleft()
            dist_map[x][y] = dist
            
            # Check all adjacent cells based on topology
            for dx, dy in self.agent.directions:
                nx, ny = x + dx, y + dy
                
                # Skip if out of bounds or is an obstacle or already visited
                if not (0 <= nx < self.agent.grid_size[0] and 0 <= ny < self.agent.grid_size[1]):
                    continue
                if (nx, ny) in self.agent.obstacles or (nx, ny) in visited:
                    continue
                    
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
        
        # Cache the result
        self.agent.distance_map_cache[target] = dist_map
        return dist_map
        
    cpdef double obstacle_aware_distance(self, tuple pos, tuple target):
        """
        Calculate the distance between a position and a target,
        accounting for obstacles
        """
        cdef list dist_map
        
        # If no obstacles, use Manhattan distance for speed
        if not self.agent.obstacles:
            return abs(pos[0] - target[0]) + abs(pos[1] - target[1])
        
        # Get or calculate distance map for this target
        dist_map = self.calculate_distance_map(target)
    
        # Bounds check to avoid IndexError
        if not (0 <= pos[0] < self.agent.grid_size[0] and 0 <= pos[1] < self.agent.grid_size[1]):
            return float('inf')
        
        # If either position is an obstacle, return infinity
        if pos in self.agent.obstacles or target in self.agent.obstacles:
            return float('inf')
    
        return dist_map[pos[0]][pos[1]]
    
    cpdef list find_clean_path(self, tuple start_pos, tuple end_pos, set obstacles):
        """
        Find a clean path between two positions, avoiding all obstacles.
        
        Args:
            start_pos: Starting position
            end_pos: Target position
            obstacles: Set of positions to avoid
            
        Returns:
            List of positions forming a path, or None if no path found
        """
        cdef set all_obstacles, closed_set
        cdef list open_set, path
        cdef dict came_from, g_score
        cdef tuple current, neighbor
        cdef int tentative_g, f_score
        
        # Combine the passed obstacles with the agent's permanent obstacles
        all_obstacles = obstacles.union(self.agent.obstacles)
        
        # Special case: if end_pos is an obstacle and not start_pos, no path is possible
        if end_pos in all_obstacles and end_pos != start_pos:
            return None
            
        # Use A* to find a path
        open_set = []
        heapq.heappush(open_set, (0, start_pos))
        closed_set = set()
        came_from = {start_pos: None}
        g_score = {start_pos: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == end_pos:
                # Reconstruct path
                path = []
                while current:
                    path.append(current)
                    current = came_from.get(current)
                path.reverse()
                
                # Validate path - make sure no obstacles
                if any(pos in self.agent.obstacles for pos in path[1:-1]):  # Skip start and end
                    print("WARNING: Path contains obstacles - rejecting path")
                    return None
                    
                return path
            
            closed_set.add(current)
            
            for dx, dy in self.agent.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if neighbor in closed_set:
                    continue
                
                # Skip if out of bounds
                if not (0 <= neighbor[0] < self.agent.grid_size[0] and 
                        0 <= neighbor[1] < self.agent.grid_size[1]):
                    continue
                
                # Skip if obstacle (unless it's the target)
                if neighbor in all_obstacles and neighbor != end_pos:
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + abs(neighbor[0] - end_pos[0]) + abs(neighbor[1] - end_pos[1])
                    heapq.heappush(open_set, (f_score, neighbor))
        
        # No path found
        return None