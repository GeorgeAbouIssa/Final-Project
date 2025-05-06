from collections import deque

class ObstacleHandler:
    def __init__(self, grid_size, obstacles, directions):
        self.grid_size = grid_size
        self.obstacles = obstacles if obstacles else set()
        self.directions = directions
        self.obstacle_maze = None
        self.distance_map_cache = {}
    
    def build_obstacle_maze(self):
        """Create a grid representation with obstacles for pathfinding"""
        self.obstacle_maze = [[0 for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        for x, y in self.obstacles:
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                self.obstacle_maze[x][y] = 1  # Mark obstacle cells
        
        # Clear the distance map cache when obstacles change
        self.distance_map_cache = {}
        
    def calculate_distance_map(self, target):
        """
        Calculate distance map from all cells to the target,
        accounting for obstacles (using BFS for accurate distances)
        """
        # Check if we've already computed this map
        if target in self.distance_map_cache:
            return self.distance_map_cache[target]
            
        # Initialize distance map with infinity
        dist_map = [[float('inf') for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        
        # BFS to calculate distances
        queue = deque([(target, 0)])  # (position, distance)
        visited = {target}
        
        while queue:
            (x, y), dist = queue.popleft()
            dist_map[x][y] = dist
            
            # Check all adjacent cells based on topology
            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                
                # Skip if out of bounds or is an obstacle or already visited
                if not (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]):
                    continue
                if (nx, ny) in self.obstacles or (nx, ny) in visited:
                    continue
                    
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
        
        # Cache the result
        self.distance_map_cache[target] = dist_map
        return dist_map
        
    def obstacle_aware_distance(self, pos, target):
        """
        Calculate the distance between a position and a target,
        accounting for obstacles
        """
        # If no obstacles, use Manhattan distance for speed
        if not self.obstacles:
            return abs(pos[0] - target[0]) + abs(pos[1] - target[1])
        
        # Get or calculate distance map for this target
        dist_map = self.calculate_distance_map(target)
    
        # Bounds check to avoid IndexError
        if not (0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]):
            return float('inf')
    
        return dist_map[pos[0]][pos[1]]