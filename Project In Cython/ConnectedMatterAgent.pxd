# cython: language_level=3
from forward_declarations cimport MovementPhases, ObstacleHandler, DisconnectedGoalHandler

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
        object start_state  # Changed from frozenset to object to avoid pxd errors
        object goal_state   # Changed from frozenset to object
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
        
    # Function declarations
    cpdef tuple calculate_centroid(self, positions)
    cpdef bint is_valid_position(self, tuple pos)
    cpdef bint is_connected(self, positions)
    cpdef set get_articulation_points(self, set state_set)
    cpdef list find_disconnected_components(self, positions)
    cpdef build_obstacle_maze(self)
    cpdef list reconstruct_path(self, dict came_from, current)
    cpdef list search(self, double time_limit=*)
    cpdef visualize_path(self, path, double interval=*)
    cpdef double obstacle_aware_distance(self, tuple pos, tuple target)
    cpdef list find_clean_path(self, tuple start_pos, tuple end_pos, set obstacles)