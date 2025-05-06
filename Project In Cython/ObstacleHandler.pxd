# cython: language_level=3
from forward_declarations cimport ConnectedMatterAgent

cdef class ObstacleHandler:
    cdef public:
        ConnectedMatterAgent agent
        
    cpdef build_obstacle_maze(self)
    cpdef list calculate_distance_map(self, tuple target)
    cpdef double obstacle_aware_distance(self, tuple pos, tuple target)
    cpdef list find_clean_path(self, tuple start_pos, tuple end_pos, set obstacles)