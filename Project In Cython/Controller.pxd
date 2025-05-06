# cython: language_level=3
from forward_declarations cimport ConnectedMatterAgent

cdef class Controller:
    cdef public:
        ConnectedMatterAgent agent
        
    cpdef list execute_search(self, double time_limit=*)
    cpdef bint verify_path(self, list path)
    cpdef double path_cost(self, list path)
    cpdef list optimize_path(self, list path)
    cpdef list check_connectivity(self, list path)
    cpdef list check_obstacle_overlap(self, list path)
    cpdef list fix_path_issues(self, list path)