# cython: language_level=3
from forward_declarations cimport ConnectedMatterAgent

cdef class MovementPhases:
    cdef public:
        ConnectedMatterAgent agent
        
    # Function declarations
    cpdef list get_valid_block_moves(self, state)
    cpdef list get_valid_morphing_moves(self, state)
    cpdef list _generate_move_combinations(self, list single_moves, int k)
    cpdef bint _is_valid_move_combination(self, list moves, set state_set)
    cpdef set _apply_moves(self, set state_set, list moves)
    cpdef list get_smart_chain_moves(self, state)
    cpdef list get_sliding_chain_moves(self, state)
    cpdef list get_all_valid_moves(self, state)
    cpdef double block_heuristic(self, state)
    cpdef double improved_morphing_heuristic(self, state)
    cpdef list block_movement_phase(self, double time_limit=*)
    cpdef list smarter_morphing_phase(self, start_state, double time_limit=*)