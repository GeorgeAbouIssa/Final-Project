# cython: language_level=3
from forward_declarations cimport ConnectedMatterAgent

cdef class DisconnectedGoalHandler:
    cdef public:
        ConnectedMatterAgent agent
        
    cpdef list disconnected_block_movement_phase(self, double time_limit=*)
    cpdef int find_closest_component(self)
    cpdef dict assign_blocks_to_components(self, state)
    cpdef list plan_disconnect_moves(self, state, dict assignments)
    cpdef double component_morphing_heuristic(self, state, goal_component)
    cpdef list component_morphing_phase(self, start_state, goal_component, double time_limit=*)
    cpdef list search_disconnected_goal(self, double time_limit=*)
    cpdef bint is_full_disconnected_goal_reached(self, state)
    cpdef list get_disconnected_valid_moves(self, state, goal_components)
    cpdef list sequential_from_strategic_position(self, strategic_state, double time_limit=*)
    cpdef list cleanup_and_retry(self, current_state, double time_limit=*)
    cpdef set reconnect_clusters_strict(self, list clusters, list cluster_distances, current_state, set blocks_at_goal, list cleanup_path, int original_block_count)
    cpdef check_and_reconnect_blocks(self, set current_blocks, set blocks_at_goal, set connected_blocks, set remaining_cluster, list cleanup_path, int original_block_count)