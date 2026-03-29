"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- Pacman step must return either a Move or a (Move, steps) tuple where
    1 <= steps <= pacman_speed (provided via kwargs)
- Ghost step must return a Move enum value
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
"""

import sys
from pathlib import Path
from heapq import heappush, heappop
# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

import heapq
from collections import deque

from heapq import heappush, heappop
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move
import numpy as np

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """

    """
    Optimized Pacman Agent using A* with turn-awareness and ghost prediction.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        # - self.name = "Your Agent Name"
        self.name = "Smart AStar Seeker"

    
    def astar(self, start, goal, map_state):
        """Standard A* implementation to find the shortest path."""
        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        frontier = [(heuristic(start), start, [])]
        visited = {start}

        while frontier:
            f, current, path = heappop(frontier)
            if current == goal:
                return path

            for next_pos, move in self._get_neighbors(current, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    new_path = path + [move]
                    heappush(frontier, (len(new_path) + heuristic(next_pos), next_pos, new_path))
        return [Move.STAY]
    
    def _minimax(self, map_state, p_pos, g_pos, depth, alpha, beta, is_maximizing):
        """Minimax with Alpha-Beta Pruning to trap the ghost."""
        # Base case: catch the ghost or reach search limit
        if depth == 0 or p_pos == g_pos:
            # Score: Manhattan distance + penalty for ghost having escape routes
            dist = abs(p_pos[0] - g_pos[0]) + abs(p_pos[1] - g_pos[1])
            ghost_exits = len(self._get_neighbors(g_pos, map_state))
            # Pacman wants to MINIMIZE this score
            return dist + (ghost_exits * 10), None

        if not is_maximizing: # Pacman's Turn (Minimizer)
            best_score = float('inf')
            best_move = Move.STAY
            for next_p, move in self._get_neighbors(p_pos, map_state):
                score, _ = self._minimax(map_state, next_p, g_pos, depth - 1, alpha, beta, True)
                if score < best_score:
                    best_score, best_move = score, move
                beta = min(beta, score)
                if beta <= alpha: break # Pruning
            return best_score, best_move
        else: # Ghost's Turn (Maximizer)
            best_score = -float('inf')
            for next_g, _ in self._get_neighbors(g_pos, map_state):
                score, _ = self._minimax(map_state, p_pos, next_g, depth - 1, alpha, beta, False)
                if score > best_score:
                    best_score = score
                alpha = max(alpha, score)
                if beta <= alpha: break # Pruning
            return best_score, None

    def _is_dead_end(self, pos, map_state):
        """Checks if a position is a dead end (only one way out)."""
        return len(self._get_neighbors(pos, map_state)) <= 1

    def _predict_ghost_move(self, ghost_pos, my_pos, map_state):
        """
        Predicts ghost movement while considering dead ends.
        Ghost prefers cells that are far from Pacman and not dead ends.
        """
        best_pos = ghost_pos
        max_score = -float('inf')
        
        for neighbor_pos, _ in self._get_neighbors(ghost_pos, map_state):
            dist = abs(neighbor_pos[0] - my_pos[0]) + abs(neighbor_pos[1] - my_pos[1])
            # Penalty for moving into a dead end
            penalty = 5 if self._is_dead_end(neighbor_pos, map_state) else 0
            score = dist - penalty
            
            if score > max_score:
                max_score = score
                best_pos = neighbor_pos
        return best_pos

    def _get_neighbors(self, pos, map_state):
        """Gets all valid adjacent moves."""
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = (pos[0] + move.value[0], pos[1] + move.value[1])
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors

    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Your current (row, col)
            enemy_position: Ghost's current (row, col)
            step_number: Current step number (starts at 1)
            
        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
        # TODO: Implement your search algorithm here
        
        # Calculate distance to decide which strategy to use

        dist = abs(my_position[0] - enemy_position[0]) + abs(my_position[1] - enemy_position[1])

        # --- STRATEGY 1: TRAPPING (Minimax + Alpha-Beta) ---
        if dist <= 6:

            _, move = self._minimax(map_state, my_position, enemy_position, 4, -float('inf'), float('inf'), False)
            return (move, 1)
        

        # --- STRATEGY 2: CHASING (A* + Ghost Prediction) ---
        target = self._predict_ghost_move(enemy_position, my_position, map_state)
        path = self.astar(my_position, target, map_state)
        
        if path and path[0] != Move.STAY:
            first_move = path[0]
            
            # Count steps in the same direction for speed advantage
            straight_steps_in_path = 0
            for move in path:
                if move == first_move:
                    straight_steps_in_path += 1
                else:
                    break
            
            # Move as fast as possible in a straight line
            allowed_by_path = min(self.pacman_speed, straight_steps_in_path)
            actual_steps = self._max_valid_steps(my_position, first_move, map_state, allowed_by_path)
            
            return (first_move, actual_steps)
            
        return (Move.STAY, 1)

    # Helper methods (you can add more)
    def _max_valid_steps(self, pos, move, map_state, max_steps):
        """Checks physical wall constraints for a straight line move."""
        steps = 0
        current = pos
        for _ in range(max_steps):
            next_pos = (current[0] + move.value[0], current[1] + move.value[1])
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps
    
    def _is_valid_position(self, pos, map_state):
        """Checks if position is inside map and not a wall."""
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0
    

    # class PacmanAgent(BasePacmanAgent):
    # """
    # Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    # Implement your search algorithm to find and catch the ghost.
    # Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    # """
    
    # def __init__(self, **kwargs):
    # super().__init__(**kwargs)
    # self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
    # # TODO: Initialize any data structures you need
    # # Examples:
    # # - self.path = []  # Store planned path
    # # - self.visited = set()  # Track visited positions
    # # - self.name = "Your Agent Name"
    # self.name = "Template Pacman"
    
    # def step(self, map_state: np.ndarray, 
    #          my_position: tuple, 
    #          enemy_position: tuple,
    #          step_number: int):
    #     """
    #     Decide the next move.
        
    #     Args:
    #         map_state: 2D numpy array where 1=wall, 0=empty
    #         my_position: Your current (row, col)
    #         enemy_position: Ghost's current (row, col)
    #         step_number: Current step number (starts at 1)
            
    #     Returns:
    #         Move or (Move, steps): Direction to move (optionally with step count)
    #     """
    #     # TODO: Implement your search algorithm here
        
    #     # Example: Simple greedy approach (replace with your algorithm)
    #     row_diff = enemy_position[0] - my_position[0]
    #     col_diff = enemy_position[1] - my_position[1]
        
    #     # Try to move towards ghost
    #     if abs(row_diff) > abs(col_diff):
    #         primary_move = Move.DOWN if row_diff > 0 else Move.UP
    #         desired_steps = abs(row_diff)
    #     else:
    #         primary_move = Move.RIGHT if col_diff > 0 else Move.LEFT
    #         desired_steps = abs(col_diff)

    #     action = self._choose_action(
    #         my_position,
    #         [primary_move],
    #         map_state,
    #         desired_steps
    #     )
    #     if action:
    #         return action

    #     # If the primary direction is blocked, try other moves
    #     fallback_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
    #     action = self._choose_action(my_position, fallback_moves, map_state, self.pacman_speed)
    #     if action:
    #         return action
        
    #     return (Move.STAY, 1)
    
    # # Helper methods (you can add more)
    
    # def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
    #     for move in moves:
    #         max_steps = min(self.pacman_speed, max(1, desired_steps))
    #         steps = self._max_valid_steps(pos, move, map_state, max_steps)
    #         if steps > 0:
    #             return (move, steps)
    #     return None

    # def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
    #     steps = 0
    #     current = pos
    #     for _ in range(max_steps):
    #         delta_row, delta_col = move.value
    #         next_pos = (current[0] + delta_row, current[1] + delta_col)
    #         if not self._is_valid_position(next_pos, map_state):
    #             break
    #         steps += 1
    #         current = next_pos
    #     return steps
    
    # def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
    #     """Check if a move from pos is valid for at least one step."""
    #     return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    # def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
    #     """Check if a position is valid (not a wall and within bounds)."""
    #     row, col = pos
    #     height, width = map_state.shape
        
    #     if row < 0 or row >= height or col < 0 or col >= width:
    #         return False
        
    #     return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):
    """
    Advanced Ghost Agent: 'Area-Aware Evader'
    Strategy: Uses Speed-aware Voronoi Partitioning and Flood Fill 
    to maximize survival space and avoid dead-ends.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Area-Aware Evader"
        # Extract Pacman's speed from environment context
        self.pacman_max_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.inf_dist = 999

    def _get_area_score(self, pos, map_state, max_depth=5):
        """
        Performs a limited-depth Flood Fill to evaluate the 'living space' 
        available from a given position. Helps detect and avoid dead-ends.
        """
        visited = {pos}
        queue = deque([(pos, 0)])
        area_size = 0
        
        while queue:
            curr, d = queue.popleft()
            if d >= max_depth:
                continue
            
            for nxt, _ in self._get_neighbor_positions(curr, map_state):
                if nxt not in visited:
                    visited.add(nxt)
                    area_size += 1
                    queue.append((nxt, d + 1))
        return area_size

    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Core decision-making loop. Finds the optimal target cell 
        within the calculated Safe Zone.
        """
        # 1. Map Pacman's potential reach using Speed-aware BFS
        pacman_steps = self._bfs_speed_dist(map_state, enemy_position, self.pacman_max_speed)
        
        # 2. Map Ghost's reach using standard BFS
        ghost_dirs = [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]
        ghost_dist = self._bfs_dist(map_state, my_position, ghost_dirs)

        best_target = None
        best_score = -1

        # 3. Voronoi Selection: Filter for cells Ghost can reach before Pacman
        for pos, gd in ghost_dist.items():
            if gd == 0: continue
            
            ps = pacman_steps.get(pos, self.inf_dist)
            
            # Safe Zone logic: ghost_steps <= pacman_steps
            if gd <= ps:
                # Heuristic: Balance between distance from seeker and future mobility
                area_val = self._get_area_score(pos, map_state, max_depth=5)
                score = ps * 200 + area_val * 50
                
                if score > best_score:
                    best_score = score
                    best_target = pos

        # 4. Navigate to the best safe target using A*
        if best_target is not None:
            path = self._astar_one_step(map_state, my_position, best_target)
            if path and len(path) >= 2:
                return self._get_move_from_positions(my_position, path[1])

        # 5. Fallback: If cornered, execute greedy evasion
        return self._greedy_evade(map_state, my_position, pacman_steps)

    # --- Search Algorithms ---

    def _bfs_dist(self, map_state, start, dirs):
        """Standard BFS to calculate step distance to all reachable cells."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            p = queue.popleft()
            d = dist[p]
            for mv in dirs:
                nxt = (p[0] + mv.value[0], p[1] + mv.value[1])
                if nxt not in dist and self._is_valid(nxt, map_state):
                    dist[nxt] = d + 1
                    queue.append(nxt)
        return dist

    def _bfs_speed_dist(self, map_state, start, max_speed):
        """
        Speed-aware BFS: Accounts for Pacman's ability to dash up to 
        'max_speed' cells in a straight line per game turn.
        """
        dist = {start: 0}
        queue = deque([start])
        while queue:
            p = queue.popleft()
            d = dist[p]
            for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                for n in range(1, max_speed + 1):
                    nxt = (p[0] + mv.value[0] * n, p[1] + mv.value[1] * n)
                    if not self._is_valid(nxt, map_state): break
                    if nxt not in dist:
                        dist[nxt] = d + 1
                        queue.append(nxt)
        return dist

    def _astar_one_step(self, map_state, start, goal):
        """Standard A* search for single-step navigation."""
        if start == goal: return [start]
        def h(p): return abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        heap = [(h(start), 0, start)]
        came_from = {}; g_score = {start: 0}
        while heap:
            _, cost, p = heapq.heappop(heap)
            if p == goal:
                path = [p]
                while p in came_from: p = came_from[p]; path.append(p)
                return path[::-1]
            for nxt, move in self._get_neighbor_positions(p, map_state):
                new_g = cost + 1
                if new_g < g_score.get(nxt, self.inf_dist):
                    g_score[nxt] = new_g; came_from[nxt] = p
                    heapq.heappush(heap, (new_g + h(nxt), new_g, nxt))
        return None

    def _greedy_evade(self, map_state, my_pos, pacman_dist_map):
        """
        Heuristic-based greedy evasion used when no safe zone is available.
        """
        best_mv = Move.STAY
        best_score = pacman_dist_map.get(my_pos, 0) * 100 + self._get_area_score(my_pos, map_state) * 10
        for nxt, mv in self._get_neighbor_positions(my_pos, map_state):
            score = pacman_dist_map.get(nxt, 0) * 100 + self._get_area_score(nxt, map_state) * 10
            if score > best_score:
                best_score = score; best_mv = mv
        return best_mv

    # --- Utility Methods ---

    def _get_neighbor_positions(self, pos, map_state):
        """Returns valid adjacent positions and their corresponding Move enum."""
        neighbors = []
        for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = (pos[0] + mv.value[0], pos[1] + mv.value[1])
            if self._is_valid(nxt, map_state): neighbors.append((nxt, mv))
        return neighbors

    def _is_valid(self, pos, map_state):
        """Checks map boundaries and wall collisions."""
        r, c = pos; h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

    def _get_move_from_positions(self, from_pos, to_pos):
        """Translates a coordinate delta into a Move enum."""
        dr, dc = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if mv.value == (dr, dc): return mv
        return Move.STAY

# class GhostAgent(BaseGhostAgent):
#     """
#     Simple Ghost Agent that moves randomly to valid neighboring positions.
#     Ghost (Hider) Agent - Goal: Evade Pacman.
#     Strategy: Move to a neighbor that maximizes distance from Pacman 
#     and has the most escape routes (to avoid dead ends).
#     """
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.name = "Balanced Hider"
    
#     def _is_valid_position(self, pos, map_state):
#         """Check if position is inside map and not a wall."""
#         row, col = pos
#         h, w = map_state.shape
#         return 0 <= row < h and 0 <= col < w and map_state[row, col] == 0

#     def _get_neighbors(self, pos, map_state):
#         """Get all valid neighboring positions."""
#         neighbors = []
#         for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
#             next_pos = (pos[0] + move.value[0], pos[1] + move.value[1])
#             if self._is_valid_position(next_pos, map_state):
#                 neighbors.append((next_pos, move))
#         return neighbors

#     def _manhattan_distance(self, pos1, pos2):
#         """Calculate Manhattan distance between two points."""
#         return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

#     def step(self, map_state, my_position, enemy_position, step_number):
#         """
#         Ghost decides its move by scoring neighbors.
#         Higher score = Further from Pacman + More exit options.
#         """
#         neighbors = self._get_neighbors(my_position, map_state)
        
#         # If trapped, stay put (though usually not possible in Pacman maps)
#         if not neighbors:
#             return Move.STAY
            
#         best_move = Move.STAY
#         max_score = -1
        
#         for next_pos, move in neighbors:
#             # 1. Base distance from Pacman (The further, the better)
#             dist = self._manhattan_distance(next_pos, enemy_position)
            
#             # 2. Safety factor: How many exits does this next position have?
#             # A position with only 1 neighbor is a dead end.
#             exits = len(self._get_neighbors(next_pos, map_state))
            
#             # 3. Final score: Distance is primary, but exits help avoid traps
#             # We multiply exits by a small factor to break ties or avoid close dead-ends
#             score = dist + (exits * 0.5)
            
#             if score > max_score:
#                 max_score = score
#                 best_move = move
                
#         return best_move