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


from heapq import heappush, heappop
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move
import numpy as np

class PacmanAgent(BasePacmanAgent):
    """
    Optimized Pacman Agent using A* with turn-awareness and ghost prediction.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Max speed is still needed as a safety limit to prevent AgentLoadError
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
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
        Calculates the optimal number of steps to take without overshooting turns.
        """
        target = self._predict_ghost_move(enemy_position, my_position, map_state)
        path = self.astar(my_position, target, map_state)
        
        if path and path[0] != Move.STAY:
            first_move = path[0]
            
            # Count how many steps in the A* path go in the same direction
            straight_steps_in_path = 0
            for move in path:
                if move == first_move:
                    straight_steps_in_path += 1
                else:
                    break # Stop at the first turn
            
            # The steps we take should be the MINIMUM of:
            # 1. The speed limit set in terminal (to avoid crashing the loader)
            # 2. The number of steps until the next turn (to avoid overshooting)
            # 3. The physical distance to the next wall
            
            allowed_by_path = min(self.pacman_speed, straight_steps_in_path)
            actual_steps = self._max_valid_steps(my_position, first_move, map_state, allowed_by_path)
            
            return (first_move, actual_steps)
            
        return (Move.STAY, 1)

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

class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        pass
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Your current (row, col)
            enemy_position: Pacman's current (row, col)
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here
        from collections import deque

        def get_manhattan(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        dist_to_pacman = get_manhattan(my_position, enemy_position)

        if dist_to_pacman <= 5:
            def heuristic(g_pos, p_pos):
                score = get_manhattan(g_pos, p_pos) * 10
                
                moves_available = len(self.get_neighbors(g_pos, map_state))
                if moves_available <= 1:
                    score -= 10000 
                elif moves_available >= 3:
                    score += 50
                return score

            def minimax(g_pos, p_pos, depth, is_maximizing):
                if depth == 0:
                    return heuristic(g_pos, p_pos), None
                
                if is_maximizing:
                    best_score, best_move = -float('inf'), Move.STAY
                    for nxt_g, m in self.get_neighbors(g_pos, map_state):
                        score, _ = minimax(nxt_g, p_pos, depth - 1, False)
                        if score > best_score:
                            best_score, best_move = score, m
                    return best_score, best_move
                else:
                    best_score = float('inf')
                    for nxt_p, _ in self.get_neighbors(p_pos, map_state):
                        score, _ = minimax(g_pos, nxt_p, depth - 1, True)
                        if score < best_score:
                            best_score = score
                    return best_score, None

            _, chosen_move = minimax(my_position, enemy_position, 3, True)
            return chosen_move
        
        else:
            def get_distance_map(start_pos):
                dist_map = {start_pos: 0}
                queue_local = deque([start_pos])
                while queue_local:
                    curr = queue_local.popleft()
                    for nxt, _ in self.get_neighbors(curr, map_state):
                        if nxt not in dist_map:
                            dist_map[nxt] = dist_map[curr] + 1
                            queue_local.append(nxt)
                return dist_map

            pacman_dist = get_distance_map(enemy_position)
            ghost_dist = get_distance_map(my_position)

            target_cell = None
            max_p_steps = -1
            for cell, g_steps in ghost_dist.items():
                p_steps = pacman_dist.get(cell, 9999)
                if g_steps < p_steps:
                    if p_steps > max_p_steps:
                        max_p_steps = p_steps
                        target_cell = cell

            if target_cell and target_cell != my_position:
                path_queue = deque([(my_position, [])])
                visited = {my_position}
                while path_queue:
                    curr_p, path = path_queue.popleft()
                    if curr_p == target_cell:
                        return path[0]
                    for nxt, m in self.get_neighbors(curr_p, map_state):
                        if nxt not in visited:
                            visited.add(nxt)
                            path_queue.append((nxt, path + [m]))

        # Example: Simple evasive approach (replace with your algorithm)
        row_diff = my_position[0] - enemy_position[0]
        col_diff = my_position[1] - enemy_position[1]
        
        # Try to move away from Pacman
        if abs(row_diff) > abs(col_diff):
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT
        
        # Check if move is valid
        if self._is_valid_move(my_position, move, map_state):
            return move
        
        # If not valid, try other moves
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                return move
        
        return Move.STAY
    
    # Helper methods (you can add more)
    def get_neighbors(self, pos: tuple, map_state: np.ndarray):
        """Find all valid neighboring squares and the move to get there."""
        neighbors = []
        for move_enum in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move_enum.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_position(nxt, map_state):
                neighbors.append((nxt, move_enum))
        return neighbors
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

"""
class GhostAgent(BaseGhostAgent):
    """
    Simple Ghost Agent that moves randomly to valid neighboring positions.
    Ghost (Hider) Agent - Goal: Evade Pacman.
    Strategy: Move to a neighbor that maximizes distance from Pacman 
    and has the most escape routes (to avoid dead ends).
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Balanced Hider"
    
    def _is_valid_position(self, pos, map_state):
        """Check if position is inside map and not a wall."""
        row, col = pos
        h, w = map_state.shape
        return 0 <= row < h and 0 <= col < w and map_state[row, col] == 0

    def _get_neighbors(self, pos, map_state):
        """Get all valid neighboring positions."""
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = (pos[0] + move.value[0], pos[1] + move.value[1])
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two points."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Ghost decides its move by scoring neighbors.
        Higher score = Further from Pacman + More exit options.
        """
        neighbors = self._get_neighbors(my_position, map_state)
        
        # If trapped, stay put (though usually not possible in Pacman maps)
        if not neighbors:
            return Move.STAY
            
        best_move = Move.STAY
        max_score = -1
        
        for next_pos, move in neighbors:
            # 1. Base distance from Pacman (The further, the better)
            dist = self._manhattan_distance(next_pos, enemy_position)
            
            # 2. Safety factor: How many exits does this next position have?
            # A position with only 1 neighbor is a dead end.
            exits = len(self._get_neighbors(next_pos, map_state))
            
            # 3. Final score: Distance is primary, but exits help avoid traps
            # We multiply exits by a small factor to break ties or avoid close dead-ends
            score = dist + (exits * 0.5)
            
            if score > max_score:
                max_score = score
                best_move = move
                
        return best_move
"""
