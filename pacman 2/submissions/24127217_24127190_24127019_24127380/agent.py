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
- Agents are STATEFUL - you can store memory across steps
- enemy_position may be None when limited observation is enabled
- map_state cells: 1=wall, 0=empty, -1=unseen (fog)
"""

import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        self.name = "Template Pacman"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Ghost's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
        # TODO: Implement your search algorithm here
        
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or explore
        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            # No information about enemy - explore randomly
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return (move, 1)
            return (Move.STAY, 1)
        
        # Example: Simple greedy approach (replace with your algorithm)
        row_diff = target[0] - my_position[0]
        col_diff = target[1] - my_position[1]
        
        # Try to move towards ghost
        if abs(row_diff) > abs(col_diff):
            primary_move = Move.DOWN if row_diff > 0 else Move.UP
            desired_steps = abs(row_diff)
        else:
            primary_move = Move.RIGHT if col_diff > 0 else Move.LEFT
            desired_steps = abs(col_diff)

        action = self._choose_action(
            my_position,
            [primary_move],
            map_state,
            desired_steps
        )
        if action:
            return action

        # If the primary direction is blocked, try other moves
        fallback_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        action = self._choose_action(my_position, fallback_moves, map_state, self.pacman_speed)
        if action:
            return action
        
        return (Move.STAY, 1)
    
    # Helper methods (you can add more)
    
    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        # Belief map to track danger levels across the 21x21 grid
        self.belief_map = np.zeros((21, 21))
        # Standard directions for vision and neighbor checks
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Path History (To avoid pacing back and forth)
        self.history = []
        self.max_history = 10 
        # When we lose line-of-sight, we use this to change behavior
        self.stealth_timer = 0
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Pacman's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here
        
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or move randomly
        threat = enemy_position or self.last_known_enemy_pos
        
        if threat is None:
            # No information about enemy - move randomly
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return move
            return Move.STAY
        
        # 1. Update memory and the Belief Map (Danger in the fog)
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            # Reset belief: if we see him, we know exactly where danger is
            self.belief_map.fill(0)
            self.belief_map[enemy_position] = 100
        else:
            # Increase 'danger' score in unseen areas (-1) 
            # This implements the "Belief-Map Evacuation" strategy
            self.belief_map[map_state == -1] += 0.5
            if self.last_known_enemy_pos:
                # Slowly fade the old sighting
                self.belief_map[self.last_known_enemy_pos] *= 0.9 

        # 2. Start the Risk-Aversion Algorithm
        best_move = Move.STAY
        min_risk = float('inf')
        
        # We evaluate all 5 possible actions
        possible_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        
        for move in possible_moves:
            # Skip moves that hit walls
            if not self._is_valid_move(my_position, move, map_state):
                continue
            
            # Calculate coordinates for the potential move
            delta_r, delta_c = move.value
            new_pos = (my_position[0] + delta_r, my_position[1] + delta_c)
            
            # Evaluate the 'Risk Score' for this new position
            risk = self._calculate_risk(new_pos, map_state, enemy_position)

            # Penalize cells we recently stood in to prevent "vibrating" or staying still too long
            if new_pos in self.history:
                risk += 25 

            # Keep track of the safest option
            if risk < min_risk:
                min_risk = risk
                best_move = move

            # At the very end of step(), update the history
            self.history.append(my_position)
            if len(self.history) > self.max_history:
                self.history.pop(0)

        # 3. Final decision: Return the move with the absolute lowest risk
        return best_move
    
    # Helper methods (you can add more)
    
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
    
    def _calculate_risk(self, pos: tuple, map_state: np.ndarray, enemy_pos: tuple) -> float:
        risk = 0.0
        r, c = pos
        threat = enemy_pos or self.last_known_enemy_pos

        if threat:
            tr, tc = threat
            
            # 1. THE SHADOW CHECK
            # If we are in the same row or same column, we might be visible
            if r == tr or c == tc:
                # Check if there is a wall between us and the threat
                if not self._is_wall_between(pos, threat, map_state):
                    risk += 100  # EXTREME danger: we are in his line of sight!
            else:
                # We are diagonal to him! This is a "Shadow."
                # We reward this by not adding risk, or even subtracting a little.
                risk -= 5 

            # 2. PROXIMITY (Standard distance penalty)
            dist = abs(r - tr) + abs(c - tc)
            if dist < 4:
                risk += 40
                
        # C. MOBILITY BONUS: Prefer cells with many exits
        open_neighbors = self._count_neighbors(pos, map_state)
        if open_neighbors >= 3:
            risk -= 20  # Reward intersections
        elif open_neighbors == 1:
            risk += 80  # Heavily penalize dead ends

        # D. EXPLORATION REWARD: Prefer moving into the unknown
        if map_state[pos] == -1:
            risk -= 15

        # 3. DEAD END CHECK (Connectivity)
        if self._count_neighbors(pos, map_state) <= 1:
            risk += 80

        # 4. FOG DANGER (Belief Map)
        risk += self.belief_map[pos]

        return risk
    
    def _is_wall_between(self, pos1: tuple, pos2: tuple, map_state: np.ndarray) -> bool:
        r1, c1 = pos1
        r2, c2 = pos2
        
        # If in the same row, check all columns between them
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if map_state[r1, c] == 1:
                    return True
        # If in the same column, check all rows between them
        elif c1 == c2:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if map_state[r, c1] == 1:
                    return True
                    
        return False

    def _count_neighbors(self, pos, map_state):
        count = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = pos[0]+dr, pos[1]+dc
            if 0 <= nr < 21 and 0 <= nc < 21 and map_state[nr, nc] != 1:
                count += 1
        return count
