"""
CSP Solver for Zebra Logic Puzzles

This solver implements:
- Backtracking search with MRV (Minimum Remaining Values) heuristic
- Forward checking for early pruning
- Arc consistency (AC-3) for constraint propagation

The CSP is modeled as:
- Variables: Each (dimension, value) pair (e.g., ("name", "Alice"))
- Domain: House positions (1 to n)
- Constraints: Parsed from puzzle clues
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from copy import deepcopy
from dataclasses import dataclass, field
from dataParser import Puzzle
import re


@dataclass
class SolverStats:
    """Track solver statistics for evaluation."""
    backtracks: int = 0
    nodes_explored: int = 0
    arc_revisions: int = 0


@dataclass
class CSPSolver:
    """
    Constraint Satisfaction Problem Solver for Zebra puzzles.
    
    Attributes:
        num_houses: Number of houses/positions in the puzzle
        variables: Dict mapping dimension -> list of values
        constraints: List of constraint expressions
        domains: Dict mapping (dim, value) -> set of possible positions
        stats: Solver statistics for evaluation
    """
    num_houses: int
    variables: Dict[str, List[str]]
    constraints: List[Dict]
    domains: Dict[Tuple[str, str], Set[int]] = field(default_factory=dict)
    stats: SolverStats = field(default_factory=SolverStats)
    
    def __post_init__(self):
        """Initialize domains for all variables."""
        self._initialize_domains()
    
    def _initialize_domains(self):
        """
        Set initial domain for each variable.
        Each (dimension, value) can be in any house 1..num_houses.
        """
        for dim, values in self.variables.items():
            for val in values:
                # Each value can initially be in any house
                self.domains[(dim, val)] = set(range(1, self.num_houses + 1))
    
    # ==================== CONSTRAINT CHECKING ====================
    
    def check_constraint(self, constraint: Dict, assignment: Dict[Tuple[str, str], int]) -> bool:
        """
        Check if a constraint is satisfied given current assignment.
        Returns True if satisfied or not yet applicable (variables unassigned).
        
        Args:
            constraint: The constraint expression dict
            assignment: Current variable assignments {(dim, val): position}
        """
        ctype = constraint[0]
        items = [constraint[1], constraint[2]]
        
        # Get assigned positions for items in constraint
        positions = []
        for item in items:
            key = (item["dim"], item["value"])
            if key in assignment:
                positions.append(assignment[key])
            else:
                positions.append(None)  # Not yet assigned
        
        # Handle each constraint type

        # Special Cases if the second attribute is a number

        if type(items[1]) == int:
            if ctype == "=":
                # X must be in house N
                if positions[0] is None:
                    return True
                return positions[0] == items[1]
            
            elif ctype == "!=":
                # X must NOT be in house N
                if positions[0] is None:
                    return True
                return positions[0] != items[1]

        # Regular Cases

        if ctype == "=":
            # X and Y must be in the same house
            if positions[0] is None or positions[1] is None:
                return True  # Can't check yet
            return positions[0] == positions[1]
        
        elif ctype == "<":
            # X is somewhere to the left of Y (X.pos < Y.pos)
            if positions[0] is None or positions[1] is None:
                return True
            return positions[0] < positions[1]
        
        elif ctype == ">":
            # X is somewhere to the right of Y (X.pos > Y.pos)
            if positions[0] is None or positions[1] is None:
                return True
            return positions[0] > positions[1]
        
        # Distance Check

        distanceCheck = re.search(r"+-(\d+)", ctype)

        if distanceCheck:
            # Distance between X and Y equals d
            dist = distanceCheck.group(1)
            if positions[0] is None or positions[1] is None:
                return True
            return abs(positions[0] - positions[1]) == dist

        # Offset Check
        
        offsetCheckL = re.search(r"+(\d+)", ctype)

        if offsetCheckL:
            # X is directly left of Y (X.pos + 1 == Y.pos)
            dist = offsetCheckL.group(1)
            if positions[0] is None or positions[1] is None:
                return True
            return positions[1] - positions[0] == dist
        
        offsetCheckR = re.search(r"-(\d+)", ctype)

        if offsetCheckR:
            # X is directly left of Y (X.pos + 1 == Y.pos)
            dist = offsetCheckR.group(1)
            if positions[0] is None or positions[1] is None:
                return True
            return positions[1] - positions[0] == dist
        
        return True  # Unknown constraint type, assume satisfied
    
    def is_consistent(self, var: Tuple[str, str], value: int, 
                      assignment: Dict[Tuple[str, str], int]) -> bool:
        """
        Check if assigning value to var is consistent with all constraints.
        
        Args:
            var: Variable tuple (dimension, value)
            value: Position to assign
            assignment: Current assignments
        """
        # Create temporary assignment
        temp_assignment = assignment.copy()
        temp_assignment[var] = value
        
        # Check all constraints
        for constraint in self.constraints:
            if not self.check_constraint(constraint, temp_assignment):
                return False
        
        # Check uniqueness: no two values in same dimension can have same position
        dim, val = var
        for other_val in self.variables[dim]:
            if other_val != val:
                other_var = (dim, other_val)
                if other_var in assignment and assignment[other_var] == value:
                    return False
        
        return True
    
    # ==================== ARC CONSISTENCY (AC-3) ====================
    
    def ac3(self, domains: Dict[Tuple[str, str], Set[int]]) -> bool:
        """
        Apply AC-3 arc consistency algorithm.
        Reduces domains by removing inconsistent values.
        
        Returns False if any domain becomes empty (no solution possible).
        """
        # Build queue of arcs to check
        queue = []
        
        # For each constraint, add relevant arcs
        for constraint in self.constraints:
            expr = constraint[0]
            attr1 = constraint[1]
            attr2 = constraint[2]
            
            if attr1 and attr2:
                var1 = (attr1.domain, attr1.value)
                var2 = (attr2.domain, attr2.value)
                queue.append((var1, var2, constraint))
                queue.append((var2, var1, constraint))
    
        # Add uniqueness arcs (same dimension, different values)
        for dim, values in self.variables.items():
            for i, val1 in enumerate(values):
                for val2 in values[i+1:]:
                    var1 = (dim, val1)
                    var2 = (dim, val2)
                    # These must have different positions
                    queue.append((var1, var2, "!="))
                    queue.append((var2, var1, "!="))
        
        while queue:
            var1, var2, constraint = queue.pop(0)
            self.stats.arc_revisions += 1
            
            if self._revise(domains, var1, var2, constraint):
                if len(domains[var1]) == 0:
                    return False  # Domain wiped out
                
                # Add neighbors back to queue
                for c in self.constraints:
                    expr = c.get("expression", c)
                    items = expr.get("items", [])
                    if len(items) == 2:
                        cv1 = (items[0]["dim"], items[0]["value"])
                        cv2 = (items[1]["dim"], items[1]["value"])
                        if cv2 == var1 and cv1 != var2:
                            queue.append((cv1, var1, c))
                        elif cv1 == var1 and cv2 != var2:
                            queue.append((cv2, var1, c))
        
        return True
    
    def _revise(self, domains: Dict[Tuple[str, str], Set[int]], 
                var1: Tuple[str, str], var2: Tuple[str, str], 
                constraint: str) -> bool:
        """
        Remove values from var1's domain that have no support in var2's domain.
        Returns True if domain was reduced.
        """
        revised = False
        to_remove = []
        
        for val1 in domains[var1]:
            # Check if any value in var2's domain is consistent
            has_support = False
            for val2 in domains[var2]:
                if self._arc_consistent(var1, val1, var2, val2, constraint):
                    has_support = True
                    break
            
            if not has_support:
                to_remove.append(val1)
                revised = True
        
        for val in to_remove:
            domains[var1].remove(val)
        
        return revised
    
    def _arc_consistent(self, var1: Tuple[str, str], pos1: int,
                        var2: Tuple[str, str], pos2: int, 
                        constraint: str) -> bool:
        """Check if (var1=pos1, var2=pos2) satisfies the constraint."""
        
        # Handle uniqueness constraint
        if constraint == "!=":
            return pos1 != pos2
        
        # Handle regular constraints
        items = expr.get("items", [])
        if len(items) < 2:
            return True
        
        item1_key = (items[0]["dim"], items[0]["value"])
        item2_key = (items[1]["dim"], items[1]["value"])
        
        # Determine which position corresponds to which item
        if var1 == item1_key and var2 == item2_key:
            p1, p2 = pos1, pos2
        elif var1 == item2_key and var2 == item1_key:
            p1, p2 = pos2, pos1
        else:
            return True  # Constraint doesn't involve these variables
        
        if ctype == "same_house":
            return p1 == p2
        elif ctype == "left_of":
            return p1 < p2
        elif ctype == "right_of":
            return p1 > p2
        elif ctype == "next_to":
            return abs(p1 - p2) == 1
        elif ctype == "distance":
            return abs(p1 - p2) == expr.get("distance")
        elif ctype == "offset":
            return p2 - p1 == expr.get("distance", 1)
        
        return True
    
    # ==================== FORWARD CHECKING ====================
    
    def forward_check(self, var: Tuple[str, str], value: int,
                      domains: Dict[Tuple[str, str], Set[int]]) -> bool:
        """
        After assigning var=value, prune domains of related variables.
        Returns False if any domain becomes empty.
        """
        dim, val = var
        
        # Remove this position from other values in same dimension
        for other_val in self.variables[dim]:
            if other_val != val:
                other_var = (dim, other_val)
                if value in domains[other_var]:
                    domains[other_var] = domains[other_var] - {value}
                    if len(domains[other_var]) == 0:
                        return False
        
        # Prune based on constraints
        for constraint in self.constraints:
            expr = constraint.get("expression", constraint)
            items = expr.get("items", [])
            
            if len(items) < 1:
                continue
            
            item1_key = (items[0]["dim"], items[0]["value"])
            
            if len(items) == 1 and item1_key == var:
                # Unary constraint, already handled in consistency check
                continue
            
            if len(items) == 2:
                item2_key = (items[1]["dim"], items[1]["value"])
                
                # Determine which is assigned
                if item1_key == var:
                    other_var = item2_key
                    assigned_is_first = True
                elif item2_key == var:
                    other_var = item1_key
                    assigned_is_first = False
                else:
                    continue
                
                # Prune other_var's domain
                ctype = expr["type"]
                to_remove = []
                
                for other_pos in domains[other_var]:
                    if assigned_is_first:
                        p1, p2 = value, other_pos
                    else:
                        p1, p2 = other_pos, value
                    
                    valid = True
                    if ctype == "same_house":
                        valid = (p1 == p2)
                    elif ctype == "left_of":
                        valid = (p1 < p2)
                    elif ctype == "right_of":
                        valid = (p1 > p2)
                    elif ctype == "next_to":
                        valid = (abs(p1 - p2) == 1)
                    elif ctype == "distance":
                        valid = (abs(p1 - p2) == expr.get("distance"))
                    elif ctype == "offset":
                        valid = (p2 - p1 == expr.get("distance", 1))
                    
                    if not valid:
                        to_remove.append(other_pos)
                
                domains[other_var] = domains[other_var] - set(to_remove)
                if len(domains[other_var]) == 0:
                    return False
        
        return True
    
    # ==================== MRV HEURISTIC ====================
    
    def select_unassigned_variable(self, assignment: Dict[Tuple[str, str], int],
                                    domains: Dict[Tuple[str, str], Set[int]]) -> Optional[Tuple[str, str]]:
        """
        Select next variable using MRV (Minimum Remaining Values) heuristic.
        Choose variable with smallest domain to fail fast.
        """
        unassigned = [v for v in domains.keys() if v not in assignment]
        
        if not unassigned:
            return None
        
        # MRV: select variable with minimum domain size
        return min(unassigned, key=lambda v: len(domains[v]))
    
    def order_domain_values(self, var: Tuple[str, str], 
                            domains: Dict[Tuple[str, str], Set[int]]) -> List[int]:
        """
        Order domain values for assignment.
        Currently returns values in sorted order (can be enhanced with LCV).
        """
        return sorted(domains[var])
    
    # ==================== BACKTRACKING SEARCH ====================
    
    def backtrack(self, assignment: Dict[Tuple[str, str], int],
                  domains: Dict[Tuple[str, str], Set[int]]) -> Optional[Dict[Tuple[str, str], int]]:
        """
        Recursive backtracking search with forward checking.
        
        Args:
            assignment: Current variable assignments
            domains: Current domains (may be pruned)
        
        Returns:
            Complete assignment if solution found, None otherwise
        """
        self.stats.nodes_explored += 1
        
        # Check if assignment is complete
        if len(assignment) == len(domains):
            return assignment
        
        # Select next variable (MRV)
        var = self.select_unassigned_variable(assignment, domains)
        if var is None:
            return assignment
        
        # Try each value in domain
        for value in self.order_domain_values(var, domains):
            # Check consistency
            if self.is_consistent(var, value, assignment):
                # Make assignment
                new_assignment = assignment.copy()
                new_assignment[var] = value
                
                # Copy domains and apply forward checking
                new_domains = {k: v.copy() for k, v in domains.items()}
                new_domains[var] = {value}
                
                # Forward check
                if self.forward_check(var, value, new_domains):
                    # Recurse
                    result = self.backtrack(new_assignment, new_domains)
                    if result is not None:
                        return result
                
                # Backtrack
                self.stats.backtracks += 1
        
        return None
    
    # ==================== MAIN SOLVE METHOD ====================
    
    def solve(self) -> Optional[Dict[int, Dict[str, str]]]:
        """
        Solve the CSP and return solution in readable format.
        
        Returns:
            Dict mapping house number -> {dimension: value}
            e.g., {1: {"name": "Alice", "color": "red"}, ...}
            Returns None if no solution exists.
        """
        # Reset stats
        self.stats = SolverStats()
        
        # Initialize domains
        self._initialize_domains()
        
        # Apply initial constraint propagation
        # First, apply unary constraints (position_equals, not_position_equals)
        for constraint in self.constraints:
            
            ctype = constraint[0]
            attr1 = constraint[1]
            attr2 = constraint[2]
            
            if ctype == "=" and type(attr2) == int:
                var = (attr1.domain, attr1.value)
                self.domains[var] = {attr2}
            
            elif ctype == "!=" and type(attr2) == int:
                var = (attr1.domain, attr1.value)
                self.domains[var].discard(attr2)
        
        # Apply AC-3
        domains_copy = {k: v.copy() for k, v in self.domains.items()}
        if not self.ac3(domains_copy):
            return None  # No solution
        
        # Run backtracking search
        assignment = self.backtrack({}, domains_copy)
        
        if assignment is None:
            return None
        
        # Convert to readable format: house -> {dim: value}
        solution = {i: {} for i in range(1, self.num_houses + 1)}
        for (dim, val), pos in assignment.items():
            solution[pos][dim] = val
        
        return solution


def create_solver_from_csp(csp: Puzzle) -> CSPSolver:
    """
    Factory function to create a solver from parsed CSP dict.
    
    Args:
        csp: Dict with 'houses', 'variables', 'constraints' keys
    
    Returns:
        Configured CSPSolver instance
    """
    return CSPSolver(
        num_houses=csp.entities,
        variables=csp.variables,
        constraints=csp.constraints
    )


def solve_puzzle(csp: Puzzle) -> Tuple[Optional[Dict], SolverStats]:
    """
    Convenience function to solve a puzzle and return solution + stats.
    
    Args:
        csp: Parsed CSP dictionary
    
    Returns:
        Tuple of (solution dict or None, solver statistics)
    """
    solver = create_solver_from_csp(csp)
    solution = solver.solve()
    return solution, solver.stats
