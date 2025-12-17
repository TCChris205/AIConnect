from typing import Dict, List, Set, Literal, Tuple, Optional, Any
from copy import deepcopy
from dataclasses import dataclass, field
import re
import pandas as pd
import json
import time

# =================================== Solver ===================================

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

@dataclass
class SolverStats:
    """Track solver statistics for evaluation."""
    backtracks: int = 0
    nodes_explored: int = 0
    arc_revisions: int = 0
    steps: int = 0      #changed from steps = int = 0 (syntax error)

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
        expr = constraint.get("expression", constraint)
        ctype = expr["type"]
        items = expr.get("items", [])
        
        # Get assigned positions for items in constraint
        positions = []
        for item in items:
            key = (item["dim"], item["value"])
            if key in assignment:
                positions.append(assignment[key])
            else:
                positions.append(None)  # Not yet assigned
        
        # Handle each constraint type
        if ctype == "same_house":
            # X and Y must be in the same house
            if positions[0] is None or positions[1] is None:
                return True  # Can't check yet
            return positions[0] == positions[1]
        
        elif ctype == "position_equals":
            # X must be in house N
            target_pos = expr.get("position")
            if positions[0] is None:
                return True
            return positions[0] == target_pos
        
        elif ctype == "not_position_equals":
            # X must NOT be in house N
            target_pos = expr.get("position")
            if positions[0] is None:
                return True
            return positions[0] != target_pos
        
        elif ctype == "left_of":
            # X is somewhere to the left of Y (X.pos < Y.pos)
            if positions[0] is None or positions[1] is None:
                return True
            return positions[0] < positions[1]
        
        elif ctype == "right_of":
            # X is somewhere to the right of Y (X.pos > Y.pos)
            if positions[0] is None or positions[1] is None:
                return True
            return positions[0] > positions[1]
        
        elif ctype == "next_to":
            # X and Y are neighbors (|X.pos - Y.pos| == 1)
            if positions[0] is None or positions[1] is None:
                return True
            return abs(positions[0] - positions[1]) == 1
        
        elif ctype == "distance":
            # Distance between X and Y equals d
            dist = expr.get("distance")
            if positions[0] is None or positions[1] is None:
                return True
            return abs(positions[0] - positions[1]) == dist
        
        elif ctype == "offset":
            # X is directly left of Y (X.pos + 1 == Y.pos)
            dist = expr.get("distance", 1)
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
            expr = constraint.get("expression", constraint)
            items = expr.get("items", [])
            
            if len(items) == 2:
                var1 = (items[0]["dim"], items[0]["value"])
                var2 = (items[1]["dim"], items[1]["value"])
                queue.append((var1, var2, constraint))
                queue.append((var2, var1, constraint))
        
        # Add uniqueness arcs (same dimension, different values)
        for dim, values in self.variables.items():
            for i, val1 in enumerate(values):
                for val2 in values[i+1:]:
                    var1 = (dim, val1)
                    var2 = (dim, val2)
                    # These must have different positions
                    queue.append((var1, var2, {"type": "different"}))
                    queue.append((var2, var1, {"type": "different"}))
        
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
                constraint: Dict) -> bool:
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
                        constraint: Dict) -> bool:
        """Check if (var1=pos1, var2=pos2) satisfies the constraint."""
        expr = constraint.get("expression", constraint)
        ctype = expr.get("type", constraint.get("type"))
        
        # Handle uniqueness constraint
        if ctype == "different":
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
        self.stats.steps += 1

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
            expr = constraint.get("expression", constraint)
            ctype = expr["type"]
            items = expr.get("items", [])
            
            if ctype == "position_equals" and len(items) == 1:
                var = (items[0]["dim"], items[0]["value"])
                pos = expr.get("position")
                self.domains[var] = {pos}
            
            elif ctype == "not_position_equals" and len(items) == 1:
                var = (items[0]["dim"], items[0]["value"])
                pos = expr.get("position")
                self.domains[var].discard(pos)
        
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

def create_solver_from_csp(csp: Dict) -> CSPSolver:
    """
    Factory function to create a solver from parsed CSP dict.
    
    Args:
        csp: Dict with 'houses', 'variables', 'constraints' keys
    
    Returns:
        Configured CSPSolver instance
    """
    return CSPSolver(
        num_houses=csp["houses"],
        variables=csp["variables"],
        constraints=csp["constraints"]
    )

def solve_puzzle(csp: Dict) -> Tuple[Optional[Dict], SolverStats]:
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

# =================================== Parser ===================================

# Global ordinal word mapping (dataset uses at most 6 houses in many cases)
ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
}

# Word numbers for flexible "three houses" / "There are five houses" patterns
NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

def _word_or_digit_to_int(s: str) -> Optional[int]:
    s = s.lower().strip()
    if s.isdigit():
        return int(s)
    return NUM_WORDS.get(s)

def normalize_text(s: str) -> str:
    """
    Normalize text for matching:
    - lowercase
    - remove apostrophes
    - replace hyphens with spaces
    - remove the standalone word 'person'
    - collapse multiple spaces
    """
    s = s.lower()
    s = re.sub(r"[’']", "", s)          # remove apostrophes
    s = s.replace("-", " ")             # hyphen -> space
    s = re.sub(r"\bperson\b", "", s)    # remove 'person'
    s = re.sub(r"\s+", " ", s)          # collapse spaces
    return s.strip()


@dataclass
class Clue:
    number: int
    text: str


@dataclass
class PuzzleSkeleton:
    houses: int
    dimensions: Dict[str, List[str]]  # e.g. {"name": [...], "car": [...], ...}


@dataclass
class ItemRef:
    dim: str
    value: str


@dataclass
class Constraint:
    """
    Internal representation of a constraint.
    Will be converted to a dictionary under "expression".
    """
    type: Literal[
        "same_house",          # X and Y are in the same house
        "position_equals",     # X is in house N
        "not_position_equals", # X is NOT in house N
        "left_of",             # X is somewhere to the left of Y
        "right_of",            # X is somewhere to the right of Y
        "next_to",             # X and Y are neighbors
        "distance",            # distance between positions = d
        "offset",              # X is directly left/right of Y (distance=1)
    ]
    items: List[ItemRef] = field(default_factory=list)
    position: Optional[int] = None
    distance: Optional[int] = None


def split_description_and_clues(text: str) -> tuple[str, List[str]]:
    lines = text.splitlines()
    desc_lines: List[str] = []
    clue_lines: List[str] = []
    in_clues = False

    for line in lines:
        stripped = line.strip()
        if not in_clues:
            # accept both headers
            if stripped.lower().startswith("## clues") or stripped.lower() == "clues:" or stripped.lower().startswith("clues:"):
                in_clues = True
                # If the line is exactly "Clues:" we skip it.
                continue
            desc_lines.append(line)
        else:
            if stripped:
                clue_lines.append(line.rstrip("\n"))

    return "\n".join(desc_lines), clue_lines

def parse_description(desc_text: str) -> PuzzleSkeleton:
    """
    Parse description to:
      - number of houses
      - dimensions/domains from bullet lines with backticks
    """
    # ALWAYS define dimensions first (prevents "dimensions not defined" bugs)
    dimensions: Dict[str, List[str]] = {}

    # --- Robust house count detection ---
    houses: Optional[int] = None
    txt = desc_text

    # Pattern A: "There are 6 houses" / "There are six houses" (also 'homes')
    m = re.search(
        r"there\s+are\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(houses|homes)\b",
        txt,
        re.IGNORECASE,
    )
    if m:
        houses = _word_or_digit_to_int(m.group(1))

    # Pattern B: "Three friends live in three houses in a row"
    if houses is None:
        m = re.search(
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+\w+\s+live\s+in\s+"
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(houses|homes)\b",
            txt,
            re.IGNORECASE,
        )
        if m:
            houses = _word_or_digit_to_int(m.group(2))

    # Pattern C: "numbered 1 to 6"
    if houses is None:
        m = re.search(r"numbered\s+1\s+to\s+(\d+)", txt, re.IGNORECASE)
        if m:
            houses = int(m.group(1))

    # --- Parse dimensions/domains from bullet lines ---
    lines = [l.strip() for l in desc_text.splitlines() if l.strip()]

    STOPWORDS = {
        "each", "person", "people", "has", "have", "a", "an",
        "unique", "different", "favorite", "various", "distinct",
        "their", "his", "her", "the", "with", "of", "style", "type",
        "lives", "live", "in"
    }

    def _split_csv_list(s: str) -> List[str]:
        s = s.strip().strip(".")
        return [p.strip() for p in s.split(",") if p.strip()]

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # --- Case 1: old ZebraLogicBench bullet lines with backticks ---
        if s.startswith("-"):
            values = re.findall(r"`([^`]+)`", s)
            if not values:
                continue

            without_dash = s[1:].strip()
            before_colon = without_dash.split(":", 1)[0]

            tokens = [t.strip().lower().rstrip(".,;:") for t in before_colon.split()]
            content_tokens = [t for t in tokens if t and t not in STOPWORDS]

            if content_tokens:
                dim_key = content_tokens[-1]
                if dim_key.endswith("s") and not dim_key.endswith("ss"):
                    dim_key = dim_key[:-1]
            else:
                dim_key = f"attr_{len(dimensions) + 1}"

            dimensions[dim_key] = [v.strip() for v in values]
            continue

        # --- Case 2: new format lines like "Colors: orange, blue, green." ---
        if ":" in s:
            left, right = s.split(":", 1)
            dim_key = left.strip().lower().rstrip(".")
            values = _split_csv_list(right)

            # keep only real domain lines
            if len(values) >= 2:
                if dim_key.endswith("s") and not dim_key.endswith("ss"):
                    dim_key = dim_key[:-1]
                dimensions[dim_key] = values


    # Fallback: infer houses from any dimension length
    if houses is None:
        if dimensions:
            first_dim = next(iter(dimensions))
            houses = len(dimensions[first_dim])
        else:
            raise ValueError("Could not determine number of houses (and could not infer from domains).")

    return PuzzleSkeleton(houses=houses, dimensions=dimensions)


def extract_clues(clue_lines: List[str]) -> List[Clue]:
    """Turn numbered clue lines into Clue(number, text). Supports multi-line clues."""
    pattern = re.compile(r"^\s*(\d+)\.\s*(.*)$")
    clues: List[Clue] = []
    current_number: Optional[int] = None
    current_parts: List[str] = []

    for line in clue_lines:
        m = pattern.match(line.strip())
        if m:
            if current_number is not None:
                full_text = " ".join(p.strip() for p in current_parts).strip()
                clues.append(Clue(number=current_number, text=full_text))
            current_number = int(m.group(1))
            current_parts = [m.group(2)]
        else:
            if current_number is not None:
                current_parts.append(line.strip())

    if current_number is not None:
        full_text = " ".join(p.strip() for p in current_parts).strip()
        clues.append(Clue(number=current_number, text=full_text))

    return clues


def build_value_index(dimensions: Dict[str, List[str]]) -> Dict[str, ItemRef]:
    """
    Build lookup normalized value -> ItemRef.
    Adds simple adjective variants (root-ish).
    """
    index: Dict[str, ItemRef] = {}
    for dim, values in dimensions.items():
        for v in values:
            norm = normalize_text(v)
            ref = ItemRef(dim=dim, value=v)
            index[norm] = ref

            root = norm
            variants = []
            if root.endswith("e"):
                variants.append(root[:-1] + "ish")
            variants.append(root + "ish")

            for var in variants:
                index.setdefault(var, ref)
    return index


def find_items_in_text(value_index: Dict[str, ItemRef], text: str) -> List[ItemRef]:
    """Find all known values in the clue text using normalized substring match."""
    t_norm = normalize_text(text)
    found: List[ItemRef] = []
    for key, item in value_index.items():
        if key and key in t_norm:
            found.append(item)
    return found


def parse_single_clue(clue: Clue, value_index: Dict[str, ItemRef]) -> Optional[Constraint]:
    """
    Convert one clue sentence into a Constraint.
    (Same patterns as your current parser.)
    """
    t = clue.text.lower()

    m_not_pos_word = re.search(r"is not in the (first|second|third|fourth|fifth|sixth) house", t)
    if m_not_pos_word:
        pos = ORDINAL_WORDS[m_not_pos_word.group(1)]
        before = t[:m_not_pos_word.start()]
        items = find_items_in_text(value_index, before)
        if items:
            return Constraint(type="not_position_equals", items=[items[0]], position=pos)

    m_pos_word = re.search(r"is in the (first|second|third|fourth|fifth|sixth) house", t)
    if m_pos_word:
        pos = ORDINAL_WORDS[m_pos_word.group(1)]
        before = t[:m_pos_word.start()]
        items = find_items_in_text(value_index, before)
        if items:
            return Constraint(type="position_equals", items=[items[0]], position=pos)

    m_not_pos = re.search(r"is not in the (\d+)(st|nd|rd|th)? house", t)
    if m_not_pos:
        pos = int(m_not_pos.group(1))
        before = t[:m_not_pos.start()]
        items = find_items_in_text(value_index, before)
        if items:
            return Constraint(type="not_position_equals", items=[items[0]], position=pos)

    m_pos = re.search(r"in the (\d+)(st|nd|rd|th)? house", t)
    if m_pos:
        pos = int(m_pos.group(1))
        before = t[:m_pos.start()]
        items = find_items_in_text(value_index, before)
        if items:
            return Constraint(type="position_equals", items=[items[0]], position=pos)

    if "one house between" in t:
        items = find_items_in_text(value_index, t)
        if len(items) >= 2:
            return Constraint(type="distance", items=items[:2], distance=2)

    if "two houses between" in t:
        items = find_items_in_text(value_index, t)
        if len(items) >= 2:
            return Constraint(type="distance", items=items[:2], distance=3)

    if "directly left of" in t:
        left_part, right_part = t.split("directly left of", 1)
        left_items = find_items_in_text(value_index, left_part)
        right_items = find_items_in_text(value_index, right_part)
        if left_items and right_items:
            return Constraint(type="offset", items=[left_items[0], right_items[0]], distance=1)

    if "somewhere to the left of" in t:
        left_part, right_part = t.split("somewhere to the left of", 1)
        left_items = find_items_in_text(value_index, left_part)
        right_items = find_items_in_text(value_index, right_part)
        if left_items and right_items:
            return Constraint(type="left_of", items=[left_items[0], right_items[0]])

    if "somewhere to the right of" in t:
        left_part, right_part = t.split("somewhere to the right of", 1)
        left_items = find_items_in_text(value_index, left_part)
        right_items = find_items_in_text(value_index, right_part)
        if left_items and right_items:
            return Constraint(type="right_of", items=[left_items[0], right_items[0]])

    if "next to each other" in t:
        before = t.split("next to each other", 1)[0]
        parts = before.split(" and ")
        if len(parts) == 2:
            items_a = find_items_in_text(value_index, parts[0])
            items_b = find_items_in_text(value_index, parts[1])
            if items_a and items_b:
                return Constraint(type="next_to", items=[items_a[0], items_b[0]])

    all_items = find_items_in_text(value_index, t)
    if len(all_items) == 2:
        return Constraint(type="same_house", items=[all_items[0], all_items[1]])

        # immediately to the left of  -> X is directly left of Y
    if "immediately to the left of" in t:
        left_part, right_part = t.split("immediately to the left of", 1)
        left_items = find_items_in_text(value_index, left_part)
        right_items = find_items_in_text(value_index, right_part)
        if left_items and right_items:
            return Constraint(type="offset", items=[left_items[0], right_items[0]], distance=1)

    # immediately to the right of -> X is directly right of Y
    if "immediately to the right of" in t:
        left_part, right_part = t.split("immediately to the right of", 1)
        left_items = find_items_in_text(value_index, left_part)
        right_items = find_items_in_text(value_index, right_part)
        if left_items and right_items:
            # represent "X right of Y" as "Y directly left of X"
            return Constraint(type="offset", items=[right_items[0], left_items[0]], distance=1)


    if (" is " in t and "between" not in t and "left of" not in t and "right of" not in t
            and "next to" not in t and "house" not in t):
        left, right = t.split(" is ", 1)
        left_items = find_items_in_text(value_index, left)
        right_clean = right.replace("the ", "").replace(".", "").strip()
        right_items = find_items_in_text(value_index, right_clean)
        if left_items and right_items:
            return Constraint(type="same_house", items=[left_items[0], right_items[0]])

    return None


def clues_to_constraint_objects(dimensions: Dict[str, List[str]], clues: List[Clue]) -> List[Constraint]:
    """Parse all clues into Constraint objects; print warnings for unparsed clues."""
    value_index = build_value_index(dimensions)
    constraints: List[Constraint] = []
    for clue in clues:
        c = parse_single_clue(clue, value_index)
        if c is None:
            print(f"[WARN] Clue {clue.number} not parsed: {clue.text}")
        else:
            constraints.append(c)
    return constraints


def constraint_to_expression_dict(c: Constraint) -> dict:
    """Convert internal Constraint object into dict under 'expression'."""
    return {
        "type": c.type,
        "items": [{"dim": it.dim, "value": it.value} for it in c.items],
        "position": c.position,
        "distance": c.distance,
    }


def puzzle_text_to_csp(text: str) -> Dict:
    """
    Puzzle string -> CSP dict:
      - houses
      - variables (dimensions/domains)
      - constraints: list of {"expression": {...}}
    """
    desc_text, clue_lines = split_description_and_clues(text)
    skeleton = parse_description(desc_text)
    clues = extract_clues(clue_lines)

def infer_names_from_clues(clues, houses: int):
    blacklist = {
        "The", "House", "Clues", "Colors", "Pets",
        "There", "Each", "Three", "Four", "Five", "Six",
        "Orange", "Purple", "Green", "Blue", "Red",
        "Yellow", "White", "Black"
    }
    found = []
    for c in clues:
        tokens = re.findall(r"\b[A-Z][a-z]+\b", c.text)
        for t in tokens:
            if t not in blacklist:
                found.append(t)

    uniq = []
    seen = set()
    for n in found:
        if n not in seen:
            seen.add(n)
            uniq.append(n)

    return uniq[:houses]


def puzzle_text_to_csp(text: str) -> Dict:
    """
    Puzzle string -> CSP dict:
      - houses
      - variables (dimensions/domains)
      - constraints: list of {"expression": {...}}
    """
    desc_text, clue_lines = split_description_and_clues(text)
    skeleton = parse_description(desc_text)
    clues = extract_clues(clue_lines)

    if "name" not in skeleton.dimensions:
        names = infer_names_from_clues(clues, skeleton.houses)
        if names:
            skeleton.dimensions["name"] = names

    constraint_objects = clues_to_constraint_objects(skeleton.dimensions, clues)
    constraints = [{"expression": constraint_to_expression_dict(c)} for c in constraint_objects]

    return {
        "houses": skeleton.houses,
        "variables": skeleton.dimensions,
        "constraints": constraints,
    }

# =================================== Run ===================================

"""
Main runner script for the Zebra Logic Puzzle CSP Solver.

This script:
1. Loads puzzles from the ZebraLogicBench dataset (parquet format)
2. Parses each puzzle into CSP format
3. Solves using backtracking with MRV and forward checking
4. Evaluates accuracy and efficiency
5. Outputs results and statistics
"""

@dataclass
class PuzzleResult:
    """Store results for a single puzzle."""
    puzzle_id: str
    solved: bool
    correct: bool
    solution: Optional[Dict]
    expected: Optional[Dict]
    stats: SolverStats
    time_seconds: float
    error: Optional[str] = None

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the puzzle dataset from parquet file.
    
    Args:
        filepath: Path to the parquet file
    
    Returns:
        DataFrame with puzzle data
    """
    print(f"Loading dataset from {filepath}...")
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df)} puzzles")
    return df

def parse_expected_solution(solution_data: Any) -> Optional[Dict[int, Dict[str, str]]]:
    """
    Parse the expected solution from dataset format to our format.
    
    The dataset stores solutions as:
    - JSON string or dict with header and rows
    - We convert to {house_num: {dim: value}}
    """
    if solution_data is None:
        return None
    
    try:
        # Handle string format
        if isinstance(solution_data, str):
            solution_data = json.loads(solution_data)
        
        # Expected format: {"header": [...], "rows": [[...], [...]]}
        if isinstance(solution_data, dict) and "header" in solution_data:
            header = solution_data["header"]
            rows = solution_data["rows"]
            
            result = {}
            for row in rows:
                # First column is usually house number
                house_num = int(row[0]) if isinstance(row[0], (int, str)) else row[0]
                result[house_num] = {}
                for i, col in enumerate(header[1:], start=1):
                    if i < len(row):
                        result[house_num][col.lower()] = row[i]
            
            return result
        
        return solution_data
    except Exception as e:
        print(f"Warning: Could not parse solution: {e}")
        return None

def compare_solutions(computed: Optional[Dict], expected: Optional[Dict]) -> bool:
    """
    Compare computed solution with expected solution.
    
    Handles differences in key naming and formatting.
    Returns True if solutions match.
    """
    if computed is None or expected is None:
        return False
    
    # Normalize both solutions for comparison
    def normalize(sol: Dict) -> Dict:
        normalized = {}
        for house, attrs in sol.items():
            house_key = int(house) if not isinstance(house, int) else house
            normalized[house_key] = {}
            for dim, val in attrs.items():
                # Normalize dimension and value to lowercase
                dim_norm = dim.lower().strip()
                val_norm = str(val).lower().strip()
                normalized[house_key][dim_norm] = val_norm
        return normalized
    
    try:
        comp_norm = normalize(computed)
        exp_norm = normalize(expected)
        
        # Check if all houses match
        if set(comp_norm.keys()) != set(exp_norm.keys()):
            return False
        
        for house in comp_norm:
            comp_attrs = comp_norm[house]
            exp_attrs = exp_norm.get(house, {})
            
            # Check common dimensions
            for dim in set(comp_attrs.keys()) & set(exp_attrs.keys()):
                if comp_attrs[dim] != exp_attrs[dim]:
                    return False
        
        return True
    except Exception:
        return False

def solve_single_puzzle(puzzle_id: str, puzzle_text: str, 
                        expected_solution: Any = None) -> PuzzleResult:
    """
    Solve a single puzzle and return results.
    
    Args:
        puzzle_id: Unique identifier for the puzzle
        puzzle_text: The puzzle description text
        expected_solution: Expected solution for validation
    
    Returns:
        PuzzleResult with solution and statistics
    """
    start_time = time.time()
    error = None
    solution = None
    stats = SolverStats()
    
    try:
        # Step 1: Parse puzzle into CSP format
        csp = puzzle_text_to_csp(puzzle_text)
        
        # Step 2: Solve the CSP
        solution, stats = solve_puzzle(csp)
        
    except Exception as e:
        error = str(e)
    
    elapsed = time.time() - start_time
    
    # Parse expected solution
    expected = parse_expected_solution(expected_solution)
    
    # Determine if solution is correct
    solved = solution is not None
    correct = compare_solutions(solution, expected) if expected else solved
    
    return PuzzleResult(
        puzzle_id=puzzle_id,
        solved=solved,
        correct=correct,
        solution=solution,
        expected=expected,
        stats=stats,
        time_seconds=elapsed,
        error=error
    )

def print_solution(solution: Dict[int, Dict[str, str]]):
    """Pretty print a puzzle solution as a table."""
    if not solution:
        print("No solution found")
        return
    
    # Get all dimensions
    all_dims = set()
    for attrs in solution.values():
        all_dims.update(attrs.keys())
    dims = sorted(all_dims)
    
    # Print header
    header = ["House"] + dims
    col_widths = [max(len(h), 10) for h in header]
    
    # Update widths based on content
    for house in sorted(solution.keys()):
        for i, dim in enumerate(dims):
            val = solution[house].get(dim, "")
            col_widths[i + 1] = max(col_widths[i + 1], len(str(val)))
    
    # Print table
    header_line = " | ".join(h.ljust(w) for h, w in zip(header, col_widths))
    print(header_line)
    print("-" * len(header_line))
    
    for house in sorted(solution.keys()):
        row = [str(house)]
        for dim in dims:
            row.append(solution[house].get(dim, ""))
        row_line = " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(row_line)

def run_evaluation(df: pd.DataFrame, max_puzzles: int = -1, 
                   verbose: bool = True) -> List[PuzzleResult]:
    """
    Run evaluation on all puzzles in the dataset.
    
    Args:
        df: DataFrame with puzzles
        max_puzzles: Maximum number of puzzles to solve (None for all)
        verbose: Whether to print progress
    
    Returns:
        List of PuzzleResult for each puzzle
    """
    results = []
    
    # Determine columns
    id_col = "id" if "id" in df.columns else df.columns[0]
    puzzle_col = "puzzle" if "puzzle" in df.columns else df.columns[1]
    solution_col = "solution" if "solution" in df.columns else None
    
    puzzles_to_solve = df.head(max_puzzles) if not max_puzzles == -1 else df
    total = len(puzzles_to_solve)
    
    print(f"\n{'='*60}")
    print(f"Starting evaluation on {total} puzzles")
    print(f"{'='*60}\n")
    
    for idx, row in puzzles_to_solve.iterrows():
        puzzle_id = str(row[id_col])
        puzzle_text = row[puzzle_col]
        expected = row[solution_col] if solution_col else None
        
        if verbose:
            print(f"\n[{len(results)+1}/{total}] Solving puzzle: {puzzle_id}")
        
        result = solve_single_puzzle(puzzle_id, puzzle_text, expected)
        results.append(result)
        
        if verbose:
            status = "✓ CORRECT" if result.correct else ("✗ WRONG" if result.solved else "✗ FAILED")
            print(f"  Status: {status}")
            print(f"  Time: {result.time_seconds:.3f}s")
            print(f"  Stats: {result.stats.nodes_explored} nodes, "
                  f"{result.stats.backtracks} backtracks")
            
            if result.error:
                print(f"  Error: {result.error}")
    
    return results

def print_summary(results: List[PuzzleResult]):
    """Print evaluation summary statistics."""
    total = len(results)
    solved = sum(1 for r in results if r.solved)
    correct = sum(1 for r in results if r.correct)
    
    total_time = sum(r.time_seconds for r in results)
    avg_time = total_time / total if total > 0 else 0
    
    total_nodes = sum(r.stats.nodes_explored for r in results)
    total_backtracks = sum(r.stats.backtracks for r in results)
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total puzzles:     {total}")
    print(f"Solved:            {solved} ({100*solved/total:.1f}%)")
    print(f"Correct:           {correct} ({100*correct/total:.1f}%)")
    print(f"{'='*60}")
    print(f"Total time:        {total_time:.2f}s")
    print(f"Average time:      {avg_time:.3f}s per puzzle")
    print(f"Total nodes:       {total_nodes}")
    print(f"Total backtracks:  {total_backtracks}")
    print(f"{'='*60}")
    
    # Show failed puzzles
    failed = [r for r in results if not r.solved or not r.correct]
    if failed:
        print(f"\nFailed puzzles ({len(failed)}):")
        for r in failed[:10]:  # Show first 10
            print(f"  - {r.puzzle_id}: {'Parse error' if r.error else 'Incorrect solution'}")

import csv

import csv

def convert_solution_to_grid_format(solution: Optional[Dict], variables: Dict[str, List[str]]) -> str:
    """
    Convert our solution format to the required grid_solution format.
    
    Required:   {"header": ["House", "Name", "Car"], "rows": [["1", "Alice", "Ford"], ...]}
    """
    if solution is None:
        return ""
    
    # Get dimensions from solution 
    dims = set()
    for house_attrs in solution.values():
        dims.update(house_attrs.keys())
    dims = sorted(dims)
    
    # Build header: "House" + capitalized dimension names
    header = ["House"] + [dim.capitalize() for dim in dims]
    
    # Build rows: each house as a list of strings
    rows = []
    for house_num in sorted(solution.keys()):
        row = [str(house_num)]
        for dim in dims:
            val = solution[house_num].get(dim, "")
            row.append(str(val))
        rows.append(row)
    
    # Create the grid_solution dict
    grid_solution = {
        "header": header,
        "rows": rows
    }
    
    # Convert to JSON string (this handles the escaping)
    return json.dumps(grid_solution)

def save_results_to_csv_detailed(results: List[PuzzleResult], filepath: str = "results.csv"):
    """
    Save evaluation results to CSV file.
    
    Columns: puzzle_id, solved, correct, time_seconds, nodes_explored, 
             backtracks, arc_revisions, steps, solution_json, error
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'puzzle_id', 'solved', 'correct', 'time_seconds',
            'nodes_explored', 'backtracks', 'arc_revisions', 'steps', 'error'
        ])
        
        # Data rows
        for r in results:
            #converting solution (dict into json str)
            solution_str = json.dumps(r.solution) if r.solution else ''
            writer.writerow([
                r.puzzle_id,
                r.solved,
                r.correct,
                f"{r.time_seconds:.4f}",
                r.stats.nodes_explored,
                r.stats.backtracks,
                r.stats.arc_revisions,
                r.stats.steps,
                solution_str,
                r.error or ''
            ])
    
    print(f"Results saved to {filepath}")

def save_results_to_csv(results: List[PuzzleResult], filepath: str = "results.csv"):
    """
    Save evaluation results to CSV file in the REQUIRED format:
    
    Columns: id, grid_solution, steps
    
    grid_solution format: {"header": [...], "rows": [[...], ...]}
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header (exactly as required)
        writer.writerow(['id', 'grid_solution', 'steps'])
        
        # Data rows
        for r in results:
            # Convert solution to required grid format
            grid_solution = convert_solution_to_grid_format(r.solution, {})
            
            writer.writerow([
                r.puzzle_id,                    # id
                grid_solution,                  # grid_solution (JSON string)
                r.stats.steps                   # steps
            ])
    
    print(f"Results saved to {filepath}")
    print(f"Total rows: {len(results)} (+ 1 header = {len(results) + 1} lines)")

def main():
    """Main entry point for the solver."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Zebra Logic Puzzle CSP Solver")
    parser.add_argument("--data", type=str, 
                        default="Gridmode-00000-of-00001.parquet",
                        help="Path to puzzle dataset (parquet)")
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum puzzles to solve")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed progress")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    parser.add_argument("--single", type=int, default=None,
                        help="Solve single puzzle by index")
    
    args = parser.parse_args()
    
    df = load_dataset(args.data)
    df = normalize_dataset(df)

# Debug info (remove later)

    print("Using dataset:", args.data)
    print("Columns after normalize:", list(df.columns))

    sample = df.iloc[0]["puzzle"]
    print("Puzzle type:", type(sample))
    print("Puzzle preview:", str(sample)[:250].replace("\n", "\\n"))

    # Quick sanity check: does it look like a Zebra puzzle?
    s = str(sample).lower()
    print("Looks like puzzle:", ("there are" in s and "houses" in s and "## clues" in s))

# Debug info end (remove later)

    # Load dataset
    df = load_dataset(args.data)
    
    if args.single is not None:
        # Solve single puzzle
        row = df.iloc[args.single]
        puzzle_id = str(row.get("id", args.single))
        puzzle_text = row["puzzle"]
        solution = row.get("solution")
        
        print(f"\nPuzzle: {puzzle_id}")
        print(f"\n{puzzle_text[:500]}...")
        
        result = solve_single_puzzle(puzzle_id, puzzle_text, solution)
        
        if result.solution:
            print(f"\n{'='*60}")
            print("SOLUTION")
            print(f"{'='*60}")
            print_solution(result.solution)
        else:
            print(f"\n{'='*60}")
            print("NO SOLUTION FOUND")
            print(f"{'='*60}")
        
        print(f"\n{'='*60}")
        print("STATISTICS")
        print(f"{'='*60}")
        print(f"Solved: {result.solved}")
        print(f"Correct: {result.correct}")
        print(f"Time: {result.time_seconds:.3f}s")
        print(f"Nodes explored: {result.stats.nodes_explored}")
        print(f"Backtracks: {result.stats.backtracks}")
        print(f"Arc revisions: {result.stats.arc_revisions}")
        
    else:
        # Run full evaluation
        verbose = args.verbose and not args.quiet
        results = run_evaluation(df, max_puzzles=args.max, verbose=verbose)
        print_summary(results)
        save_results_to_csv(results, "evaluation_results.csv")
        
        # verifying the row count 
        if len(results) != 100:
            print(f"\n⚠️  WARNING: exceeded 100 puzzles, got {len(results)}")
            print(f"    CSV has {len(results) + 1} lines (including header)")

# Alternative entry point for direct module execution
def run():
    """
    Simplified run function that can be called from other modules.
    Returns the DataFrame and results for further processing.
    """
    # Load dataset
    df = load_dataset("Gridmode-00000-of-00001.parquet")
    
    # Run evaluation on first few puzzles
    results = run_evaluation(df, max_puzzles=5, verbose=True)
    
    # Print summary
    print_summary(results)
    
    return df, results

# Normalize dataset after loading

def _looks_like_puzzle_text(x: Any) -> bool:
    if not isinstance(x, str):
        return False
    t = x.strip()
    return ("## clues" in t.lower()) and ("there are" in t.lower()) and ("houses" in t.lower())

def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has at least:
      - df["puzzle"]   (string puzzle text)
      - df["solution"] (optional)
      - df["id"]       (optional)
    Works even if the parquet uses different column names or stores dict/JSON.
    """
    # 1) If there's a single column that stores dicts/JSON with keys like "puzzle"
    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        v = sample.iloc[0]

        # dict case
        if isinstance(v, dict) and ("puzzle" in v or "text" in v or "problem" in v):
            # expand dict column into separate columns
            expanded = pd.json_normalize(df[col])
            # keep original columns too (id might be outside)
            for k in expanded.columns:
                if k not in df.columns:
                    df[k] = expanded[k]

        # JSON string case
        if isinstance(v, str) and v.strip().startswith("{") and '"puzzle"' in v:
            try:
                obj0 = json.loads(v)
                if isinstance(obj0, dict):
                    expanded = pd.json_normalize(df[col].apply(lambda s: json.loads(s) if isinstance(s, str) else {}))
                    for k in expanded.columns:
                        if k not in df.columns:
                            df[k] = expanded[k]
            except Exception:
                pass

    # 2) Determine which column is the puzzle text
    if "puzzle" not in df.columns:
        # common alternatives
        candidates = ["text", "problem", "prompt", "puzzle_text", "question", "input"]
        for c in candidates:
            if c in df.columns:
                df["puzzle"] = df[c].astype(str)
                break

    # 3) If still no puzzle column, auto-detect by content
    if "puzzle" not in df.columns:
        for col in df.columns:
            # only check object-like columns
            if df[col].dtype != "object":
                continue
            sample = df[col].dropna().astype(str).head(20)
            if any(_looks_like_puzzle_text(s) for s in sample):
                df["puzzle"] = df[col].astype(str)
                break

    # 4) Optional: map solution column similarly
    if "solution" not in df.columns:
        for c in ["answer", "target", "ground_truth", "gt", "label", "solutions"]:
            if c in df.columns:
                df["solution"] = df[c]
                break

    # 5) Optional: ensure id exists
    if "id" not in df.columns:
        if "puzzle_id" in df.columns:
            df["id"] = df["puzzle_id"]
        else:
            df["id"] = [f"idx_{i}" for i in range(len(df))]

    # Final sanity check
    if "puzzle" not in df.columns:
        raise ValueError(
            "Could not find puzzle text column in dataset. "
            f"Columns are: {list(df.columns)}"
        )

    return df

# end new changes

if __name__ == "__main__":
    main()