from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
import re
import pandas as pd

# Global ordinal word mapping (dataset uses at most 6 houses)

ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
}

# Normalization helpers

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
    # remove apostrophes (both ' and ’)
    s = re.sub(r"[’']", "", s)
    # replace hyphens with spaces
    s = s.replace("-", " ")
    # remove the word 'person'
    s = re.sub(r"\bperson\b", "", s)
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# Data classes (internal)

@dataclass
class Clue:
    number: int
    text: str

@dataclass
class PuzzleSkeleton:
    houses: int
    domains: Dict[str, List[str]]  # e.g. {"name": [...], "car": [...], ...}

@dataclass
class Attribute:
    domain: str
    value: str

@dataclass
class Puzzle:
    id: int = 0
    entities: int = 0
    variables: dict = field(default_factory=dict)
    constraints: list = field(default_factory=list)

# 1. Split puzzle into description + clues

def split_description_and_clues(text: str):
    """
    Takes the full puzzle string.
    Splits into:
    - desc_text: everything before '## Clues'
    - clue_lines: non-empty lines in the clues section
    """
    lines = text.splitlines()
    desc_lines = []
    clue_lines = []
    in_clues = False

    for line in lines:
        stripped = line.strip()
        if not in_clues:
            if "Clues:" in stripped:
                in_clues = True
            else:
                desc_lines.append(line)
        else:
            if stripped:  # skip empty lines
                clue_lines.append(line.rstrip("\n"))

    return "\n".join(desc_lines), clue_lines

# 2. Parse description → houses + generic domains

def parse_description(desc_text: str) -> PuzzleSkeleton:
    """
    Parse the description part into:
    - number of houses
    - domains (domain per attribute), fully generic.

    Any bullet line with backtick values is treated as a dimension, e.g.:

    - Each person has a unique car: `Ford F-150`, `Honda Civic`
    - People use unique phone models: `iphone 13`, `google pixel 6`
    - Each person has a favorite drink: `tea`, `coffee`

    The dimension key is derived from the last meaningful word before ':'.
    """
    lines = [l.strip() for l in desc_text.splitlines() if l.strip()]

    # 1) find number of houses
    m = re.search(r"There are\s+(\d+)\s+houses", desc_text)
    if not m:
        raise ValueError("Could not find number of houses")
    houses = int(m.group(1))

    domains: Dict[str, List[str]] = {}

    # simple stopword set for deriving dimension names
    STOPWORDS = {
        "each", "person", "people", "has", "have", "a", "an",
        "unique", "different", "favorite", "various", "distinct",
        "their", "his", "her", "the", "with", "of", "style", "type"
    }

    for line in lines:
        if not line.startswith("-"):
            continue

        # extract values in backticks
        values = re.findall(r"`([^`]+)`", line)
        if not values:
            continue

        # part before ':' describes the attribute
        without_dash = line[1:].strip()            # remove leading '-'
        before_colon = without_dash.split(":", 1)[0]

        tokens = [t.strip().lower().rstrip(".,;:")
                  for t in before_colon.split()]
        content_tokens = [t for t in tokens if t and t not in STOPWORDS]

        if content_tokens:
            # pick last meaningful word, e.g. "car", "job", "model", "drink", "color"
            dim_key = content_tokens[-1]
            # trivial plural handling: trim trailing 's' if present
            if dim_key.endswith("s") and not dim_key.endswith("ss"):
                dim_key = dim_key[:-1]
        else:
            dim_key = f"attr_{len(domains) + 1}"

        domains[dim_key] = [v.strip() for v in values]

    return PuzzleSkeleton(houses=houses, domains=domains)

# 3. Clue lines → Clue objects

def extract_clues(clue_lines: List[str]) -> List[Clue]:
    """
    Turn numbered clue lines into Clue(number, text).
    Supports multi-line clues.

    Lines like: '1. The German is Bob.'
    start a new clue; following lines (without leading number) are appended.
    """
    pattern = re.compile(r"^\s*(\d+)\.\s*(.*)$")
    clues: List[Clue] = []
    current_number: Optional[int] = None
    current_parts: List[str] = []

    for line in clue_lines:
        m = pattern.match(line.strip())
        if m:
            # finish previous clue
            if current_number is not None:
                full_text = " ".join(p.strip() for p in current_parts).strip()
                clues.append(Clue(number=current_number, text=full_text))

            current_number = int(m.group(1))
            current_parts = [m.group(2)]
        else:
            # continuation of the current clue
            if current_number is not None:
                current_parts.append(line.strip())

    # add last clue
    if current_number is not None:
        full_text = " ".join(p.strip() for p in current_parts).strip()
        clues.append(Clue(number=current_number, text=full_text))

    return clues

# 4. Value index (find items in clue text)

def build_value_index(domains: Dict[str, List[str]]) -> Dict[str, Attribute]:
    """
    Build a lookup from normalized value string to Attribute.
    Also adds simple adjective variants (…ish) so that e.g.
    'swede' can be matched by 'swedish person'.
    """
    index: Dict[str, Attribute] = {}

    for domain, values in domains.items():
        for v in values:
            norm = normalize_text(v)  # e.g. 'swede', 'bachelors degree'
            ref = Attribute(domain=domain, value=v)
            index[norm] = ref

            # Heuristic adjective forms:
            # 'swede' -> 'swedish', 'dane' -> 'danish', 'brit' -> 'british'
            # This is generic and not nationality-specific.
            root = norm
            variants = []

            if root.endswith("e"):
                # swede -> swed + ish = swedish, dane -> danish
                variants.append(root[:-1] + "ish")

            # brit -> british, etc. (for some values this will just be unused)
            variants.append(root + "ish")

            for var in variants:
                if var not in index:
                    index[var] = ref

    return index

def find_items_in_text(value_index: Dict[str, Attribute], text: str) -> List[Attribute]:
    """
    Find all known values in the clue text using normalized strings.
    """
    t_norm = normalize_text(text)
    found: List[Attribute] = []
    for key, item in value_index.items():
        if key in t_norm:
            found.append(item)
    return found

# 5. Single clue → Constraint

def parse_single_clue(clue: Clue, value_index: Dict[str, Attribute]) -> Optional[tuple]:
    """
    Convert a single clue sentence into a Constraint object.
    Supported patterns:
    - "X is in the first/second/.../sixth house"          -> position_equals
    - "X is not in the first/second/.../sixth house"      -> not_position_equals
    - "X is in the 1st/2nd/... house"                     -> position_equals
    - "X is not in the 1st/2nd/... house"                 -> not_position_equals
    - "There is one house between X and Y"                -> distance=2
    - "There are two houses between X and Y"              -> distance=3
    - "X is directly left of Y"                           -> offset (distance=1)
    - "X is somewhere to the left of Y"                   -> left_of
    - "X is somewhere to the right of Y"                  -> right_of
    - "X and Y are next to each other"                    -> next_to
    - "X is Y" / "The person who ... is Z" (two items)    -> same_house
    """
    t = clue.text.lower()

    # 1) Negative position with word ordinals: "X is not in the fourth house."
    m_not_pos_word = re.search(
        r"is not in the (first|second|third|fourth|fifth|sixth) house",
        t
    )
    if m_not_pos_word:
        word = m_not_pos_word.group(1)
        pos = ORDINAL_WORDS[word]
        before = t[:m_not_pos_word.start()]
        items = find_items_in_text(value_index, before)
        if items:
            return ('!=', items[0], pos)

    # 2) Positive position with word ordinals: "X is in the first/third/sixth house."
    m_pos_word = re.search(
        r"is in the (first|second|third|fourth|fifth|sixth) house",
        t
    )
    if m_pos_word:
        word = m_pos_word.group(1)
        pos = ORDINAL_WORDS[word]
        before = t[:m_pos_word.start()]
        items = find_items_in_text(value_index, before)
        if items:
            return ('=', items[0], pos)

    # 3) Negative position with digits: "X is not in the 4th house."
    m_not_pos = re.search(r"is not in the (\d+)(st|nd|rd|th)? house", t)
    if m_not_pos:
        pos = int(m_not_pos.group(1))
        before = t[:m_not_pos.start()]
        items = find_items_in_text(value_index, before)
        if items:
            return ('!=', items[0], pos)

    # 4) Positive position with digits: "X is in the 2nd house."
    m_pos = re.search(r"in the (\d+)(st|nd|rd|th)? house", t)
    if m_pos:
        pos = int(m_pos.group(1))
        before = t[:m_pos.start()]
        items = find_items_in_text(value_index, before)
        if items:
            return ('=', items[0], pos)

    # 5) Distance: "There is one house between X and Y." -> distance=2
    if "one house between" in t:
        items = find_items_in_text(value_index, t)
        if len(items) >= 2:
            return ('+-2', items[0], items[1])

    # "There are two houses between X and Y." -> distance=3
    if "two houses between" in t:
        items = find_items_in_text(value_index, t)
        if len(items) >= 2:
            return ('+-3', items[0], items[1])

    # 6) Directly left: "X is directly left of Y." -> offset
    if "directly left of" in t:
        left_part, right_part = t.split("directly left of", 1)
        left_items = find_items_in_text(value_index, left_part)
        right_items = find_items_in_text(value_index, right_part)
        if left_items and right_items:
            return ('+1', left_items[0], right_items[0])

    # 7) Somewhere to the left: "X is somewhere to the left of Y." -> left_of
    if "somewhere to the left of" in t:
        left_part, right_part = t.split("somewhere to the left of", 1)
        left_items = find_items_in_text(value_index, left_part)
        right_items = find_items_in_text(value_index, right_part)
        if left_items and right_items:
            return ('<', left_items[0], right_items[0])

    # 8) Somewhere to the right: "X is somewhere to the right of Y." -> right_of
    if "somewhere to the right of" in t:
        left_part, right_part = t.split("somewhere to the right of", 1)
        left_items = find_items_in_text(value_index, left_part)
        right_items = find_items_in_text(value_index, right_part)
        if left_items and right_items:
            return ('>', left_items[0], right_items[0])

    # 9) Next to each other: "X and Y are next to each other." -> next_to
    if "next to each other" in t:
        before = t.split("next to each other", 1)[0]
        parts = before.split(" and ")
        if len(parts) == 2:
            items_a = find_items_in_text(value_index, parts[0])
            items_b = find_items_in_text(value_index, parts[1])
            if items_a and items_b:
                return ('+-1', items_a[0], items_b[0])

    # 10) Special same_house: if exactly 2 known values appear in the sentence
    all_items = find_items_in_text(value_index, t)
    if len(all_items) == 2:
        return ('=', all_items[0], all_items[1])

    # 11) Generic fallback: "X is Y." / "X is the Y."
    if (
        " is " in t
        and "between" not in t
        and "left of" not in t
        and "right of" not in t
        and "next to" not in t
        and "house" not in t
    ):
        left, right = t.split(" is ", 1)
        left_items = find_items_in_text(value_index, left)
        right_clean = right.replace("the ", "").replace(".", "").strip()
        right_items = find_items_in_text(value_index, right_clean)
        if left_items and right_items:
            return ('=', left_items[0], right_items[0])

    # nothing matched
    return None

def clues_to_constraint_objects(domains: Dict[str, List[str]],
                                clues: List[Clue]) -> List[tuple]:
    """
    Apply parse_single_clue to all clues.
    Returns a list of internal Constraint objects.
    """
    value_index = build_value_index(domains)
    constraints: List[tuple] = []
    for clue in clues:
        c = parse_single_clue(clue, value_index)
        if c is None:
            print(f"[WARN] Clue {clue.number} not parsed: {clue.text}")
        else:
            constraints.append(c)
    return constraints

# 7. Main function: puzzle string → CSP object

def puzzle_text_to_csp(text: str) -> Puzzle:
    """
    Takes the puzzle string (as stored in the CSV 'puzzle' column)
    and returns a CSP object with:
    - 'houses'
    - 'variables' (domains/domains)
    - 'constraints': list of { "expression": { ... } }
    """
    # 1) split description and clues
    desc_text, clue_lines = split_description_and_clues(text)

    # 2) parse description
    skeleton = parse_description(desc_text)

    # 3) extract clues
    clues = extract_clues(clue_lines)

    # 4) create internal constraint objects
    constraints = clues_to_constraint_objects(skeleton.domains, clues)

    # 6) final CSP representation
    return Puzzle(entities=skeleton.houses, variables=skeleton.domains, constraints=constraints)

# 8. Primarily Called Function

def run(df: pd.DataFrame):
    """
    Imports the appropriate .parquet file and parses each constrain within into a list as a specified object.
    """

    csps = []
    solutions = []

    for _, row in df.iterrows():
        puzzle_id = row["id"]
        puzzle_text = row["puzzle"]
        try:
            solution = row["solution"]
            solutions.append(solution)
        except:
            pass
        print(f"\nPuzzle {puzzle_id} loaded\n")
        print(puzzle_text[:200], "...")

        csp = puzzle_text_to_csp(puzzle_text)
        csp.id = puzzle_id

        print("\nCSP VARIABLES / DOMAINS")
        for domain, vals in csp.variables.items():
            print(f"{domain}: {vals}")

        print("\nCSP CONSTRAINTS (expression dicts)")
        for i, c in enumerate(csp.constraints, start=1):
            print(f"{i}. {c}")
        
        csps.append(csp)

    return csps, solutions
