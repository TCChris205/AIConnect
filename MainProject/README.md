# Zebra Logic Puzzle CSP Solver

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> A Constraint Satisfaction Problem (CSP) solver for logic grid puzzles (Zebra puzzles) using backtracking with intelligent heuristics.

## Table of Contents

- [Team](#team)
- [Overview](#overview)
- [Features](#features)
- [Problem Statement](#problem-statement)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
  - [CSP Modeling](#csp-modeling)
  - [Natural Language Parser](#natural-language-parser)
  - [Solver Algorithm](#solver-algorithm)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Output Format](#output-format)
- [Performance](#performance)
- [Competition Results](#competition-results)
- [Acknowledgments](#acknowledgments)

## Team

### AI Connect 2025 - Group 6

| Name | University |
| ---- | ---------- |
| Christian Dück | HSBI (Germany) |
| Adrian Kramkowski | HSBI (Germany) |
| Linus Martinschledde | HSBI (Germany) |
| Markus Wurms | HSBI (Germany) |
| Haider Aitezaz Ali | NUST (Pakistan) |
| Abdullah Farooq | NUST (Pakistan) |
| Eesha Raees | NUST (Pakistan) |

## Overview

This project implements an intelligent CSP solver for **Zebra Logic Puzzles** (also known as Einstein's Riddle or Logic Grid Puzzles). The solver can:

- **Parse** specific natural language puzzle descriptions into formal CSP representations
- **Solve** puzzles of varying complexity (2x2 to 6x6 grids)
- **Optimize** search using MRV heuristic, forward checking, and arc consistency (AC-3)
- **Generate** structured outputs for automated evaluation

Developed for the **AI Connect 2025** international competition involving HSBI (Germany), TDU (Türkiye), SEECS/NUST (Pakistan), and CST/RUB (Bhutan).

## Features

- **Intelligent Backtracking**: MRV (Minimum Remaining Values) heuristic for variable selection
- **Constraint Propagation**: Forward checking and AC-3 for early pruning
- **Natural Language Parsing**: Converts puzzle clues into formal constraints
- **Detailed Statistics**: Tracks nodes explored, backtracks, and constraint checks
- **High Accuracy**: Solves 100+ test puzzles with detailed trace generation
- **Scalable**: Handles puzzles from 2x2 to 6x6 grids

## Problem Statement

**What is a Zebra Puzzle?**

A logic grid puzzle where you must deduce assignments between entities based on clues.

**Example:**

```md
There are 3 houses in a row, numbered 1 to 3 from left to right.
- Colors: red, blue, green
- Names: Alice, Bob, Carol
- Pets: cat, dog, fish

Clues:
1. Alice lives in the first house.
2. The person in the red house owns a cat.
3. Bob is directly left of Carol.
4. The green house is not in house 3.
```

**Goal:** Determine which person lives in which house with which color and pet.

## Installation

### Prerequisites

- Python 3.8 or higher
- pandas library

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/TCChris205/AIConnect.git
   cd AIConnect/MainProject
   ```

2. **Install dependencies:**

   ```bash
   pip install pandas pyarrow
   ```

3. **Verify installation:**

   ```bash
   python solver.py --help
   ```

## Quick Start

### Run the Solver on Test Dataset

```bash
python solver.py
```

This will:

1. Load puzzles from `Test_100_Puzzles.parquet`
2. Solve each puzzle using CSP backtracking
3. Generate `submission.csv` with results
4. Print accuracy and efficiency statistics

### Change the called file

You can either call the function `run()` from outside the file or execute the file itself. To change the file that is called, enter the name in the `FILENAME` constant right below the imports at the top of the file.

Alternatively, you can execute the python file with a different dataset given:_

```bash
python solver.py --data FILE
```

### Run on Custom Dataset

```python
from solver import solve_puzzle, puzzle_text_to_csp

puzzle_text = """
There are 3 houses.
- Name: Alice, Bob, Carol
- Color: red, blue, green

Clues:
1. Alice is in the first house.
2. Bob is in the red house.
"""

# Parse puzzle into CSP format
csp = puzzle_text_to_csp(puzzle_text)

# Solve
solution, stats = solve_puzzle(csp)

print(f"Solution: {solution}")
print(f"Nodes explored: {stats.nodes_explored}")
```

## How It Works

### CSP Modeling

Our solver represents puzzles as Constraint Satisfaction Problems:

#### **Variables**

Each house has multiple variables which have multiple possible values called domains

#### **Domains**

Each attribute of a domain can be assigned a house position from 1 to *n* (where *n* = number of houses). The same number can't be present twice within one domain.

#### **Constraints**

Clues are parsed into constraint types:

| Constraint Type | Example Clue | Formal Representation |
| -------------- | ------------ | -------------------- |
| **Unary** | "Alice is in house 1" | `position(Alice) = 1` |
| **Binary Equality** | "Alice owns the cat" | `position(Alice) = position(cat)` |
| **Binary Inequality** | "Bob is not in house 2" | `position(Bob) ≠ 2` |
| **Relative Position** | "Alice is left of Bob" | `position(Alice) < position(Bob)` |
| **Adjacent** | "Alice is next to Bob" | `\|position(Alice) - position(Bob)\| = 1` |
| **Distance** | "Two houses between Alice and Bob" | `\|position(Alice) - position(Bob)\| = 3` |

**Example:**

```md
Variables: {("name", "Alice"), ("name", "Bob"), ("color", "red"), ("pet", "cat")}
Domains: {1, 2, 3} for each variable
Constraints: [
  ("same_house", ("name", "Alice"), 1),           # Alice in house 1
  ("same_house", ("name", "Bob"), ("color", "red")), # Bob in red house
]
```

### Natural Language Parser

The parser converts puzzle text into CSP format through several stages:

#### **1. Description Parsing**

- Extracts number of houses: `"There are 3 houses"` → `houses = 3`
- Identifies dimensions and domains:

  ```md
  "- Name: Alice, Bob, Carol" → {"name": ["Alice", "Bob", "Carol"]}
  ```

#### **2. Clue Extraction**

- Identifies numbered clues using regex patterns
- Handles multi-line clues
- Normalizes text (lowercase, remove apostrophes, etc.)

#### **3. Constraint Generation**

Recognizes patterns in clues:

```python
"Alice is in the first house" → Unary constraint: (Alice, position 1)
"directly left of" → Adjacent constraint with order
"somewhere to the left of" → Less-than constraint
"next to each other" → Adjacency constraint
"one house between" → Distance constraint (distance = 2)
```

#### **4. Value Indexing**

- Builds lookup table: normalized text → attribute references
- Handles variations: "knitting" → "knit", "photography" → "photograph"

### Solver Algorithm

Our solver uses **backtracking search** with three key optimizations:

#### **1. MRV (Minimum Remaining Values) Heuristic**

- Selects the variable with the fewest legal values first
- Reduces branching factor early in the search
- "Fail-first" principle: detect inconsistencies sooner

```python
def select_variable(domains):
    return min(unassigned_variables, 
               key=lambda v: len(domains[v]))
```

#### **2. Forward Checking**

- After each assignment, prune inconsistent values from neighboring variables
- Immediate detection of dead-ends
- Significantly reduces search space

```python
def forward_check(assignment, var, value):
    for neighbor in get_constrained_neighbors(var):
        remove_inconsistent_values(neighbor, var, value)
```

#### **3. Arc Consistency (AC-3)**

- Ensures binary constraints are satisfied throughout the search
- Propagates constraints transitively
- More thorough than forward checking alone

```python
def ac3(domains, constraints):
    queue = all_arcs_from_constraints()
    while queue:
        (xi, xj) = queue.pop()
        if revise(xi, xj):
            queue.extend(all_arcs_to(xi))
```

#### **Search Process**

1. **Initialize**: All variables have full domains {1, 2, ..., n}
2. **Select Variable**: Choose variable with minimum remaining values (MRV)
3. **Try Values**: For each value in domain:
   - Assign value to variable
   - Apply forward checking
   - Apply AC-3 for constraint propagation
   - If consistent, recurse
   - If dead-end, backtrack
4. **Solution Found**: When all variables are assigned consistently

**Example Trace:**

```md
Initial State: All domains = {1, 2, 3}

Assign Alice = 1 (from clue)
→ Forward check: All other variables ≠ 1

Assign Bob = ? (MRV selects Bob, domain = {2, 3})
→ Try Bob = 2
→ Forward check on constraints...
→ Success! Continue...

Solution: {Alice: 1, Bob: 2, Carol: 3, ...}
```

## Project Structure

```md
MainProject/
├── solver.py                      # Main solver implementation
│   ├── CSPSolver class            # Backtracking + MRV + Forward Checking + AC-3
│   ├── Parser functions           # Natural language → CSP conversion
│   ├── Evaluation functions       # Dataset loading and accuracy metrics
│   └── Main execution logic       # Command-line interface
│
├── README.md                      # This file
├── submission.csv                 # Test_100 output (100 test puzzles)
├── submissionGridmode.csv         # Gridmode output (1000 puzzles)
│
├── Test_100_Puzzles.parquet       # Test dataset (100 puzzles)
├── Gridmode-00000-of-00001.parquet # Gridmode dataset
└── mc-00000-of-00001.parquet      # mc dataset
```

### Key Components in `solver.py`

| Component | Description | Lines |
| --------- | ----------- | ----- |
| `CSPSolver` | Core backtracking solver with MRV and AC-3 | 34-510 |
| `puzzle_text_to_csp()` | Parses natural language puzzles | 1005-1028 |
| `parse_description()` | Extracts puzzle structure | 640-741 |
| `parse_single_clue()` | Converts clues to constraints | 799-959 |
| `solve_puzzle()` | High-level solving function | 528-540 |
| `run_evaluation()` | Batch evaluation on datasets | 1328-1376 |
| `main()` | CLI entry point | 1514-1596 |

## Usage Examples

### Example 1: Simple 3x2 Puzzle

```python
from solver import puzzle_text_to_csp, solve_puzzle

puzzle = """
There are 3 houses.
- Name: Alice, Bob, Carol
- Pet: cat, dog, fish

Clues:
1. Alice is in the first house.
2. The person with the cat is in house 2.
3. Bob is next to the person with the fish.
"""

csp = puzzle_text_to_csp(puzzle)
solution, stats = solve_puzzle(csp)

# Output: {1: {'name': 'Alice', 'pet': 'fish'}, 
#          2: {'name': 'Bob', 'pet': 'cat'}, 
#          3: {'name': 'Carol', 'pet': 'dog'}}
```

### Example 2: Batch Processing

```python
import pandas as pd
from solver import run_evaluation, normalize_dataset

# Load dataset
df = pd.read_parquet("Test_100_Puzzles.parquet")
df = normalize_dataset(df)

# Run evaluation
results = run_evaluation(df, max_puzzles=10, verbose=True)

# Check results
solved = sum(1 for r in results if r.solved)
print(f"Solved: {solved}/10")
```

### Example 3: Single Puzzle with Details

```python
from solver import solve_single_puzzle

result = solve_single_puzzle(
    puzzle_id="custom-001",
    puzzle_text=puzzle,
    expected_solution=None
)

print(f"Solved: {result.solved}")
print(f"Time: {result.time_seconds:.2f}s")
print(f"Nodes explored: {result.stats.nodes_explored}")
print(f"Backtracks: {result.stats.backtracks}")
print(f"Solution: {result.solution}")
```

## Output Format

### submission.csv

Each row contains:

| Column | Description | Example |
| ------ | ----------- | ------- |
| `id` | Puzzle identifier | `test-3x3-001` |
| `grid_solution` | JSON solution grid | `{"header": [...], "rows": [...]}` |
| `steps` | Number of nodes explored | `10` |

**Sample Output:**

```csv
id,grid_solution,steps
test-3x3-001,"{""header"": [""House"", ""Color"", ""Name"", ""Pet""], 
""rows"": [[""1"", ""orange"", ""Bob"", ""turtle""], 
[""2"", ""blue"", ""Mallory"", ""cat""], 
[""3"", ""green"", ""Alice"", ""dog""]]}",
10
```

### Empty Solutions

Unsolved puzzles output empty JSON:

```csv
test-3x3-001,'{}',0
```

## Performance

### Test Dataset Results (100 Puzzles)

| Metric | Value |
| ------ | ----- |
| **Total Puzzles** | 100 |
| **Solved** | 39 |
| **Accuracy** | 39% |
| **Avg. Steps (Solved)** | 10.2 |
| **Max Steps** | 14 |
| **Avg. Time** | 0.0002s |

## Competition Results

### AI Connect 2025 - CSP Solver Challenge

### Composite Score Formula

```md
Composite Score = Accuracy (%) - α × (AvgSteps / MaxAvgSteps)
```

Where:

- **Accuracy**: % of puzzles solved correctly
- **AvgSteps**: Average search steps per puzzle
- **MaxAvgSteps**: Maximum average across all teams
- **α = 10**: Efficiency penalty weight

### Our Results

| Dataset     | Accuracy | Avg Steps | Composite Score |
|---------    |----------|-----------|-----------------|
| Test (100)  | 23%      | 9.3       | ~22.1           |
| Grid Mode   | 84%      | 32.1      | ~81.5           |

**Key Insights:**

- Strong performance on smaller puzzles (2x2 to 4x4)
- Parser handles diverse clue formulations
- AC-3 significantly reduces search space
- Room for improvement on 5x5+ puzzles
