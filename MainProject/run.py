"""
Main runner script for the Zebra Logic Puzzle CSP Solver.

This script:
1. Loads puzzles from the ZebraLogicBench dataset (parquet format)
2. Parses each puzzle into CSP format
3. Solves using backtracking with MRV and forward checking
4. Evaluates accuracy and efficiency
5. Outputs results and statistics
"""

import pandas as pd
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

# Import our modules
from dataParserWIP import puzzle_text_to_csp
from solver import solve_puzzle, SolverStats


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


def run_evaluation(df: pd.DataFrame, max_puzzles: int = None, 
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
    
    puzzles_to_solve = df.head(max_puzzles) if max_puzzles else df
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
        
        print(f"\n{'='*60}")
        print("SOLUTION")
        print(f"{'='*60}")
        print_solution(result.solution)
        
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


if __name__ == "__main__":
    main()
