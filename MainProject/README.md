# Explanation of approach

## CSP Solver

The CSPs will be modeled in the Following way:

Every possible value of each variable is associated with a number.  
The numbers are in a range from 1 to n, where n is the number of entities present in the puzzle.  
That means if a puzzle describes 3 homes and their inhabitant, then n is equal to 3.  
Every variable must only have as many possible values as there is entities in the puzzle, and each possible value must have a distinct number.  

To determine the solutions after the complete calculation of the solution, you iterate over every variable and check its domain. You then determine the value with the fitting number and add it as the solution for your current entity.

## Data Parser

The Data Parser will be called on the string containing the puzzles tips.

## CSP Object

- number of entities
- variables (list)
  - name
- domains (list)
  - reference to the variable
  - list of all attributes <- Dictionary
    - number assigned to the attribute
- constrains (list)
  - expression
    - dictionary
    - Contents/structure of dictionary tbd
