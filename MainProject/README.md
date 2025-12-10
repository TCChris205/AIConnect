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

### Regex Clues to identify bits of data

#### Number of Entities and Attributes

- Check the "size" string in the csv entry.
- Take Numbers until '\*'-> Number of Entities
- Skip '\*'
- Take Numbers until End of String -> Number of Attributes

#### Attribute Names and Domains

- Getting the Attributes
  - Check "solution" dict in the csv entry.
  - Get the list that is the first element in the header section
    - contains all Attribute Names that should be used in the end
  - Ignore the first entry, as that is always the House number and can be ignored in our computation approach (will obviously be added back later)
- Getting the Domains
  - Check "puzzle" string in the csv entry.
  - Skip until first ':'
    - For each Following '-'
      - Determine the current Attribute List member
      - Skip until the ':'
      - Add the text elements inside single quotation marks into a list
      - define that list as the domain of the determined attribute
        - Check if number of domain elements fits to the number of entities
      - move the "List Pointer" to the next entry and start the loop again.

#### Constrains

- Check "puzzle" dict in the csv entry.
- Skip until '##'
- Each sequence of numbers followed by a dot represents the start of a constraint.

## CSP Object

- number of entities
- variables and domains (dict)
  - name of the variable
  - list of all domains
  - [variableName : (domain in form of list)]
- constrains (list)
  - List of Tupels
  - First Value: Type of constraint
    - '=' -> attr1 == attr2 OR attr1 = num, depending on third value
    - '!=' -> attr1 != attr2 OR attr1 != num, depending on third value
    - '>' -> attr1 > attr2
    - '<' -> attr1 < attr2
    - '+[number]' -> attr1 == attr2+[number]
    - '-[number]' -> attr1 == attr2-[number]
    - '+-[number]' -> attr1 == attr2+[number] OR attr1 == attr2-[number], either is fine
  - Second Value
    - attribute (attr1)
      - First attribute or attribute that something has to be assigned to
  - Third Value
    - attribute (attr2) or number (num)
      - number has to be assigned to attr1
