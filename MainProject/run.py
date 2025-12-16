import dataParser, solver

import pandas as pd
import csv

# Constants

FILENAME = "Gridmode-00000-of-00001.parquet"

def run():

    df, csps = dataParser.run(FILENAME)
    solutions = solver.solve(df, csps)

    with open("results.csv", 'w') as csvfile:
        for row in solutions:
            writer = csv.writer(csvfile, quotechar='"')
            writer.writerow(row)

if __name__ == "__main__":
    run()
