#!/usr/bin/env python3
import sys
import csv

def main():
    reader = csv.reader(sys.stdin)
    header_skipped = False

    for row in reader:
        if not header_skipped:
            header_skipped = True
            if row and row[0] == "userId":
                continue

        if len(row) < 3:
            continue

        user_id = row[0]
        movie_id = row[1]
        rating = row[2]

        print(f"{user_id}\t{movie_id}:{rating}")

if __name__ == "__main__":
    main()
