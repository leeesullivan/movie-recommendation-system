#!/usr/bin/env python3
import sys
from itertools import combinations

def emit_pairs(user_id, movie_ratings):
    for (m1, r1), (m2, r2) in combinations(movie_ratings, 2):
        if m1 > m2:
            m1, m2 = m2, m1
            r1, r2 = r2, r1
        print(f"{m1},{m2}\t{r1}:{r2}")

def main():
    current_user = None
    movie_ratings = []

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        user_id, value = line.split("\t", 1)
        movie_id, rating_str = value.split(":", 1)

        if current_user is None:
            current_user = user_id

        if user_id != current_user:
            if movie_ratings:
                emit_pairs(current_user, movie_ratings)
            current_user = user_id
            movie_ratings = []

        movie_ratings.append((movie_id, float(rating_str)))

    if current_user is not None and movie_ratings:
        emit_pairs(current_user, movie_ratings)

if __name__ == "__main__":
    main()
