#!/usr/bin/env python3
import sys
from itertools import combinations

def emit_pairs(movie_id, user_ratings):
    for (u1, r1), (u2, r2) in combinations(user_ratings, 2):
        if u1 > u2:
            u1, u2 = u2, u1
            r1, r2 = r2, r1
        print(f"{u1},{u2}\t{r1}:{r2}")

def main():
    current_movie = None
    user_ratings = []

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        movie_id, value = line.split("\t", 1)
        user_id, rating_str = value.split(":", 1)

        if current_movie is None:
            current_movie = movie_id

        if movie_id != current_movie:
            if user_ratings:
                emit_pairs(current_movie, user_ratings)
            current_movie = movie_id
            user_ratings = []

        user_ratings.append((user_id, float(rating_str)))

    if current_movie is not None and user_ratings:
        emit_pairs(current_movie, user_ratings)

if __name__ == "__main__":
    main()
