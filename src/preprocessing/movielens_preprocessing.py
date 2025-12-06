from pathlib import Path
import pandas as pd


def load_ratings():
    base_dir = Path(__file__).resolve().parents[2]
    raw_dir = base_dir / "data" / "ml-1m"

    ratings_path = raw_dir / "ratings.dat"

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1",
    )

    return ratings

def load_movies():
    base_dir = Path(__file__).resolve().parents[2]
    raw_dir = base_dir / "data" / "ml-1m"

    movies_path = raw_dir / "movies.dat"

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1",
    )

    return movies


def train_test_split_ratings(ratings, test_fraction=0.2, random_state=42):
    ratings = ratings.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    n_total = len(ratings)
    n_test = int(n_total * test_fraction)

    test = ratings.iloc[:n_test].copy()
    train = ratings.iloc[n_test:].copy()

    return train, test


def main():
    base_dir = Path(__file__).resolve().parents[2]
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Ratings
    print("Loading ratings...")
    ratings = load_ratings()
    print(f"Total ratings: {len(ratings)}")

    print("Splitting into train and test...")
    train, test = train_test_split_ratings(ratings, test_fraction=0.2)

    train_path = processed_dir / "ratings_train.csv"
    test_path = processed_dir / "ratings_test.csv"

    print(f"Saving train to: {train_path}")
    train.to_csv(train_path, index=False)

    print(f"Saving test to:  {test_path}")
    test.to_csv(test_path, index=False)

    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # Movies
    print("Loading movies...")
    movies = load_movies()
    print(f"Total movies: {len(movies)}")

    movies_path = processed_dir / "movies.csv"
    print(f"Saving movies to: {movies_path}")
    movies.to_csv(movies_path, index=False)

    print("Done preprocessing MovieLens 1M!")


if __name__ == "__main__":
    main()
