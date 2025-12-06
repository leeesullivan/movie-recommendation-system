from pathlib import Path
from collections import defaultdict
import math

import pandas as pd


class ItemBasedCF:
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                 k: int = 20, alpha: float = 0.7):
        self.k = k
        self.alpha = alpha
        self.ratings_df = ratings_df
        self.movies_df = movies_df

        self.user_item_ratings = defaultdict(dict)
        self.item_user_ratings = defaultdict(dict)

        for row in ratings_df.itertuples(index=False):
            u = int(row.userId)
            i = int(row.movieId)
            r = float(row.rating)
            self.user_item_ratings[u][i] = r
            self.item_user_ratings[i][u] = r

        self.global_mean = ratings_df["rating"].mean()

        self.item_genres = {}
        for row in movies_df.itertuples(index=False):
            i = int(row.movieId)
            if isinstance(row.genres, str):
                genres = set(row.genres.split("|"))
            else:
                genres = set()
            self.item_genres[i] = genres

    def rating_similarity_items(self, i1: int, i2: int) -> float:
        users1 = self.item_user_ratings.get(i1)
        users2 = self.item_user_ratings.get(i2)
        if not users1 or not users2:
            return 0.0

        common_users = set(users1.keys()) & set(users2.keys())
        if not common_users:
            return 0.0

        num = 0.0
        denom1 = 0.0
        denom2 = 0.0

        for u in common_users:
            r1 = users1[u]
            r2 = users2[u]
            num += r1 * r2
            denom1 += r1 * r1
            denom2 += r2 * r2

        if denom1 == 0.0 or denom2 == 0.0:
            return 0.0

        return num / math.sqrt(denom1 * denom2)

    def genre_similarity_items(self, i1: int, i2: int) -> float:
        g1 = self.item_genres.get(i1, set())
        g2 = self.item_genres.get(i2, set())

        if not g1 or not g2:
            return 0.0

        inter = len(g1 & g2)
        union = len(g1 | g2)
        if union == 0:
            return 0.0
        return inter / union

    def combined_similarity_items(self, i1: int, i2: int) -> float:
        sim_r = self.rating_similarity_items(i1, i2)
        sim_g = self.genre_similarity_items(i1, i2)
        return self.alpha * sim_r + (1.0 - self.alpha) * sim_g

    def predict_rating(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_item_ratings:
            return self.global_mean

        user_ratings = self.user_item_ratings[user_id]

        if not user_ratings:
            return self.global_mean

        if item_id not in self.item_user_ratings:
            return sum(user_ratings.values()) / len(user_ratings)

        candidate_items = user_ratings.keys()

        sims = []
        for j in candidate_items:
            if j == item_id:
                continue
            sim = self.combined_similarity_items(item_id, j)
            if sim > 0:
                sims.append((j, sim))

        if not sims:
            return sum(user_ratings.values()) / len(user_ratings)

        sims.sort(key=lambda x: x[1], reverse=True)
        top_k = sims[: self.k]

        num = 0.0
        denom = 0.0
        for j, sim in top_k:
            r_uj = user_ratings.get(j)
            if r_uj is None:
                continue
            num += sim * r_uj
            denom += sim

        if denom == 0.0:
            return sum(user_ratings.values()) / len(user_ratings)

        return num / denom

    def evaluate_rmse(self, test_df: pd.DataFrame, max_samples: int = 20000) -> float:
        if len(test_df) > max_samples:
            test_sample = test_df.sample(n=max_samples, random_state=42)
        else:
            test_sample = test_df

        se_sum = 0.0
        n = 0

        for row in test_sample.itertuples(index=False):
            u = int(row.userId)
            i = int(row.movieId)
            true_r = float(row.rating)

            pred_r = self.predict_rating(u, i)

            se = (pred_r - true_r) ** 2
            se_sum += se
            n += 1

        if n == 0:
            return float("nan")

        rmse = math.sqrt(se_sum / n)
        return rmse


def main():
    base_dir = Path(__file__).resolve().parents[2]
    processed_dir = base_dir / "data" / "processed"

    train_path = processed_dir / "ratings_train.csv"
    test_path = processed_dir / "ratings_test.csv"
    movies_path = processed_dir / "movies.csv"

    print(f"Loading train ratings from {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Train size: {len(train_df)}")

    print(f"Loading test ratings from {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test size: {len(test_df)}")

    print(f"Loading movies from {movies_path}")
    movies_df = pd.read_csv(movies_path)
    print(f"Movies: {len(movies_df)}")

    print("Building ItemBasedCF model (hybrid ratings + genres)...")
    model = ItemBasedCF(train_df, movies_df, k=20, alpha=0.7)

    print("Evaluating on test set (sample)...")
    rmse = model.evaluate_rmse(test_df, max_samples=20000)
    print(f"Hybrid Item-based CF RMSE (sampled): {rmse:.4f}")


if __name__ == "__main__":
    main()
