from pathlib import Path
from collections import defaultdict
import math

import pandas as pd


class UserBasedCF:
    def __init__(self, ratings_df: pd.DataFrame, k: int = 20):
        self.k = k
        self.ratings_df = ratings_df

        self.user_item_ratings = defaultdict(dict)
        self.item_user_ratings = defaultdict(dict)

        for row in ratings_df.itertuples(index=False):
            u = int(row.userId)
            i = int(row.movieId)
            r = float(row.rating)
            self.user_item_ratings[u][i] = r
            self.item_user_ratings[i][u] = r

        self.global_mean = ratings_df["rating"].mean()

    def cosine_similarity(self, u1: int, u2: int) -> float:
        items1 = self.user_item_ratings[u1]
        items2 = self.user_item_ratings[u2]

        common_items = set(items1.keys()) & set(items2.keys())
        if not common_items:
            return 0.0

        num = 0.0
        denom1 = 0.0
        denom2 = 0.0

        for i in common_items:
            r1 = items1[i]
            r2 = items2[i]
            num += r1 * r2
            denom1 += r1 * r1
            denom2 += r2 * r2

        if denom1 == 0.0 or denom2 == 0.0:
            return 0.0

        return num / math.sqrt(denom1 * denom2)

    def predict_rating(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_item_ratings:
            return self.global_mean

        if item_id not in self.item_user_ratings:
            user_ratings = self.user_item_ratings[user_id]
            if user_ratings:
                return sum(user_ratings.values()) / len(user_ratings)
            else:
                return self.global_mean

        neighbors = self.item_user_ratings[item_id]

        sims = []
        for neighbor_id in neighbors.keys():
            if neighbor_id == user_id:
                continue
            sim = self.cosine_similarity(user_id, neighbor_id)
            if sim > 0:
                sims.append((neighbor_id, sim))

        if not sims:
            user_ratings = self.user_item_ratings[user_id]
            if user_ratings:
                return sum(user_ratings.values()) / len(user_ratings)
            else:
                return self.global_mean

        sims.sort(key=lambda x: x[1], reverse=True)
        top_k = sims[: self.k]

        num = 0.0
        denom = 0.0
        for neighbor_id, sim in top_k:
            r_vi = self.user_item_ratings[neighbor_id].get(item_id)
            if r_vi is None:
                continue
            num += sim * r_vi
            denom += sim

        if denom == 0.0:
            user_ratings = self.user_item_ratings[user_id]
            if user_ratings:
                return sum(user_ratings.values()) / len(user_ratings)
            else:
                return self.global_mean

        return num / denom

    def top_k_neighbors_for_user(self, user_id: int, k: int = 10):
        if user_id not in self.user_item_ratings:
            return []

        sims = []
        for other_id in self.user_item_ratings.keys():
            if other_id == user_id:
                continue
            sim = self.cosine_similarity(user_id, other_id)
            if sim > 0:
                sims.append((other_id, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def recommend_for_user(self, user_id: int, all_item_ids, top_n: int = 10):
        if user_id not in self.user_item_ratings:
            return []

        rated_items = set(self.user_item_ratings[user_id].keys())
        candidates = [i for i in all_item_ids if i not in rated_items]

        scored = []
        for item_id in candidates:
            pred = self.predict_rating(user_id, item_id)
            scored.append((item_id, pred))

        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_n]

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

    print(f"Loading train ratings from {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Train size: {len(train_df)}")

    print(f"Loading test ratings from {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test size: {len(test_df)}")

    print("Building UserBasedCF model...")
    model = UserBasedCF(train_df, k=20)

    print("Evaluating on test set (sample)...")
    rmse = model.evaluate_rmse(test_df, max_samples=20000)
    print(f"User-based CF RMSE (sampled): {rmse:.4f}")


if __name__ == "__main__":
    main()
