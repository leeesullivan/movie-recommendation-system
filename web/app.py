from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

from recommender.user_based_cf import UserBasedCF

app = Flask(__name__)
app.secret_key = "change-this-in-real-life"  # needed for sessions


# ---------- Load data once at startup ----------

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ratings = pd.read_csv(PROCESSED_DIR / "ratings_train.csv")
movies = pd.read_csv(PROCESSED_DIR / "movies.csv")

ratings_with_titles = ratings.merge(movies, on="movieId", how="left")
print("Building UserBasedCF model for web app...")
user_cf_model = UserBasedCF(ratings, k=20)

all_movie_ids = movies["movieId"].tolist()


@app.route("/", methods=["GET", "POST"])
def login():
    """
    Simple login: user selects a userId from a dropdown or types it.
    """
    user_ids = sorted(ratings["userId"].unique())

    if request.method == "POST":
        user_id = request.form.get("user_id")
        if user_id:
            session["user_id"] = int(user_id)
            return redirect(url_for("dashboard"))

    return render_template("login.html", user_ids=user_ids)


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    """
    Shows the logged-in user's rated movies.
    """
    user_id = session.get("user_id")
    if user_id is None:
        return redirect(url_for("login"))

    user_ratings = ratings_with_titles[ratings_with_titles["userId"] == user_id]

    user_ratings = user_ratings.sort_values(
        by=["rating", "timestamp"], ascending=[False, False]
    )

    return render_template(
        "dashboard.html",
        user_id=user_id,
        user_ratings=user_ratings.to_dict(orient="records"),
    )

@app.route("/recommendations")
def recommendations():
    user_id = session.get("user_id")
    if user_id is None:
        return redirect(url_for("login"))

    top_n = 10
    recs = user_cf_model.recommend_for_user(user_id, all_movie_ids, top_n=top_n)

    rec_rows = []
    for movie_id, pred in recs:
        movie_row = movies[movies["movieId"] == movie_id]
        if not movie_row.empty:
            title = movie_row.iloc[0]["title"]
            genres = movie_row.iloc[0]["genres"]
        else:
            title = f"Movie {movie_id}"
            genres = ""
        rec_rows.append({
            "movieId": movie_id,
            "title": title,
            "genres": genres,
            "predicted_rating": pred
        })

    return render_template(
        "recommendations.html",
        user_id=user_id,
        recommendations=rec_rows,
    )

@app.route("/similar_users")
def similar_users():
    user_id = session.get("user_id")
    if user_id is None:
        return redirect(url_for("login"))

    neighbors = user_cf_model.top_k_neighbors_for_user(user_id, k=10)

    neighbor_rows = []
    for neighbor_id, sim in neighbors:
        movies_u = set(user_cf_model.user_item_ratings[user_id].keys())
        movies_v = set(user_cf_model.user_item_ratings[neighbor_id].keys())
        overlap = len(movies_u & movies_v)

        neighbor_rows.append({
            "neighbor_id": neighbor_id,
            "similarity": sim,
            "overlap": overlap,
        })

    return render_template(
        "similar_users.html",
        user_id=user_id,
        neighbors=neighbor_rows,
    )


if __name__ == "__main__":
    app.run(debug=True)
