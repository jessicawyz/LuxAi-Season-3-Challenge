import os
import glob
import pandas as pd
from pathlib import Path

WORKING_FOLDER = Path(__file__).parent
SUBMISSIONS_PATH = f"{WORKING_FOLDER}/submissions.csv"
GAMES_PATH = f"{WORKING_FOLDER}/games.csv"
EPISODES_DIR = f"{WORKING_FOLDER}/episodes"


def update_data():
    submission_id_to_name = {}
    if os.path.exists(SUBMISSIONS_PATH):
        old_submissions_df = pd.read_csv(SUBMISSIONS_PATH)
        submission_id_to_name = dict(
            zip(old_submissions_df["submission_id"], old_submissions_df["name"])
        )

    downloaded_episodes = {
        int(os.path.basename(x).split(".")[0])
        for x in glob.glob(f"{EPISODES_DIR}/*.json")
    }

    games = pd.read_csv(GAMES_PATH)
    print(f"Num games {len(games)}")

    submissions = []
    for submission_id, sub_df in games.groupby("SubmissionId"):
        sub_df = sub_df.sort_values("EpisodeId", ascending=True)

        first_game = sub_df.iloc[0]
        last_game = sub_df.iloc[-1]

        episode_ids = sub_df["EpisodeId"].unique()
        num_games = len(episode_ids)
        num_downloaded = sum(
            episode_id in downloaded_episodes for episode_id in episode_ids
        )

        score = last_game["UpdatedScore"]
        created_date = first_game["CreateTime"]
        updated_date = last_game["CreateTime"]

        submissions.append(
            {
                "submission_id": submission_id,
                "name": submission_id_to_name.get(submission_id),
                "score": score,
                "num_games": num_games,
                "num_downloaded": num_downloaded,
                "created_date": created_date,
                "updated_date": updated_date,
            }
        )

    print(f"Num submissions {len(submissions)}")

    submissions_df = pd.DataFrame(submissions)
    submissions_df.sort_values("score", ascending=False, inplace=True)
    submissions_df.to_csv(SUBMISSIONS_PATH, index=False)

    print(f"Save as {SUBMISSIONS_PATH}")


if __name__ == "__main__":
    update_data()
