# update names column in submissions.csv

import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

WORKING_FOLDER = Path(__file__).parent
EPISODE_DIR = f"{WORKING_FOLDER}/episodes"
SUBMISSIONS_PATH = f"{WORKING_FOLDER}/submissions.csv"
GAMES_PATH = f"{WORKING_FOLDER}/games.csv"


def update_submissions_names(submission_to_name):
    df = pd.read_csv(SUBMISSIONS_PATH)
    names = []
    for submission_id, name in zip(df["submission_id"], df["name"]):
        name = submission_to_name.get(submission_id, name)
        names.append(name)
    df["name"] = names
    df.to_csv(SUBMISSIONS_PATH, index=False)


def update_names():
    games = pd.read_csv(GAMES_PATH)
    names = {}
    for episode_id, submission_id in tqdm(
        zip(games["EpisodeId"], games["SubmissionId"]), total=len(games)
    ):
        if submission_id in names:
            continue

        path = os.path.join(EPISODE_DIR, f"{episode_id}.json")
        if not os.path.exists(path):
            continue

        episode = json.load(open(path, "r"))
        agents = episode["metadata"]["agents"]

        for agent in agents:
            names[agent["submission_id"]] = agent["name"]

    update_submissions_names(names)


if __name__ == "__main__":
    update_names()
