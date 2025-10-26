import os
import json
import tyro
import requests
import pandas as pd
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

BASE_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/"
GET_URL = BASE_URL + "GetEpisodeReplay"
WORKING_FOLDER = Path(__file__).parent
OUTPUT_DIR = f"{WORKING_FOLDER}/episodes"
SUBMISSIONS_PATH = f"{WORKING_FOLDER}/submissions.csv"
GAMES_PATH = f"{WORKING_FOLDER}/games.csv"


def get_submissions_ids(episode_id, games_df=None):
    if games_df is None:
        games_df = pd.read_csv(GAMES_PATH)
    df = games_df[games_df["EpisodeId"] == episode_id].sort_values(
        "Index", ascending=True
    )
    return [int(x) for x in df["SubmissionId"]]


def update_submissions_names(submission_to_name):
    df = pd.read_csv(SUBMISSIONS_PATH)
    names = []
    for submission_id, name in zip(df["submission_id"], df["name"]):
        name = submission_to_name.get(submission_id, name)
        names.append(name)
    df["name"] = names
    df.to_csv(SUBMISSIONS_PATH, index=False)


def get_actions(keggle_replay):
    actions = []
    for step in keggle_replay["steps"][1:]:
        actions.append(step[0]["info"]["replay"]["actions"][0])
    return actions


def get_observations(keggle_replay):
    observations = []
    for step in keggle_replay["steps"]:
        observations.append(step[0]["info"]["replay"]["observations"][0])
    return observations


def get_relic_info(params, observations):
    map_size = params["map_width"]
    relic_config_size = params["relic_config_size"]
    relics = observations[-1]["relic_nodes"]
    relic_configs = observations[-1]["relic_node_configs"]

    rewards = set()
    for (relic_x, relic_y), config in zip(relics, relic_configs):
        for ix, column in enumerate(config):
            for iy, reward in enumerate(column):
                if not reward:
                    continue

                reward_x = relic_x - relic_config_size // 2 + ix
                if reward_x < 0 or reward_x >= map_size:
                    continue

                reward_y = relic_y - relic_config_size // 2 + iy
                if reward_y < 0 or reward_y >= map_size:
                    continue

                rewards.add((reward_x, reward_y))

    return len(relics), len(rewards)


def get_episode(episode_id, games_df=None):
    re = requests.post(GET_URL, json={"episodeId": int(episode_id)})
    keggle_replay = re.json()

    # path = os.path.join(OUTPUT_DIR, f"{episode_id}.json")
    # with open(path, 'r') as f:
    #     keggle_replay = json.load(f)

    assert episode_id == keggle_replay["info"]["EpisodeId"]

    params = keggle_replay["steps"][0][0]["info"]["replay"]["params"]
    seed = keggle_replay["steps"][0][0]["info"]["replay"]["metadata"]["seed"]
    players = keggle_replay["info"]["TeamNames"]
    submissions = get_submissions_ids(episode_id, games_df)
    actions = get_actions(keggle_replay)
    observations = get_observations(keggle_replay)
    rewards = observations[-1]["team_wins"]
    params["num_relics"], params["num_rewards"] = get_relic_info(params, observations)

    update_submissions_names(dict(zip(submissions, players)))

    data = {
        "metadata": {
            "seed": seed,
            "players": {"player_0": players[0], "player_1": players[1]},
            "episode_id": episode_id,
            "agents": [
                {
                    "name": players[0],
                    "reward": rewards[0],
                    "submission_id": submissions[0],
                },
                {
                    "name": players[1],
                    "reward": rewards[1],
                    "submission_id": submissions[1],
                },
            ],
        },
        "params": params,
        "actions": actions,
        "observations": observations,
    }

    path = os.path.join(OUTPUT_DIR, f"{episode_id}.json")
    json.dump(data, open(path, "w"))


def get_episodes(submission_id, num_episodes=1000, min_score=None):
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    games = pd.read_csv(GAMES_PATH)

    if min_score is not None:
        submissions = pd.read_csv(SUBMISSIONS_PATH)
        sid_id_to_score = dict(zip(submissions["submission_id"], submissions["score"]))
        opp_scores = [sid_id_to_score[x] for x in games["OppSubmissionId"]]
        games["opp_score"] = opp_scores
        games = games[games["opp_score"] >= min_score]

    episodes = set(games[games["SubmissionId"] == submission_id]["EpisodeId"])
    episodes_to_download = []
    for episode_id in episodes:
        path = os.path.join(OUTPUT_DIR, f"{episode_id}.json")
        if os.path.exists(path):
            continue
        episodes_to_download.append(episode_id)
    print(
        f"Episodes total: {len(episodes)}, episodes to download {len(episodes_to_download)}"
    )

    for i, episode_id in enumerate(episodes_to_download[:num_episodes], start=1):
        print(f"request episode {episode_id}: {i}/{len(episodes_to_download)}")
        get_episode(episode_id, games_df=games)


@dataclass
class Args:
    # submission to download
    submission_id: int

    # number of episodes to download
    num_episodes: Optional[int] = 1000

    # minimum score for an opponent
    min_score: Optional[int] = None


if __name__ == "__main__":
    args = tyro.cli(Args)
    get_episodes(args.submission_id, args.num_episodes, args.min_score)
