# Should be run in kaggle notebook with meta-kaggle dataset

from pathlib import Path
import datetime
import collections
import polars as pl

META_DIR = Path("../input/meta-kaggle/")
COMPETITION_ID = 86411  # lux-ai-s3

episodes_df = pl.scan_csv(META_DIR / "Episodes.csv")
episodes_df = (
    episodes_df.filter(pl.col("CompetitionId") == COMPETITION_ID)
    .with_columns(
        pl.col("CreateTime").str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False),
        pl.col("EndTime").str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False),
    )
    .sort("Id")
    .collect()
)

episodes_df = episodes_df.filter(pl.col("CreateTime") > datetime.datetime(2025, 1, 30))

print(f"Episodes.csv: {len(episodes_df)} rows.")

agents_df = pl.scan_csv(
    META_DIR / "EpisodeAgents.csv",
    schema_overrides={
        "Reward": pl.Float32,
        "UpdatedConfidence": pl.Float32,
        "UpdatedScore": pl.Float32,
    },
)

agents_df = (
    agents_df.filter(pl.col("EpisodeId").is_in(episodes_df["Id"].to_list()))
    .with_columns(
        [
            pl.when(pl.col("InitialConfidence") == "")
            .then(None)
            .otherwise(pl.col("InitialConfidence"))
            .cast(pl.Float64)
            .alias("InitialConfidence"),
            pl.when(pl.col("InitialScore") == "")
            .then(None)
            .otherwise(pl.col("InitialScore"))
            .cast(pl.Float64)
            .alias("InitialScore"),
        ]
    )
    .collect()
)
print(f"EpisodeAgents.csv: {len(agents_df)} rows.")

games_df = agents_df.join(episodes_df, left_on="EpisodeId", right_on="Id").select(
    ["EpisodeId", "Index", "Reward", "SubmissionId", "UpdatedScore", "CreateTime"]
)

episode_id_to_sids = collections.defaultdict(lambda: [None, None])
for episode_id, index, sid in zip(
    games_df["EpisodeId"], games_df["Index"], games_df["SubmissionId"]
):
    episode_id_to_sids[episode_id][index] = sid

opp_sids = []
for episode_id, index in zip(games_df["EpisodeId"], games_df["Index"]):
    opp_sid = episode_id_to_sids[episode_id][1 - index]
    opp_sids.append(opp_sid)

games_df = games_df.with_columns(pl.Series(name="OppSubmissionId", values=opp_sids))

games_df.write_csv("games.csv")
