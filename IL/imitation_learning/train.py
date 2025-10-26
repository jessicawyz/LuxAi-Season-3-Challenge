# import os
# import time
# import torch
# import random
# import numpy as np
# import pandas as pd
# from torch import nn
# from tqdm import tqdm
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# from agent.base import SPACE_SIZE, transpose
# from agent.path import ActionType

# AGENT_EPISODES_DIR = "imitation_learning/dataset/agent_episodes"
# MODEL_NAME = "imitation_learning/unit_unet"

# N_CHANNELS = 28
# N_GLOBAL = 17
# N_CLASSES = 6

# device = torch.device("cpu")

# def seed_everything(seed_value):
#     random.seed(seed_value)
#     np.random.seed(seed_value)
#     torch.manual_seed(seed_value)
#     os.environ["PYTHONHASHSEED"] = str(seed_value)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed_value)
#         torch.cuda.manual_seed_all(seed_value)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = True


# def select_episodes(submission_ids, min_opp_score, val_ratio=0.05, num_episodes=None):
#     seed_everything(42)

#     submissions_df = pd.read_csv("imitation_learning/dataset/submissions.csv")
#     sid_to_score = dict(zip(submissions_df["submission_id"], submissions_df["score"]))

#     games_df = pd.read_csv("imitation_learning/dataset/games.csv")
#     games_df["opp_score"] = [sid_to_score[x] for x in games_df["OppSubmissionId"]]
#     games_df = games_df[
#         games_df["SubmissionId"].isin(submission_ids)
#         & (games_df["opp_score"] >= min_opp_score)
#     ]

#     episodes = set()
#     for sid, episode_id in zip(games_df["SubmissionId"], games_df["EpisodeId"]):
#         path = f"{AGENT_EPISODES_DIR}/{sid}_{episode_id}.npz"
#         if os.path.exists(path):
#             episodes.add(path)

#     episodes = sorted(episodes)
#     if num_episodes is not None:
#         episodes = episodes[:num_episodes]

#     print(f"total number of episodes: {len(episodes)}")

#     random.shuffle(episodes)
#     num_train = int(len(episodes) * (1 - val_ratio))

#     train_episodes, val_episodes = episodes[:num_train], episodes[num_train:]

#     print(f"train size: {len(train_episodes)}")
#     print(f"val size: {len(val_episodes)}")

#     return train_episodes, val_episodes


# # ===================#
# #      Dataset       #
# # ===================#


# class LuxDataset(Dataset):

#     def __init__(self, episodes, aug=True):
#         if not episodes:
#             raise ValueError("Can't create datasert without episodes")

#         self.episode_steps = []
#         self.episode_id_to_data = {}
#         for episode_id, episode_path in enumerate(episodes):
#             npz_file = np.load(episode_path)

#             self.episode_id_to_data[episode_id] = {
#                 "states": npz_file["states"],
#                 "gfs": npz_file["gfs"],
#                 "actions": npz_file["actions"],
#             }

#             for j in range(len(npz_file["states"])):
#                 self.episode_steps.append((episode_id, j))

#         self.aug = aug

#     def __len__(self):
#         return len(self.episode_steps)

#     def __getitem__(self, idx):
#         episode_id, step = self.episode_steps[idx]

#         data = self.episode_id_to_data[episode_id]

#         state = data["states"][step]
#         gf_values = data["gfs"][step]
#         actions = data["actions"][step]

#         if self.aug:
#             aug = random.random() > 0.5
#         else:
#             aug = False

#         gf = np.zeros((len(gf_values), 3, 3), dtype=np.float32)
#         for i, x in enumerate(gf_values):
#             gf[i] = x

#         label = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int64)
#         for x, y, action_type in actions:
#             if x == -1 or y == -1:
#                 break
#             if aug:
#                 action_type = ActionType(action_type).transpose()
#             label[y, x] = action_type

#         if aug:
#             state = transpose(state)
#             label = transpose(label)
#             gf[0] *= -1  # nebula_tile_drift_direction

#         mask = state[25] > 0

#         return state, gf, label, mask


# # ===================#
# #       Model        #
# # ===================#

# # source https://github.com/milesial/Pytorch-UNet/tree/master/unet


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(
#                 in_channels, in_channels // 2, kernel_size=2, stride=2
#             )
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)


# class AddGlobal(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, gf):
#         return torch.cat([x, gf], dim=1)


# class UNet(nn.Module):
#     def __init__(
#         self,
#         n_channels=N_CHANNELS,
#         n_classes=N_CLASSES,
#         n_global=N_GLOBAL,
#         bilinear=True,
#     ):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 256)
#         self.add_global = AddGlobal()
#         self.up1 = Up(512 + n_global, 256, bilinear)
#         self.up2 = Up(256 + 128, 128, bilinear)
#         self.up3 = Up(128 + 64, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x, gf):
#         x1 = self.inc(x)  # 24
#         x2 = self.down1(x1)  # 12
#         x3 = self.down2(x2)  # 6
#         x4 = self.down3(x3)  # 3
#         x5 = self.add_global(x4, gf)
#         x = self.up1(x5, x3)  # 6
#         x = self.up2(x, x2)  # 12
#         x = self.up3(x, x1)  # 24
#         logits = self.outc(x)
#         return logits


# # ===================#
# #       Train        #
# # ===================#


# def masked_loss(predictions, labels, mask, weights):
#     """
#     predictions: (batch_size, num_classes, height, width)
#     labels: Ground truth labels (batch_size, height, width)
#     mask: Binary mask (batch_size, height, width)
#     """
#     loss = F.cross_entropy(predictions, labels, weight=weights, reduction="none")
#     loss = loss * mask
#     loss = loss.sum(dim=[1, 2]) / mask.sum(dim=[1, 2])
#     return loss.mean()


# def get_acc(outs, actions, mask, label_to_acc):
#     preds = torch.argmax(outs, dim=1)
#     total = torch.sum(mask)
#     correct = torch.sum((preds == actions) * mask)

#     for action_type in ActionType:
#         action_type = int(action_type)
#         action_mask = (actions == action_type) * mask

#         action_correct = torch.sum((preds == actions) * action_mask)
#         action_total = torch.sum(action_mask)

#         label_to_acc[action_type][0] += action_correct
#         label_to_acc[action_type][1] += action_total

#     return correct, total


# def train_model(
#     model,
#     train_episodes,
#     val_episodes,
#     optimizer,
#     scheduler,
#     weights,
#     num_epochs,
#     model_name="model",
#     num_episodes_per_epoch=1000,
#     batch_size=128,
# ):

#     best_loss = 10**9

#     val_loader = DataLoader(
#         LuxDataset(val_episodes, aug=False),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#     )

#     for epoch in range(num_epochs):
#         model.cuda()

#         np.random.shuffle(train_episodes)

#         train_dataloader = DataLoader(
#             LuxDataset(train_episodes[:num_episodes_per_epoch]),
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=0,
#         )

#         phases = [("train", train_dataloader)]
#         if (epoch + 1) % 5 == 0:
#             phases.append(("val", val_loader))

#         val_loss = 10**9

#         for phase, dataloader in phases:
#             if phase == "train":
#                 model.train()
#             else:
#                 model.eval()

#             epoch_loss = 0.0
#             epoch_acc = 0
#             correct = 0
#             total = 0

#             label_to_acc = {x: [0, 0] for x in ActionType}

#             for item in tqdm(dataloader, leave=False):
#                 states = item[0].cuda().float()
#                 gf = item[1].cuda().float()
#                 label = item[2].cuda().long()
#                 mask = item[3].cuda().bool()
#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == "train"):
#                     policy = model(states, gf)
#                     loss = masked_loss(policy, label, mask, weights)

#                     if phase == "train":
#                         loss.backward()
#                         optimizer.step()
#                     else:
#                         _correct, _total = get_acc(policy, label, mask, label_to_acc)
#                         correct += _correct
#                         total += _total

#                     epoch_loss += loss.item() * len(policy)

#             data_size = len(dataloader.dataset)
#             epoch_loss = epoch_loss / data_size

#             if phase != "train":
#                 val_loss = epoch_loss
#                 epoch_acc = correct.double() / total

#                 msg = []
#                 for l, (c, t) in label_to_acc.items():
#                     if t > 0:
#                         msg.append(f"{ActionType(l)}: {float(c / t):.2f}")
#                     else:
#                         msg.append(f"{ActionType(l)}: nan")
#                 msg = ", ".join(msg)

#                 print("label to auc", msg)

#                 if scheduler is not None:
#                     scheduler.step(epoch_loss)

#             time.sleep(1)
#             print(
#                 f"Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.5f} | Acc: {epoch_acc:.4f}"
#             )

#         if val_loss < best_loss:
#             traced = torch.jit.trace(
#                 model.cpu(),
#                 example_inputs=(
#                     torch.rand(1, N_CHANNELS, 24, 24),
#                     torch.rand(1, N_GLOBAL, 3, 3),
#                 ),
#             )
#             model_path = f"{model_name}.pth"
#             print(f"Saving model to `{model_path}`.")
#             traced.save(model_path)
#             best_loss = val_loss


# def main(submission_ids, min_opp_score):
#     train_episodes, val_episodes = select_episodes(submission_ids, min_opp_score)

#     seed_everything(42)
#     model = UNet()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, "min", factor=0.1, patience=0, min_lr=1e-6, 
#     )

#     # center, up, right, down, left, sap
#     weights = torch.Tensor([0.3, 1.0, 0.5, 0.5, 1.0, 1.0]).cuda().float()

#     train_model(
#         model,
#         train_episodes,
#         val_episodes,
#         optimizer,
#         scheduler,
#         weights,
#         num_epochs=50,
#         model_name=MODEL_NAME,
#     )

import os
import time
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from agent.base import SPACE_SIZE, transpose
from agent.path import ActionType

AGENT_EPISODES_DIR = "imitation_learning/dataset/agent_episodes"
MODEL_NAME = "imitation_learning/unit_unet"

N_CHANNELS = 28
N_GLOBAL = 17
N_CLASSES = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def select_episodes(submission_ids, min_opp_score, val_ratio=0.05, num_episodes=None):
    seed_everything(42)

    submissions_df = pd.read_csv("imitation_learning/dataset/submissions.csv")
    sid_to_score = dict(zip(submissions_df["submission_id"], submissions_df["score"]))

    games_df = pd.read_csv("imitation_learning/dataset/games.csv")
    games_df["opp_score"] = [sid_to_score[x] for x in games_df["OppSubmissionId"]]
    games_df = games_df[
        games_df["SubmissionId"].isin(submission_ids)
        & (games_df["opp_score"] >= min_opp_score)
    ]

    episodes = set()
    for sid, episode_id in zip(games_df["SubmissionId"], games_df["EpisodeId"]):
        path = f"{AGENT_EPISODES_DIR}/{sid}_{episode_id}.npz"
        if os.path.exists(path):
            episodes.add(path)

    episodes = sorted(episodes)
    if num_episodes is not None:
        episodes = episodes[:num_episodes]

    print(f"total number of episodes: {len(episodes)}")

    random.shuffle(episodes)
    num_train = int(len(episodes) * (1 - val_ratio))

    train_episodes, val_episodes = episodes[:num_train], episodes[num_train:]

    print(f"train size: {len(train_episodes)}")
    print(f"val size: {len(val_episodes)}")

    return train_episodes, val_episodes


# ===================#
#      Dataset       #
# ===================#


class LuxDataset(Dataset):

    def __init__(self, episodes, aug=True):
        if not episodes:
            raise ValueError("Can't create datasert without episodes")

        self.episode_steps = []
        self.episode_id_to_data = {}
        for episode_id, episode_path in enumerate(episodes):
            npz_file = np.load(episode_path)

            self.episode_id_to_data[episode_id] = {
                "states": npz_file["states"],
                "gfs": npz_file["gfs"],
                "actions": npz_file["actions"],
            }

            for j in range(len(npz_file["states"])):
                self.episode_steps.append((episode_id, j))

        self.aug = aug

    def __len__(self):
        return len(self.episode_steps)

    def __getitem__(self, idx):
        episode_id, step = self.episode_steps[idx]

        data = self.episode_id_to_data[episode_id]

        state = data["states"][step]
        gf_values = data["gfs"][step]
        actions = data["actions"][step]

        if self.aug:
            aug = random.random() > 0.5
        else:
            aug = False

        gf = np.zeros((len(gf_values), 3, 3), dtype=np.float32)
        for i, x in enumerate(gf_values):
            gf[i] = x

        label = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int64)
        for x, y, action_type in actions:
            if x == -1 or y == -1:
                break
            if aug:
                action_type = ActionType(action_type).transpose()
            label[y, x] = action_type

        if aug:
            state = transpose(state)
            label = transpose(label)
            gf[0] *= -1  # nebula_tile_drift_direction

        mask = state[25] > 0

        return state, gf, label, mask


# ===================#
#       Model        #
# ===================#

# source https://github.com/milesial/Pytorch-UNet/tree/master/unet


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AddGlobal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gf):
        return torch.cat([x, gf], dim=1)


class UNet(nn.Module):
    def __init__(
        self,
        n_channels=N_CHANNELS,
        n_classes=N_CLASSES,
        n_global=N_GLOBAL,
        bilinear=True,
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.add_global = AddGlobal()
        self.up1 = Up(512 + n_global, 256, bilinear)
        self.up2 = Up(256 + 128, 128, bilinear)
        self.up3 = Up(128 + 64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, gf):
        x1 = self.inc(x)  # 24
        x2 = self.down1(x1)  # 12
        x3 = self.down2(x2)  # 6
        x4 = self.down3(x3)  # 3
        x5 = self.add_global(x4, gf)
        x = self.up1(x5, x3)  # 6
        x = self.up2(x, x2)  # 12
        x = self.up3(x, x1)  # 24
        logits = self.outc(x)
        return logits


# ===================#
#       Train        #
# ===================#


def masked_loss(predictions, labels, mask, weights):
    """
    predictions: (batch_size, num_classes, height, width)
    labels: Ground truth labels (batch_size, height, width)
    mask: Binary mask (batch_size, height, width)
    """
    loss = F.cross_entropy(predictions, labels, weight=weights, reduction="none")
    loss = loss * mask
    loss = loss.sum(dim=[1, 2]) / mask.sum(dim=[1, 2])
    return loss.mean()


def get_acc(outs, actions, mask, label_to_acc):
    preds = torch.argmax(outs, dim=1)
    total = torch.sum(mask)
    correct = torch.sum((preds == actions) * mask)

    for action_type in ActionType:
        action_type = int(action_type)
        action_mask = (actions == action_type) * mask

        action_correct = torch.sum((preds == actions) * action_mask)
        action_total = torch.sum(action_mask)

        label_to_acc[action_type][0] += action_correct
        label_to_acc[action_type][1] += action_total

    return correct, total


def train_model(
    model,
    train_episodes,
    val_episodes,
    optimizer,
    scheduler,
    weights,
    num_epochs,
    model_name="model",
    num_episodes_per_epoch=10, # 1000
    batch_size=128,
):

    best_loss = 10**9

    val_loader = DataLoader(
        LuxDataset(val_episodes, aug=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    for epoch in range(num_epochs):
        model.to(device)

        np.random.shuffle(train_episodes)

        train_dataloader = DataLoader(
            LuxDataset(train_episodes[:num_episodes_per_epoch]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        phases = [("train", train_dataloader)]
        if (epoch + 1) % 5 == 0:
            phases.append(("val", val_loader))

        val_loss = 10**9

        for phase, dataloader in phases:
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_acc = 0
            correct = 0
            total = 0

            label_to_acc = {x: [0, 0] for x in ActionType}

            for item in tqdm(dataloader, leave=False):
                states = item[0].float()
                gf = item[1].float()
                label = item[2].long()
                mask = item[3].bool()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    policy = model(states, gf)
                    loss = masked_loss(policy, label, mask, weights)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    else:
                        _correct, _total = get_acc(policy, label, mask, label_to_acc)
                        correct += _correct
                        total += _total

                    epoch_loss += loss.item() * len(policy)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size

            if phase != "train":
                val_loss = epoch_loss
                epoch_acc = correct.double() / total

                msg = []
                for l, (c, t) in label_to_acc.items():
                    if t > 0:
                        msg.append(f"{ActionType(l)}: {float(c / t):.2f}")
                    else:
                        msg.append(f"{ActionType(l)}: nan")
                msg = ", ".join(msg)

                print("label to auc", msg)

                if scheduler is not None:
                    scheduler.step(epoch_loss)

            time.sleep(1)
            print(
                f"Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.5f} | Acc: {epoch_acc:.4f}"
            )

        if val_loss < best_loss:
            traced = torch.jit.trace(
                model.cpu(),
                example_inputs=(
                    torch.rand(1, N_CHANNELS, 24, 24),
                    torch.rand(1, N_GLOBAL, 3, 3),
                ),
            )
            model_path = f"{model_name}.pth"
            print(f"Saving model to `{model_path}`.")
            traced.save(model_path)
            best_loss = val_loss


def main(submission_ids, min_opp_score):
    train_episodes, val_episodes = select_episodes(submission_ids, min_opp_score)

    seed_everything(42)
    model = UNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=0, min_lr=1e-6, 
    )

    # center, up, right, down, left, sap
    weights = torch.Tensor([0.3, 1.0, 0.5, 0.5, 1.0, 1.0]).float()

    train_model(
        model,
        train_episodes,
        val_episodes,
        optimizer,
        scheduler,
        weights,
        num_epochs=20, # 50
        model_name=MODEL_NAME,
    )

if __name__ == "__main__":
    submission_ids = [42613059, 4261318]
    min_opp_score = 647
    main(submission_ids, min_opp_score)