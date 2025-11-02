import numpy as np
import torch
from typing import Dict, Tuple, Optional
from envs.il_wrapper import ILWrapper


class ILRewardComputer:
    """
    Computes IL-based auxiliary rewards for RL training
    Rewards agents for matching IL expert actions
    """
    
    def __init__(
        self,
        bonus_per_match: float = 0.1,
        penalty_per_mismatch: float = 0.1,
        weight: float = 0.1,  # lambda
        anneal: bool = True,
        anneal_start: float = 0.5,
        anneal_end: float = 0.05,
        anneal_steps: int = 500_000,
        compare_sap_targets: bool = False,  # Compare SAP targets (needs SAP-UNet)
    ):
        """
            bonus_per_match: Reward bonus per matching action
            penalty_per_mismatch: Reward penalty per mismatched action
            weight: Base weight for IL reward (lambda)
            anneal: Whether to anneal weight over training
            anneal_start: Starting weight value
            anneal_end: Ending weight value
            anneal_steps: Number of steps to anneal over
            compare_sap_targets: Whether to compare SAP target positions
        """
        self.bonus_per_match = bonus_per_match
        self.penalty_per_mismatch = penalty_per_mismatch
        self.base_weight = weight
        self.anneal = anneal
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end
        self.anneal_steps = anneal_steps
        self.compare_sap_targets = compare_sap_targets
        
        self.current_weight = anneal_start if anneal else weight
        self.current_step = 0
        
        # Statistics
        self.total_comparisons = 0
        self.total_matches = 0
    
    def get_weight(self, global_step: Optional[int] = None) -> float:
        # current IL reward weight with annealing
        if not self.anneal:
            return self.base_weight
        
        if global_step is not None:
            self.current_step = global_step
        
        if self.current_step >= self.anneal_steps:
            self.current_weight = self.anneal_end
        else:
            # Linear annealing
            progress = self.current_step / self.anneal_steps
            self.current_weight = (
                self.anneal_start + 
                (self.anneal_end - self.anneal_start) * progress
            )
        
        return self.current_weight
    
    def compute_reward(
        self,
        rl_actions: np.ndarray,
        il_actions: np.ndarray,
        il_mask: np.ndarray,
        unit_mask: np.ndarray,
        global_step: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute RL and IL reward for batch of actions
        """
        batch_size = rl_actions.shape[0]
        max_units = rl_actions.shape[1]
        
        # Get current weight
        weight = self.get_weight(global_step)
        
        # Combine masks - only compare where both IL and unit are valid
        valid_mask = il_mask & unit_mask  # (batch_size, max_units)
        
        # Compare action types (column 0)
        rl_action_types = rl_actions[:, :, 0]  # (batch_size, max_units)
        il_action_types = il_actions[:, :, 0]  # (batch_size, max_units)
        
        action_type_match = (rl_action_types == il_action_types)  # (batch_size, max_units)
        
        # If comparing SAP targets, also check those for SAP actions
        if self.compare_sap_targets:
            # Identify SAP actions
            is_sap = (rl_action_types == 5) & (il_action_types == 5)  # (batch_size, max_units)
            
            # Compare SAP targets (columns 1 and 2)
            sap_target_match = (
                (rl_actions[:, :, 1] == il_actions[:, :, 1]) &
                (rl_actions[:, :, 2] == il_actions[:, :, 2])
            )
            
            # For SAP actions, require both action type and target to match
            full_match = action_type_match.copy()
            full_match[is_sap] = full_match[is_sap] & sap_target_match[is_sap]
        else:
            full_match = action_type_match
        
        # Apply valid mask
        full_match_valid = full_match & valid_mask
        mismatch_valid = (~full_match) & valid_mask
        
        # Count matches and mismatches per environment
        matches_per_env = full_match_valid.sum(axis=1).astype(np.float32)
        mismatches_per_env = mismatch_valid.sum(axis=1).astype(np.float32)
        
        # Compute raw IL reward (before weighting)
        raw_il_rewards = (
            matches_per_env * self.bonus_per_match -
            mismatches_per_env * self.penalty_per_mismatch
        )
        
        # Apply weight
        il_rewards = weight * raw_il_rewards
        
        # Update statistics
        self.total_comparisons += valid_mask.sum()
        self.total_matches += full_match_valid.sum()
        
        # Compute info
        total_valid = valid_mask.sum()
        agreement_rate = (
            self.total_matches / max(self.total_comparisons, 1)
        )
        
        info = {
            "il_reward_mean": float(il_rewards.mean()),
            "il_reward_std": float(il_rewards.std()),
            "il_weight": float(weight),
            "il_agreement_rate": float(agreement_rate),
            "il_matches_per_env": float(matches_per_env.mean()),
            "il_mismatches_per_env": float(mismatches_per_env.mean()),
            "il_total_comparisons": int(self.total_comparisons),
            "il_total_matches": int(self.total_matches),
        }
        
        return il_rewards, info
    
    def reset_statistics(self):
        self.total_comparisons = 0
        self.total_matches = 0
    
    def get_agreement_rate(self) -> float:
        if self.total_comparisons == 0:
            return 0.0
        return self.total_matches / self.total_comparisons


class ILRewardShaper:
    
    def __init__(
        self,
        il_wrapper,
        reward_computer: ILRewardComputer,
    ):
        self.il_wrapper = il_wrapper
        self.reward_computer = reward_computer
    
    def compute_rewards(
        self,
        obs_batch: list,
        team_ids: list,
        rl_actions: np.ndarray,
        unit_masks: np.ndarray,
        game_params: dict,
        global_step: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        
        # Get IL predictions
        il_actions, il_mask = self.il_wrapper.predict(
            obs_batch,
            team_ids,
            game_params,
            rl_actions=rl_actions,
        )
        
        # Compute IL rewards
        il_rewards, info = self.reward_computer.compute_reward(
            rl_actions=rl_actions,
            il_actions=il_actions,
            il_mask=il_mask,
            unit_mask=unit_masks,
            global_step=global_step,
        )
        
        return il_rewards, info
    
    def reset(self, env_idx: Optional[int] = None):
        self.il_wrapper.reset(env_idx)
    
    def reset_statistics(self):
        self.reward_computer.reset_statistics()


def create_il_reward_shaper(
    il_model_path: str,
    num_envs: int,
    device: str = "cuda",
    bonus_per_match: float = 0.1,
    penalty_per_mismatch: float = 0.1,
    weight: float = 0.1,
    anneal: bool = True,
    anneal_start: float = 0.5,
    anneal_end: float = 0.05,
    anneal_steps: int = 500_000,
    compare_sap_targets: bool = False,
    map_size: int = 24,
    max_units: int = 16,
) -> ILRewardShaper:    
    # Create IL wrapper
    il_wrapper = ILWrapper(
        model_path=il_model_path,
        num_envs=num_envs,
        device=device,
        map_size=map_size,
        max_units=max_units,
    )
    
    # Create reward computer
    reward_computer = ILRewardComputer(
        bonus_per_match=bonus_per_match,
        penalty_per_mismatch=penalty_per_mismatch,
        weight=weight,
        anneal=anneal,
        anneal_start=anneal_start,
        anneal_end=anneal_end,
        anneal_steps=anneal_steps,
        compare_sap_targets=compare_sap_targets,
    )
    
    # Create shaper
    return ILRewardShaper(il_wrapper, reward_computer)