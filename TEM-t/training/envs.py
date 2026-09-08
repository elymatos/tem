"""Environment generation and trajectory sampling for TEM-t experiments.

Provides 2D grid world environments with shared graph structure but
randomised sensory-to-node mappings across environments, enabling
the study of abstract spatial representations independent of
specific sensory cues.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import torch

from training.batch import TrajectoryBatch


# ---------------------------------------------------------------------------
# GridWorldSpec
# ---------------------------------------------------------------------------
@dataclass
class GridWorldSpec:
    """Definition of a 2D grid world environment structure.

    The graph G = (V, E) is a 4-connected grid (N/E/S/W actions).
    Boundary behaviour controls what happens when an action would
    move the agent off the grid.

    Attributes
    ----------
    height : int
        Number of rows.
    width : int
        Number of columns.
    n_actions : int, optional
        Number of actions (default 4 for N/E/S/W).
    boundary : str, optional
        Boundary handling: "stay", "wrap", or "reflect". Default "stay".
    """

    height: int
    width: int
    n_actions: int = 4
    boundary: str = "stay"

    @property
    def n_states(self) -> int:
        """Total number of nodes in the grid graph."""
        return self.height * self.width

    def __post_init__(self) -> None:
        if self.boundary not in ("stay", "wrap", "reflect"):
            raise ValueError(
                f"Unsupported boundary '{self.boundary}'. "
                f"Expected 'stay', 'wrap', or 'reflect'."
            )


# ---------------------------------------------------------------------------
# EnvironmentInstance
# ---------------------------------------------------------------------------
@dataclass
class EnvironmentInstance:
    """A concrete environment with a specific sensory assignment.

    The graph structure is shared across all environments, but the
    mapping from graph nodes to sensory classes (sensory_map) is
    independently randomised for each environment. This eliminates
    sensory correlations between adjacent positions, forcing the
    model to rely on transition structure.

    Attributes
    ----------
    spec : GridWorldSpec
        Shared grid structure definition.
    env_id : int
        Unique identifier for this environment instance.
    sensory_map : LongTensor [N_s]
        Mapping from graph node index to sensory class ID.
    """

    spec: GridWorldSpec
    env_id: int
    sensory_map: torch.LongTensor  # [N_s], sensory_map[state] = sensory_id


# ---------------------------------------------------------------------------
# Grid world transition helper
# ---------------------------------------------------------------------------
def _grid_to_xy(state: int, width: int) -> Tuple[int, int]:
    """Convert a linear state index to (row, col) coordinates."""
    return state // width, state % width


def _xy_to_grid(row: int, col: int, height: int, width: int) -> int:
    """Convert (row, col) coordinates to a linear state index."""
    return row * width + col


def step_grid(
    state: int,
    action: int,
    spec: GridWorldSpec,
) -> int:
    """Execute one action on the grid, returning the next state.

    Action encoding: 0 = North (row-1), 1 = East (col+1),
    2 = South (row+1), 3 = West (col-1), 4 = Stay.

    Parameters
    ----------
    state : int
        Current linear state index.
    action : int
        Action index (0-3).
    spec : GridWorldSpec
        Grid structure defining dimensions and boundary behaviour.

    Returns
    -------
    int
        Next linear state index.
    """
    H, W = spec.height, spec.width
    row, col = _grid_to_xy(state, W)

    if action == 0:      # North
        row -= 1
    elif action == 1:    # East
        col += 1
    elif action == 2:    # South
        row += 1
    elif action == 3:    # West
        col -= 1
    elif action == 4:    # Stay — no movement
        pass

    # Boundary handling
    if spec.boundary == "stay":
        row = max(0, min(H - 1, row))
        col = max(0, min(W - 1, col))
    elif spec.boundary == "wrap":
        row = row % H
        col = col % W
    elif spec.boundary == "reflect":
        if row < 0:
            row = 1
        elif row >= H:
            row = H - 2
        if col < 0:
            col = 1
        elif col >= W:
            col = W - 2
        row = max(0, min(H - 1, row))
        col = max(0, min(W - 1, col))

    return _xy_to_grid(row, col, H, W)


# ---------------------------------------------------------------------------
# TrajectorySampler
# ---------------------------------------------------------------------------
class TrajectorySampler:
    """Sample random walk trajectories from randomised grid environments.

    For each episode, a fresh environment instance is drawn (or selected)
    and a random walk is simulated. This produces batches of (x, a, s)
    sequences suitable for training the TEM-t model.

    Parameters
    ----------
    spec : GridWorldSpec
        Grid world structure.
    n_sensory : int
        Number of distinct sensory classes.
    episode_length : int
        Number of actions per trajectory (T). Sequence length is T+1.
    n_envs : int
        Number of pre-generated environment instances.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        spec: GridWorldSpec,
        n_sensory: int,
        episode_length: int,
        n_envs: int,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        self.n_sensory = n_sensory
        self.episode_length = episode_length
        self.n_envs = n_envs

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        # Pre-generate environment instances
        self.envs = self._generate_envs(n_envs)

    def _generate_envs(self, n: int) -> list:
        """Generate n environment instances with random sensory maps.

        Each node is assigned a sensory class uniformly at random
        (with replacement), so multiple nodes can share the same
        sensory class, creating sensory aliasing.

        Parameters
        ----------
        n : int
            Number of environments to generate.

        Returns
        -------
        list of EnvironmentInstance
        """
        envs = []
        for i in range(n):
            sensory_map = torch.randint(
                0, self.n_sensory,
                (self.spec.n_states,),
                generator=self.generator,
            )
            envs.append(EnvironmentInstance(
                spec=self.spec,
                env_id=i,
                sensory_map=sensory_map,
            ))
        return envs

    def _sample_single_trajectory(
        self,
        env: EnvironmentInstance,
    ) -> Tuple[
        torch.LongTensor,   # x_ids [T+1]
        torch.LongTensor,   # actions [T]
        torch.LongTensor,   # states [T+1]
    ]:
        """Sample one random walk trajectory in a given environment.

        The agent starts at a random node and takes random actions.

        Parameters
        ----------
        env : EnvironmentInstance
            The environment to walk in.

        Returns
        -------
        x_ids : LongTensor [T+1]
            Sensory class IDs along the trajectory.
        actions : LongTensor [T]
            Action indices.
        states : LongTensor [T+1]
            Graph node indices along the trajectory.
        """
        T = self.episode_length
        N_s = self.spec.n_states

        # Random start state
        s0 = int(torch.randint(0, N_s, (1,), generator=self.generator).item())

        states = torch.zeros(T + 1, dtype=torch.long)
        actions = torch.zeros(T, dtype=torch.long)

        states[0] = s0

        for t in range(T):
            a = int(torch.randint(0, self.spec.n_actions, (1,), generator=self.generator).item())
            actions[t] = a
            states[t + 1] = step_grid(int(states[t].item()), a, self.spec)

        # Convert states to sensory IDs via the environment's mapping
        x_ids = env.sensory_map[states]

        return x_ids, actions, states

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> TrajectoryBatch:
        """Sample a batch of random-walk trajectories.

        Each trajectory uses a randomly selected environment instance.

        Parameters
        ----------
        batch_size : int
            Number of trajectories in the batch.
        device : torch.device
            Target device for output tensors.

        Returns
        -------
        TrajectoryBatch with x_ids, x, actions, states, and env_ids.
        """
        T = self.episode_length

        x_ids_batch = torch.zeros(batch_size, T + 1, dtype=torch.long)
        actions_batch = torch.zeros(batch_size, T, dtype=torch.long)
        states_batch = torch.zeros(batch_size, T + 1, dtype=torch.long)
        env_ids_batch = torch.zeros(batch_size, dtype=torch.long)

        for b in range(batch_size):
            env_idx = int(torch.randint(0, self.n_envs, (1,), generator=self.generator).item())
            env = self.envs[env_idx]

            x_ids, actions, states = self._sample_single_trajectory(env)

            x_ids_batch[b] = x_ids
            actions_batch[b] = actions
            states_batch[b] = states
            env_ids_batch[b] = env_idx

        # One-hot encode sensory IDs
        x_batch = torch.nn.functional.one_hot(
            x_ids_batch, num_classes=self.n_sensory
        ).float()

        return TrajectoryBatch(
            x_ids=x_ids_batch.to(device),
            x=x_batch.to(device),
            actions=actions_batch.to(device),
            states=states_batch.to(device),
            env_ids=env_ids_batch.to(device),
        )
