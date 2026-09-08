"""TEM-t episodic memory management.

Implements a fixed-capacity episodic memory with optional deduplication.
Each memory slot stores a position-sensory binding:
    (key_g, value_x, value_g, raw_x)

Memory is read via TEMAttention (position-as-query keys) and written
after each observation. Deduplication prevents redundant writes when
the agent revisits previously seen position-sensory conjunctions,
avoiding attention bias toward frequently visited locations.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.types import MemoryState


class TEMMemory(nn.Module):
    """Fixed-capacity episodic memory with optional deduplication.

    Memory slots are pre-allocated as fixed-size tensors of shape
    [B, M_max, ...] with a boolean validity mask. This enables
    efficient batched attention without variable-length sequences.

    Parameters
    ----------
    max_memory : int
        Maximum number of memory slots (M). Typically T + 1.
    d_g : int
        Position representation dimensionality.
    d_k : int
        Key/query dimensionality.
    d_v : int
        Sensory value dimensionality.
    n_sensory : int
        Number of sensory classes (N_x).
    dedup : bool, optional
        Enable cosine-similarity-based deduplication. Default True.
    dedup_threshold : float, optional
        Threshold for the sum of cos(g) + cos(x) below which
        a new slot is considered novel enough to write. Default 1.5.
    """

    def __init__(
        self,
        max_memory: int,
        d_g: int,
        d_k: int,
        d_v: int,
        n_sensory: int,
        dedup: bool = True,
        dedup_threshold: float = 1.5,
    ) -> None:
        super().__init__()
        self.max_memory = max_memory
        self.d_g = d_g
        self.d_k = d_k
        self.d_v = d_v
        self.n_sensory = n_sensory
        self.dedup_enabled = dedup
        self.dedup_threshold = dedup_threshold

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    def init_empty(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> MemoryState:
        """Create an empty memory with no valid slots.

        Parameters
        ----------
        batch_size : int
            Number of parallel trajectories.
        device : torch.device
            Target device.
        dtype : torch.dtype, optional
            Floating-point type. Default float32.

        Returns
        -------
        MemoryState with size = 0 and all masks False.
        """
        M = self.max_memory

        keys_g = torch.zeros(batch_size, M, self.d_k, device=device, dtype=dtype)
        values_x = torch.zeros(batch_size, M, self.d_v, device=device, dtype=dtype)
        values_g = torch.zeros(batch_size, M, self.d_g, device=device, dtype=dtype)
        raw_x = torch.zeros(batch_size, M, self.n_sensory, device=device, dtype=dtype)
        valid_mask = torch.zeros(batch_size, M, device=device, dtype=torch.bool)
        size = torch.zeros(batch_size, device=device, dtype=torch.long)

        return MemoryState(
            keys_g=keys_g,
            values_x=values_x,
            values_g=values_g,
            raw_x=raw_x,
            valid_mask=valid_mask,
            size=size,
        )

    def init_from_observation(
        self,
        g_0: torch.FloatTensor,
        x_0: torch.FloatTensor,
        key_g_0: torch.FloatTensor,
        value_x_0: torch.FloatTensor,
        states_0: Optional[torch.LongTensor] = None,
    ) -> MemoryState:
        """Initialize memory with the first observation-position pair.

        Parameters
        ----------
        g_0 : FloatTensor [B, d_g]
            Initial position encoding.
        x_0 : FloatTensor [B, N_x]
            Initial sensory observation (one-hot).
        key_g_0 : FloatTensor [B, d_k]
            Projected position key from g_0.
        value_x_0 : FloatTensor [B, d_v]
            Projected sensory value from x_0.
        states_0 : Optional LongTensor [B]
            Ground-truth state indices.

        Returns
        -------
        MemoryState with size = 1 for each batch item.
        """
        batch_size = g_0.shape[0]
        device = g_0.device
        dtype = g_0.dtype

        memory = self.init_empty(batch_size, device, dtype)

        # Write into slot 0
        slot_idx = 0
        memory.keys_g[:, slot_idx] = key_g_0
        memory.values_x[:, slot_idx] = value_x_0
        memory.values_g[:, slot_idx] = g_0
        memory.raw_x[:, slot_idx] = x_0
        memory.valid_mask[:, slot_idx] = True
        memory.size[:] = 1

        if states_0 is not None:
            if memory.raw_states is None:
                memory.raw_states = torch.full(
                    (batch_size, self.max_memory), -1,
                    device=device, dtype=torch.long,
                )
            memory.raw_states[:, slot_idx] = states_0

        return memory

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    def should_write(
        self,
        memory: MemoryState,
        key_g_new: torch.FloatTensor,
        value_x_new: torch.FloatTensor,
    ) -> torch.BoolTensor:
        """Determine whether to write a new memory slot.

        A new slot is written if:
          - Memory is not full (size < max_memory)
          - AND (dedup is off OR the cosine similarity check passes)

        The dedup check computes:
          max over existing slots of [cos(key_g_new, key_g) + cos(value_x_new, value_x)]
          and requires this maximum to be < dedup_threshold.

        Parameters
        ----------
        memory : MemoryState
            Current memory state.
        key_g_new : FloatTensor [B, d_k]
            Projected position key candidate.
        value_x_new : FloatTensor [B, d_v]
            Projected sensory value candidate.

        Returns
        -------
        BoolTensor [B]
            True for batch items where a write should occur.
        """
        # Constraint 1: memory must not be full
        not_full = memory.size < self.max_memory  # [B]

        if not self.dedup_enabled:
            return not_full

        # Constraint 2: dedup via cosine similarity.
        # Skip when memory is empty (nothing to compare against).
        max_size = memory.size.max().item()
        if max_size == 0:
            return not_full

        # Only check dedup on items that can still write
        if not not_full.any():
            return not_full

        # Normalize and compute cosine similarities via batched BMM
        key_g_new_norm = F.normalize(key_g_new, p=2, dim=-1)          # [B, d_k]
        key_g_existing_norm = F.normalize(memory.keys_g, p=2, dim=-1) # [B, M, d_k]
        cos_g = torch.bmm(
            key_g_existing_norm,
            key_g_new_norm.unsqueeze(-1),
        ).squeeze(-1)                                                   # [B, M]

        value_x_new_norm = F.normalize(value_x_new, p=2, dim=-1)        # [B, d_v]
        value_x_existing_norm = F.normalize(memory.values_x, p=2, dim=-1)
        cos_x = torch.bmm(
            value_x_existing_norm,
            value_x_new_norm.unsqueeze(-1),
        ).squeeze(-1)

        combined = cos_g + cos_x  # [B, M]

        # Mask invalid slots
        combined_masked = combined.masked_fill(~memory.valid_mask, float("-inf"))

        max_similarity = combined_masked.max(dim=-1).values  # [B]

        is_novel = max_similarity < self.dedup_threshold  # [B]

        return not_full & is_novel

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write(
        self,
        memory: MemoryState,
        key_g_new: torch.FloatTensor,
        value_x_new: torch.FloatTensor,
        g_new: torch.FloatTensor,
        x_new: torch.FloatTensor,
        states_new: Optional[torch.LongTensor] = None,
    ) -> Tuple[MemoryState, torch.BoolTensor]:
        """Write a new memory slot for batch items that should be written.

        During inference (torch.no_grad): operates in-place for speed.
        During training (grad enabled): clones data tensors to preserve
        the autograd chain for BPTT through historical memory reads.

        Parameters
        ----------
        memory : MemoryState
            Current memory.
        key_g_new : FloatTensor [B, d_k]
            Position key to write.
        value_x_new : FloatTensor [B, d_v]
            Sensory value to write.
        g_new : FloatTensor [B, d_g]
            Raw position encoding to write.
        x_new : FloatTensor [B, N_x]
            Raw sensory observation to write.
        states_new : Optional LongTensor [B]
            Ground-truth state for the new observation.

        Returns
        -------
        memory_next : MemoryState
            Updated memory.
        wrote_mask : BoolTensor [B]
            True for batch items where a write actually occurred.
        """
        wrote_mask = self.should_write(memory, key_g_new, value_x_new)  # [B]

        if not wrote_mask.any():
            return memory, wrote_mask

        # Under autograd: clone the 3 data tensors that participate
        # in the backward pass. Under no_grad: in-place update.
        if torch.is_grad_enabled():
            memory = MemoryState(
                keys_g=memory.keys_g.clone(),
                values_x=memory.values_x.clone(),
                values_g=memory.values_g.clone(),
                raw_x=memory.raw_x.clone(),
                valid_mask=memory.valid_mask.clone(),
                size=memory.size.clone(),
                raw_states=(
                    memory.raw_states.clone()
                    if memory.raw_states is not None
                    else None
                ),
            )

        # Write into slot index = current size per batch item
        batch_indices = wrote_mask.nonzero(as_tuple=False).squeeze(-1)  # [n_write]
        write_slots = memory.size[wrote_mask]  # [n_write]

        for b_idx, slot in zip(batch_indices.tolist(), write_slots.tolist()):
            s = int(slot)
            memory.keys_g[b_idx, s] = key_g_new[b_idx]
            memory.values_x[b_idx, s] = value_x_new[b_idx]
            memory.values_g[b_idx, s] = g_new[b_idx]
            memory.raw_x[b_idx, s] = x_new[b_idx]
            memory.valid_mask[b_idx, s] = True
            memory.size[b_idx] = s + 1
            if memory.raw_states is not None and states_new is not None:
                memory.raw_states[b_idx, s] = states_new[b_idx]

        return memory, wrote_mask
