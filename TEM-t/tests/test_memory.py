"""Unit tests for TEMMemory module.

Tests:
  - init_empty creates empty memory
  - init_from_observation creates memory with size 1
  - write increments memory size
  - dedup prevents redundant writes
  - memory does not exceed max capacity
"""

import torch

from model.memory import TEMMemory
from model.types import MemoryState


class TestTEMMemory:
    """Tests for TEMMemory."""

    @staticmethod
    def _make_dummy_memory_module() -> TEMMemory:
        """Create a memory module with small dimensions for testing."""
        return TEMMemory(
            max_memory=20,
            d_g=16,
            d_k=8,
            d_v=12,
            n_sensory=10,
            dedup=True,
            dedup_threshold=1.5,
        )

    @staticmethod
    def test_init_empty() -> None:
        """init_empty should create memory with size 0."""
        mem_mod = TestTEMMemory._make_dummy_memory_module()
        memory = mem_mod.init_empty(batch_size=4, device=torch.device("cpu"))

        assert isinstance(memory, MemoryState)
        assert memory.keys_g.shape == (4, 20, 8)
        assert memory.values_x.shape == (4, 20, 12)
        assert memory.values_g.shape == (4, 20, 16)
        assert memory.raw_x.shape == (4, 20, 10)
        assert memory.size.tolist() == [0, 0, 0, 0]
        assert not memory.valid_mask.any()

    @staticmethod
    def test_init_from_observation() -> None:
        """init_from_observation should set size to 1."""
        mem_mod = TestTEMMemory._make_dummy_memory_module()

        B = 2
        g_0 = torch.randn(B, 16)
        x_0 = torch.zeros(B, 10)
        x_0[0, 3] = 1.0
        x_0[1, 7] = 1.0
        key_g = torch.randn(B, 8)
        value_x = torch.randn(B, 12)

        memory = mem_mod.init_from_observation(g_0, x_0, key_g, value_x)

        assert memory.size.tolist() == [1, 1]
        assert memory.valid_mask[:, 0].all()
        assert not memory.valid_mask[:, 1:].any()

    @staticmethod
    def test_write_increments_size() -> None:
        """Writing should increment memory size."""
        mem_mod = TestTEMMemory._make_dummy_memory_module()
        mem_mod.dedup_enabled = False  # disable dedup for this test

        B = 2
        memory = mem_mod.init_empty(B, torch.device("cpu"))

        # Write first slot
        g = torch.randn(B, 16)
        x = torch.zeros(B, 10)
        x[0, 2] = 1.0
        x[1, 5] = 1.0
        kg = torch.randn(B, 8)
        vx = torch.randn(B, 12)

        memory, wrote = mem_mod.write(memory, kg, vx, g, x)

        assert wrote.all()
        assert memory.size.tolist() == [1, 1]
        assert memory.valid_mask[:, 0].all()

        # Write second slot with different values
        g2 = torch.randn(B, 16)
        x2 = torch.zeros(B, 10)
        x2[0, 8] = 1.0
        x2[1, 1] = 1.0
        kg2 = torch.randn(B, 8)
        vx2 = torch.randn(B, 12)

        memory, wrote = mem_mod.write(memory, kg2, vx2, g2, x2)

        assert wrote.all()
        assert memory.size.tolist() == [2, 2]
        assert memory.valid_mask[:, 0].all()
        assert memory.valid_mask[:, 1].all()

    @staticmethod
    def test_dedup_prevents_duplicate() -> None:
        """Writing the same conjunction should be blocked by dedup."""
        mem_mod = TestTEMMemory._make_dummy_memory_module()

        B = 1
        g = torch.randn(B, 16)
        x = torch.zeros(B, 10)
        x[0, 3] = 1.0
        kg = torch.randn(B, 8)
        vx = torch.randn(B, 12)

        # Write initial
        memory, wrote = mem_mod.write(
            mem_mod.init_empty(B, torch.device("cpu")),
            kg, vx, g, x,
        )
        assert wrote.item()

        # Write the same again
        memory, wrote = mem_mod.write(memory, kg, vx, g, x)
        assert not wrote.item(), "Dedup should prevent writing duplicate"
        assert memory.size.item() == 1, "Size should remain 1"

    @staticmethod
    def test_max_capacity() -> None:
        """Memory should not exceed max_memory."""
        max_mem = 5
        mem_mod = TEMMemory(
            max_memory=max_mem,
            d_g=8,
            d_k=4,
            d_v=8,
            n_sensory=5,
            dedup=False,
        )

        B = 1
        memory = mem_mod.init_empty(B, torch.device("cpu"))

        for i in range(max_mem + 3):
            g = torch.randn(B, 8) * (i + 1)  # ensure different
            x = torch.zeros(B, 5)
            x[0, i % 5] = 1.0
            kg = torch.randn(B, 4)
            vx = torch.randn(B, 8)
            memory, wrote = mem_mod.write(memory, kg, vx, g, x)

        assert memory.size.max().item() <= max_mem

    @staticmethod
    def test_memory_size_tracking() -> None:
        """size should accurately track the number of written slots."""
        mem_mod = TestTEMMemory._make_dummy_memory_module()
        mem_mod.dedup_enabled = False

        B = 3
        memory = mem_mod.init_empty(B, torch.device("cpu"))

        for i in range(5):
            g = torch.randn(B, 16) * (i + 1)
            x = torch.zeros(B, 10)
            x[:, i % 10] = 1.0
            kg = torch.randn(B, 8)
            vx = torch.randn(B, 12)
            memory, _ = mem_mod.write(memory, kg, vx, g, x)

        assert memory.size.tolist() == [5, 5, 5]
        assert memory.valid_mask[:, :5].all()
        assert not memory.valid_mask[:, 5:].any()
