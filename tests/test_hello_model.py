"""Hello-world CI tests.

These assert the three things a 3rd-party module must get right, without any
dataset or training:
  1. importing the package registers the model + transform into graphkit,
  2. graphkit can instantiate the model from a config namespace,
  3. the model runs a forward pass on a tiny synthetic HeteroData batch and
     returns the required {"bus", "gen"} prediction dict with correct shapes.

A tiny hand-built graph is used so the test is fast, deterministic, and needs
no data download.
"""

import torch
import pytest
from torch_geometric.data import HeteroData

from gridfm_graphkit.io.registries import MODELS_REGISTRY, TRANSFORM_REGISTRY
from gridfm_graphkit.io.param_handler import NestedNamespace

# Importing the package triggers registration via its decorators.
import gridfm_module_template  # noqa: F401


# Feature dims mirror the case14 example configs.
INPUT_BUS_DIM = 15
INPUT_GEN_DIM = 6
OUTPUT_BUS_DIM = 2
OUTPUT_GEN_DIM = 1


def make_args():
    """Minimal config namespace, shaped like a parsed graphkit YAML."""
    return NestedNamespace(
        model={
            "type": "HelloGridModel",
            "hidden_size": 16,
            "input_bus_dim": INPUT_BUS_DIM,
            "input_gen_dim": INPUT_GEN_DIM,
            "output_bus_dim": OUTPUT_BUS_DIM,
            "output_gen_dim": OUTPUT_GEN_DIM,
        },
        task={"task_name": "PowerFlow"},
    )


def make_batch(num_bus=4, num_gen=2):
    """A tiny synthetic heterogeneous grid honoring graphkit's schema."""
    data = HeteroData()
    data["bus"].x = torch.randn(num_bus, INPUT_BUS_DIM)
    data["gen"].x = torch.randn(num_gen, INPUT_GEN_DIM)

    # bus-bus ring so every bus has a neighbor
    src = torch.arange(num_bus)
    dst = (src + 1) % num_bus
    data["bus", "connects", "bus"].edge_index = torch.stack([src, dst])

    # each generator attached to a distinct bus (both directions)
    gen_idx = torch.arange(num_gen)
    bus_idx = torch.arange(num_gen) % num_bus
    data["gen", "connected_to", "bus"].edge_index = torch.stack([gen_idx, bus_idx])
    data["bus", "connected_to", "gen"].edge_index = torch.stack([bus_idx, gen_idx])
    return data


def test_model_is_registered():
    assert "HelloGridModel" in MODELS_REGISTRY


def test_transform_is_registered():
    assert "HelloWorld" in TRANSFORM_REGISTRY


def test_model_instantiates_via_registry():
    model = MODELS_REGISTRY.create("HelloGridModel", make_args())
    assert isinstance(model, torch.nn.Module)


def test_forward_pass_shapes():
    model = MODELS_REGISTRY.create("HelloGridModel", make_args())
    model.eval()
    batch = make_batch(num_bus=4, num_gen=2)
    with torch.no_grad():
        out = model(batch)
    assert set(out.keys()) == {"bus", "gen"}
    assert out["bus"].shape == (4, OUTPUT_BUS_DIM)
    assert out["gen"].shape == (2, OUTPUT_GEN_DIM)


def test_forward_returns_embeddings_when_requested():
    model = MODELS_REGISTRY.create("HelloGridModel", make_args())
    model.eval()
    batch = make_batch()
    with torch.no_grad():
        preds, emb = model(batch, return_embeddings=True)
    assert set(preds.keys()) == {"bus", "gen"}
    assert set(emb.keys()) == {"bus", "gen"}


def test_config_declared_transform_list_builds():
    """`data.transforms` in config composes exactly the listed transforms,
    overriding the default task-name composition."""
    from gridfm_graphkit.io.param_handler import get_task_transforms

    args = NestedNamespace(
        task={"task_name": "PowerFlow"},
        data={
            "mask_value": 0.0,
            "transforms": [
                "RemoveInactiveBranches",
                "RemoveInactiveGenerators",
                {"name": "AddPFHeteroMask", "mask_ref_vm": False},
                "ApplyMasking",
            ],
        },
    )
    compose = get_task_transforms(args)
    # PyG Compose stores its steps in `.transforms`; we declared four.
    assert len(compose.transforms) == 4


def test_default_composition_still_used_without_list():
    """Absent `data.transforms`, graphkit falls back to the task-name
    composition (backward compatible)."""
    from gridfm_graphkit.io.param_handler import get_task_transforms

    args = NestedNamespace(
        task={"task_name": "PowerFlow"},
        data={"mask_value": 0.0},  # no `transforms` key
    )
    compose = get_task_transforms(args)
    # The built-in PowerFlow composition is a non-empty Compose.
    assert len(compose.transforms) >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
