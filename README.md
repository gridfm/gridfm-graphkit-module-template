# gridfm-module-template

A **forkable hello-world plugin** for [`gridfm-graphkit`](https://github.com/gridfm/gridfm-graphkit).
Fork it to ship your own power-grid model (and transforms) that run through
graphkit's training / evaluation / physics stack — **without forking graphkit
itself**.

## What it demonstrates

| Piece | File | What it shows |
|-------|------|---------------|
| Custom model | `src/gridfm_module_template/model.py` | The minimal `nn.Module` contract graphkit expects, registered as `HelloGridModel`. |
| Custom transform | `src/gridfm_module_template/transforms.py` | Registering your own transform composition (`HelloWorld`) that reuses graphkit's built-in preprocessing. |
| Configs | `configs/*.yaml` | PF / OPF configs wiring the model to graphkit's existing PowerFlow / OptimalPowerFlow transforms, plus one using the custom transform. |
| Tests + CI | `tests/`, `.github/workflows/ci.yml` | Registration + instantiation + forward pass on a synthetic graph — no dataset needed. |

## The model contract

graphkit treats your model as a black box. All it requires:

- `__init__(self, args)` — read hyperparameters from `args.model.*`, task from `args.task.task_name`.
- `forward(self, batch, return_embeddings=False)` — `batch` is a PyG `HeteroData`
  with node types `bus` / `gen` and relations `(bus,connects,bus)`,
  `(gen,connected_to,bus)`, `(bus,connected_to,gen)`.
- Return `{"bus": Tensor[n_bus, output_bus_dim], "gen": Tensor[n_gen, output_gen_dim]}`.

Register it with `@MODELS_REGISTRY.register("YourModelName")`. That's the whole deal — no base class.

## Install

```bash
pip install -e ".[test]"
```

This pulls `gridfm-graphkit` from its git branch (pinned in `pyproject.toml` —
switch to a PyPI release once one exists).

## Run

```bash
gridfm_graphkit train --config configs/hello_pf_case14.yaml \
    --plugins gridfm_module_template
```

`--plugins gridfm_module_template` imports this package so its registration
decorators fire. (Once graphkit adds a `gridfm_graphkit.plugins` entry-point
scan, installing the package will be enough — the entry point is already
declared in `pyproject.toml`.)

## Test

```bash
pytest tests/ -v
```

## Make it yours

1. Fork this repo.
2. Rename `src/gridfm_module_template/` to your package name (and update `pyproject.toml`).
3. Replace the body of `HelloGridModel` with your architecture — keep the `__init__(args)` / `forward -> {"bus","gen"}` shape.
4. Edit the configs to point `model.type` at your registered name.
5. Keep the tests green.
