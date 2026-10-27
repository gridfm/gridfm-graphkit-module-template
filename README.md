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
| Configs | `configs/*.yaml` | PF / OPF configs wiring the model to graphkit's existing transforms, one using the custom transform composition, and one declaring an explicit transform list. |
| Tests + CI | `tests/`, `.github/workflows/ci.yml` | Registration + instantiation + forward pass on a synthetic graph — no dataset needed. |

### Configs at a glance

| File | Shows |
|------|-------|
| `hello_pf_case14.yaml` | Model swap; reuse graphkit's built-in PowerFlow transforms by `task_name`. |
| `hello_opf_case14.yaml` | Same for OptimalPowerFlow. |
| `hello_se_case14.yaml` | Select the template's own registered transform composition (`HelloWorld`). |
| `hello_pf_custom_transforms_case14.yaml` | **Declare the transform pipeline explicitly** via `data.transforms`, overriding the default composition. |

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
gridfm_graphkit train --config configs/hello_pf_case14.yaml
```

No `--plugins` flag: graphkit auto-discovers this package through the
`gridfm_graphkit.plugins` entry point (declared in `pyproject.toml`), so just
having it installed registers the model + transforms. `--plugins
gridfm_module_template` still works if you want to load it explicitly.

### Declaring your own transform pipeline

Instead of letting `task.task_name` select a whole transform composition, a
config can list transforms explicitly — graphkit composes them in order,
resolving each from the registry (built-in or plugin). Entries are a bare name
or a `name` + kwargs mapping:

```yaml
data:
  transforms:
    - RemoveInactiveBranches
    - name: AddPFHeteroMask
      mask_ref_vm: false
    - ApplyMasking
```

If `data.transforms` is omitted, graphkit falls back to the default composition
for `task.task_name`. See `configs/hello_pf_custom_transforms_case14.yaml`.

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
