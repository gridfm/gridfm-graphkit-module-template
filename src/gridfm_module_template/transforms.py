"""Hello-world custom transform plugin for gridfm-graphkit.

graphkit selects a transform *composition* by task name: ``get_task_transforms``
calls ``TRANSFORM_REGISTRY.create(args.task.task_name, args)``. The built-in
compositions are ``PowerFlow`` / ``OptimalPowerFlow`` / ``StateEstimation``
(see ``gridfm_graphkit/datasets/task_transforms.py``). You reuse those simply by
setting ``task.task_name`` in your config — no code needed.

This file shows the *other* half: registering your OWN transform composition so
a config can select it by name. It subclasses PyG ``Compose`` exactly like the
built-ins, reuses graphkit's existing preprocessing transforms, and adds a
trivial no-op step so you can see where a custom transform would slot in.

To use it: set ``task.task_name: HelloWorld`` in your config (see
``configs/hello_se_case14.yaml``). Note that the task_name also selects the
*task module* and *physics decoder*, so a genuinely new task_name needs those
registered too; this example registers under ``HelloWorld`` purely to
demonstrate the transform-registration mechanism.
"""

from torch_geometric.transforms import BaseTransform, Compose

from gridfm_graphkit.datasets.transforms import (
    RemoveInactiveBranches,
    RemoveInactiveGenerators,
    ApplyMasking,
)
from gridfm_graphkit.datasets.masking import AddPFHeteroMask
from gridfm_graphkit.io.registries import TRANSFORM_REGISTRY


class HelloNoOp(BaseTransform):
    """A custom transform that does nothing but prove the hook works.

    Replace the body with your own feature engineering. A transform receives a
    PyG ``HeteroData`` object and must return it (mutated or not).
    """

    def __call__(self, data):
        # Your custom preprocessing goes here. We just pass the graph through.
        return data


@TRANSFORM_REGISTRY.register("HelloWorld")
class HelloWorldTransforms(Compose):
    """Example 3rd-party transform composition.

    Reuses graphkit's built-in preprocessing + PF masking, then appends the
    custom :class:`HelloNoOp` step. This is the pattern for shipping your own
    transform pipeline while standing on graphkit's existing building blocks.
    """

    def __init__(self, args):
        transforms = [
            RemoveInactiveBranches(),
            RemoveInactiveGenerators(),
            AddPFHeteroMask(mask_ref_vm=False),
            HelloNoOp(),  # <-- your custom transform
            ApplyMasking(args=args),
        ]
        super().__init__(transforms)
