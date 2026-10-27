"""gridfm_module_template — a forkable hello-world plugin for gridfm-graphkit.

Importing this package registers everything the plugin provides into graphkit's
global registries (see ``model.py`` and ``transforms.py``). graphkit's
``--plugins gridfm_module_template`` flag imports this package so the
``@MODELS_REGISTRY.register`` / ``@TRANSFORM_REGISTRY.register`` decorators fire.

To build your own module: fork this repo, rename the package, and replace the
model/transform with your own. Keep the registration decorators — that is the
whole contract with graphkit.
"""

# Importing these submodules executes the registration decorators as a side
# effect. Keep these imports; without them nothing registers.
from . import model  # noqa: F401  (registers "HelloGridModel")
from . import transforms  # noqa: F401  (registers "HelloWorld")


def register() -> None:
    """Idempotent entry-point hook.

    Importing this package already registers everything (the module-level
    imports above). This function exists so a future graphkit entry-point scan
    (``gridfm_graphkit.plugins``) has something to call. It is a no-op beyond
    ensuring the submodules are imported.
    """
    # The imports at module scope already did the work; nothing else to do.
    return None
