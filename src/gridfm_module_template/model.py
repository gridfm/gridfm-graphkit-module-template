"""Hello-world model plugin for gridfm-graphkit.

This is the minimal shape of a 3rd-party model. It mirrors the contract that
graphkit's built-in ``GNS_heterogeneous`` satisfies (see
``gridfm_graphkit/models/gnn_heterogeneous_gns.py``), stripped down to the bare
essentials so you can see what graphkit actually requires:

The contract (duck-typed — graphkit treats your model as a black box):
  * ``__init__(self, args)`` — receives the whole config namespace. Read your
    hyperparameters from ``args.model.*`` and the active task from
    ``args.task.task_name``.
  * ``forward(self, batch, return_embeddings=False)`` — ``batch`` is a PyG
    ``HeteroData`` with node types ``"bus"`` / ``"gen"`` and relations
    ``("bus","connects","bus")``, ``("gen","connected_to","bus")``,
    ``("bus","connected_to","gen")``. Access features via ``batch.x_dict`` etc.
  * Return a dict ``{"bus": Tensor[num_bus, output_bus_dim],
    "gen": Tensor[num_gen, output_gen_dim]}``. When ``return_embeddings`` is
    True, return ``(predictions, embeddings_dict)``.

That is the entire interface. No base class to inherit, no ABC — just register
an ``nn.Module`` with the shape above.
"""

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from gridfm_graphkit.io.registries import MODELS_REGISTRY


@MODELS_REGISTRY.register("HelloGridModel")
class HelloGridModel(nn.Module):
    """A deliberately tiny heterogeneous GNN — one message-passing layer and a
    linear head per node type. Not meant to be accurate; meant to be the
    smallest thing that honors graphkit's model contract so you can swap it for
    your own architecture one piece at a time.
    """

    def __init__(self, args) -> None:
        super().__init__()
        # Read config exactly like the built-in model does, so the example
        # configs (configs/*.yaml) carry over to your real model unchanged.
        self.hidden_dim = getattr(args.model, "hidden_size", 32)
        self.input_bus_dim = args.model.input_bus_dim
        self.input_gen_dim = args.model.input_gen_dim
        self.output_bus_dim = args.model.output_bus_dim
        self.output_gen_dim = args.model.output_gen_dim
        self.task = args.task.task_name

        # Project each node type into a shared hidden space.
        self.in_bus = nn.Linear(self.input_bus_dim, self.hidden_dim)
        self.in_gen = nn.Linear(self.input_gen_dim, self.hidden_dim)

        # One heterogeneous message-passing layer over the three relations.
        # SAGEConv needs no edge attributes, which keeps this example minimal.
        self.conv = HeteroConv(
            {
                ("bus", "connects", "bus"): SAGEConv(self.hidden_dim, self.hidden_dim),
                ("gen", "connected_to", "bus"): SAGEConv(
                    (self.hidden_dim, self.hidden_dim), self.hidden_dim
                ),
                ("bus", "connected_to", "gen"): SAGEConv(
                    (self.hidden_dim, self.hidden_dim), self.hidden_dim
                ),
            },
            aggr="sum",
        )
        self.act = nn.LeakyReLU()

        # Prediction heads, one per node type.
        self.out_bus = nn.Linear(self.hidden_dim, self.output_bus_dim)
        self.out_gen = nn.Linear(self.hidden_dim, self.output_gen_dim)

    def forward(self, batch, return_embeddings: bool = False):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        h = {
            "bus": self.act(self.in_bus(x_dict["bus"])),
            "gen": self.act(self.in_gen(x_dict["gen"])),
        }
        h = self.conv(h, edge_index_dict)
        h = {k: self.act(v) for k, v in h.items()}

        predictions = {
            "bus": self.out_bus(h["bus"]),
            "gen": self.out_gen(h["gen"]),
        }
        if not return_embeddings:
            return predictions
        return predictions, {"bus": h["bus"], "gen": h["gen"]}
