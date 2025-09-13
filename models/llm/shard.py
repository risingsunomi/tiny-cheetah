"""
Model shard for layers to use
"""

class Shard:
    def __init__(
        self,
        model_name: str,
        start_layer: int,
        end_layer: int,
        total_layers: int
    ):
        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.total_layers = total_layers