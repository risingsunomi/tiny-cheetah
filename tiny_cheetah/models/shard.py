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

    def __str__(self):
        return f"Shard(model_name={self.model_name}, start_layer={self.start_layer}, end_layer={self.end_layer}, total_layers={self.total_layers})"