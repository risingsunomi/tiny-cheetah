import asyncio
import base64

import numpy as np
import pytest

from tiny_cheetah.orchestration.model_client import ModelClient, Tensor
from tiny_cheetah.orchestration.peer import PeerInfo
from tiny_cheetah.orchestration.server import ServerProfile
from tiny_cheetah.models.shard import Shard


@pytest.mark.skipif(Tensor is None, reason="tinygrad not available")
def test_compute_resp_returns_encoded_tensor():
    server = ServerProfile(server_id="srv-1", address=("localhost", 0))
    client = ModelClient(server)
    shard = Shard("demo", 0, 1, 1)
    data = np.arange(4, dtype=np.float32).tobytes()
    resp = client.compute_resp(shard, data, [], 0)
    assert resp["peer_id"] == server.server_id
    assert resp["shard"]["model_name"] == "demo"
    encoded = resp["tensor"]
    decoded = np.frombuffer(base64.b64decode(encoded), dtype=np.float32)
    assert decoded.shape == (4,)


def test_plan_shards_assigns_and_sets_peer_shard():
    peers = [
        PeerInfo("p1", "h1", 1, device_report={"ram_gb": 8}),
        PeerInfo("p2", "h2", 1, device_report={"ram_gb": 4}),
        PeerInfo("p3", "h3", 1, device_report={"ram_gb": 2}),
    ]
    shards = ModelClient.plan_shards(peers, "demo", total_layers=12)
    assert len(shards) == 3
    assert shards[0].start_layer == 0
    assert shards[-1].end_layer == 12
    for peer in peers:
        assert peer.shard is not None
