from tiny_cheetah.orchestration.model_engine import ModelEngine
from tiny_cheetah.orchestration.cdevice import CDevice


def test_plan_shards_assigns_and_sets_peer_shard():
    peers = []
    for idx, ram in enumerate([8, 4, 2]):
        p = CDevice(f"p{idx+1}", "0.0.0.0", 0)
        p.cpu_ram = str(ram)
        p.gpu_vram = ""
        peers.append(p)
    shards = ModelEngine.plan_shards(peers, "demo", total_layers=12)
    assert len(shards) == 3
    assert shards[0].start_layer == 0
    assert shards[-1].end_layer == 12
    for peer in peers:
        assert peer.shard is not None
