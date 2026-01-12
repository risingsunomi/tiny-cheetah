import asyncio
import base64
import json
import os
import socket
import unittest

import tinygrad
from tiny_cheetah.orchestration.peer_client import PeerClient


def _stop_peer_client(client: PeerClient) -> None:
    client._udp_stop.set()
    if client._udp_thread is not None:
        client._udp_thread.join(timeout=1.0)

TEST_UDP_PORT = 6668

class TestPeerClient(unittest.IsolatedAsyncioTestCase):
    async def test_peer_client_send_and_recv_tensor_bytes(self):
        previous_port = os.environ.get("TC_PORT")
        os.environ["TC_PORT"] = "0"
        client = None
        try:
            client = PeerClient()
            _stop_peer_client(client)

            recv_state = {}
            recv_event = asyncio.Event()

            async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                recv_state["raw"] = await reader.read(65536)
                writer.close()
                await writer.wait_closed()
                recv_event.set()

            server = await asyncio.start_server(handle, "127.0.0.1", TEST_UDP_PORT)
            host, port = server.sockets[0].getsockname()[:2]

            tensor_bytes = tinygrad.Tensor.randn(10, 10).numpy().tobytes()
            await asyncio.to_thread(client.send_tensor_bytes, tensor_bytes, address=(host, port))
            await asyncio.wait_for(recv_event.wait(), timeout=2.0)
            server.close()
            await server.wait_closed()

            msg = json.loads(recv_state["raw"].decode("utf-8"))
            self.assertEqual(msg["command"], "tensor_bytes")
            payload = base64.b64decode(msg["payload"]["buffer"])
            self.assertEqual(payload, tensor_bytes)

            recv_state = {}
            recv_task = asyncio.create_task(
                asyncio.to_thread(
                    client.recv_tensor_bytes,
                    timeout=2.0,
                    bind_address=("127.0.0.1", TEST_UDP_PORT),
                )
            )
            await asyncio.sleep(0)

            payload = {"payload": {"buffer": base64.b64encode(tensor_bytes).decode("ascii")}}
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sender.sendto(json.dumps(payload).encode("utf-8"), ("127.0.0.1", TEST_UDP_PORT))
            sender.close()

            recv_state["data"] = await asyncio.wait_for(recv_task, timeout=2.0)

            self.assertEqual(recv_state["data"], tensor_bytes)
        finally:
            if client is not None:
                _stop_peer_client(client)
            if previous_port is None:
                os.environ.pop("TC_PORT", None)
            else:
                os.environ["TC_PORT"] = previous_port
