"""Synchronous WebSocket client for the Mineflayer RL bridge."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from typing import Any

from websockets.sync.client import ClientConnection, connect


class MinecraftBridgeError(RuntimeError):
    """Raised when the Mineflayer bridge rejects or cannot complete a request."""


class MinecraftWebSocketBridge:
    """One-request-at-a-time client suitable for Gymnasium's synchronous API."""

    def __init__(
        self,
        uri: str = "ws://127.0.0.1:8765",
        *,
        timeout: float = 10.0,
    ) -> None:
        self.uri = uri
        self.timeout = timeout
        self._connection: ClientConnection | None = None
        self._next_request_id = 1
        self._lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._connection is not None

    def connect(self) -> None:
        if self._connection is not None:
            return
        try:
            self._connection = connect(
                self.uri,
                open_timeout=self.timeout,
                close_timeout=2.0,
                max_size=64 * 1024,
            )
            self.request("ping")
        except Exception as error:
            self._connection = None
            raise MinecraftBridgeError(
                f"Could not connect to Mineflayer at {self.uri}. "
                "Start the Minecraft server and bot first."
            ) from error

    def request(self, request_type: str, **payload: Any) -> dict[str, Any]:
        if self._connection is None:
            self.connect()

        with self._lock:
            assert self._connection is not None

            request_id = self._next_request_id
            self._next_request_id += 1
            message = {"id": request_id, "type": request_type, **payload}

            try:
                self._connection.send(json.dumps(message))
                raw_response = self._connection.recv(timeout=self.timeout)
                response = json.loads(raw_response)
            except Exception as error:
                self.close()
                raise MinecraftBridgeError(
                    f"Mineflayer bridge request {request_type!r} failed"
                ) from error

            if not isinstance(response, Mapping) or response.get("id") != request_id:
                raise MinecraftBridgeError("Mineflayer bridge returned an invalid response")
            if not response.get("ok"):
                raise MinecraftBridgeError(str(response.get("error", "Unknown bridge error")))
            return dict(response)

    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def __enter__(self) -> MinecraftWebSocketBridge:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
