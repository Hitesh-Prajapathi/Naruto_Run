"""Binary WebSocket frame -> OpenCV BGR image, with the wire contract enforced
before any pixels reach the recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .protocol import FrameEnvelope, ProtocolError, decode_frame


@dataclass(frozen=True)
class DecodedFrame:
    """An unmirrored BGR frame ready for `PipelineRuntimeController.process_frame`.

    The browser never mirrors submitted pixels (game_implementation_plan.md
    invariant 1); this type carries no mirroring flag because mirroring must
    never happen on this path.
    """

    sequence: int
    captured_at_ms: int
    width: int
    height: int
    image: np.ndarray


def decode_image(envelope: FrameEnvelope) -> DecodedFrame:
    """JPEG-decode an already header-validated envelope.

    Split out from decode_binary_frame so the server can check the frame
    sequence guard against the cheap header first and skip the JPEG decode
    entirely for a stale/duplicate frame (context.md step-6: "discard stale
    input sequences before JPEG decoding where possible").
    """
    buffer = np.frombuffer(envelope.payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ProtocolError("failed to decode frame payload as JPEG")
    height, width = image.shape[:2]
    if width != envelope.width or height != envelope.height:
        raise ProtocolError(
            f"declared dimensions {envelope.width}x{envelope.height} do not "
            f"match decoded dimensions {width}x{height}"
        )
    return DecodedFrame(
        sequence=envelope.sequence,
        captured_at_ms=envelope.captured_at_ms,
        width=width,
        height=height,
        image=image,
    )


def decode_binary_frame(data: bytes) -> DecodedFrame:
    """Decode one inbound binary WebSocket message into a DecodedFrame.

    Raises ProtocolError for anything that fails the wire contract or JPEG
    decoding so the caller can reject the frame before submitting it to the
    scheduler. Convenience wrapper around decode_frame + decode_image for
    callers (tests, diagnostics) that don't need the sequence-guard split.
    """
    envelope: FrameEnvelope = decode_frame(data)
    return decode_image(envelope)
