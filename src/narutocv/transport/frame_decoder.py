"""Binary WebSocket frame -> OpenCV BGR image, with the wire contract enforced
before any pixels reach the recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .protocol import (
    MAX_FRAME_HEIGHT,
    MAX_FRAME_WIDTH,
    FrameEnvelope,
    ProtocolError,
    decode_frame,
)

# Start-of-frame markers carry the image dimensions (C4/C8/CC are not SOFs).
_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
# Legitimate JPEGs reach their SOF within a handful of segments (APPn, DQT, ...).
_MAX_SEGMENTS_BEFORE_SOF = 64


def jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    """Return (width, height) from the JPEG's own SOF header without decoding it.

    The WebSocket header's width/height are client-declared, so they cannot be
    trusted to bound `cv2.imdecode`'s allocation: a tiny file can declare
    65535x65535 and make the decoder allocate gigabytes. Walking the marker
    segments by their length fields (rather than scanning for 0xFFC0) skips
    embedded EXIF thumbnails, whose own SOF would otherwise be misread.
    """
    if payload[:2] != b"\xff\xd8":
        raise ProtocolError("frame payload is not a JPEG (missing SOI marker)")
    size = len(payload)
    pos = 2
    for _ in range(_MAX_SEGMENTS_BEFORE_SOF):
        if pos >= size or payload[pos] != 0xFF:
            raise ProtocolError("malformed JPEG: expected a marker")
        while pos < size and payload[pos] == 0xFF:  # fill bytes
            pos += 1
        if pos >= size:
            raise ProtocolError("malformed JPEG: truncated marker")
        marker = payload[pos]
        pos += 1
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:  # standalone markers, no length
            continue
        if marker in (0xD8, 0xD9, 0xDA):  # SOI/EOI/SOS before any SOF
            raise ProtocolError("malformed JPEG: no frame header before image data")
        if pos + 2 > size:
            raise ProtocolError("malformed JPEG: truncated segment")
        length = int.from_bytes(payload[pos : pos + 2], "big")
        if length < 2:
            raise ProtocolError("malformed JPEG: bad segment length")
        if marker in _SOF_MARKERS:
            if length < 8 or pos + 7 > size:
                raise ProtocolError("malformed JPEG: truncated frame header")
            height = int.from_bytes(payload[pos + 3 : pos + 5], "big")
            width = int.from_bytes(payload[pos + 5 : pos + 7], "big")
            return width, height
        pos += length
    raise ProtocolError("malformed JPEG: no frame header found")


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
    entirely for a stale/duplicate frame (docs/archive/context.md step-6: "discard stale
    input sequences before JPEG decoding where possible").
    """
    # Check the JPEG's own dimensions before OpenCV allocates anything: a
    # decompression bomb is rejected here for the cost of a few byte reads.
    jpeg_width, jpeg_height = jpeg_dimensions(envelope.payload)
    if jpeg_width > MAX_FRAME_WIDTH or jpeg_height > MAX_FRAME_HEIGHT:
        raise ProtocolError(
            f"JPEG dimensions {jpeg_width}x{jpeg_height} exceed the "
            f"{MAX_FRAME_WIDTH}x{MAX_FRAME_HEIGHT} limit"
        )
    if jpeg_width != envelope.width or jpeg_height != envelope.height:
        raise ProtocolError(
            f"declared dimensions {envelope.width}x{envelope.height} do not "
            f"match JPEG header dimensions {jpeg_width}x{jpeg_height}"
        )
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
