"""Audio generation via edge-tts (free Microsoft TTS, no API key needed)."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

AVAILABLE_VOICES = {
    "en-CA-LiamNeural": "English (Canada) — Male, default for legal",
    "en-CA-ClaraNeural": "English (Canada) — Female",
    "en-US-GuyNeural": "English (US) — Male",
}


async def generate_audio(
    text: str,
    voice: str = "en-CA-LiamNeural",
) -> Optional[bytes]:
    """Generate MP3 audio from text using edge-tts.

    Returns MP3 bytes, or None on failure.
    Max input: 5000 characters (truncated).
    """
    if not text or not text.strip():
        return None

    # Truncate to avoid TTS timeout
    if len(text) > 5000:
        text = text[:4997] + "..."

    try:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        audio_chunks = []

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        if audio_chunks:
            return b"".join(audio_chunks)
        else:
            logger.warning("TTS produced no audio output")
            return None

    except ImportError:
        logger.error("edge-tts not installed. Run: pip install edge-tts")
        return None
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        return None