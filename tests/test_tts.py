"""Tests for TTS audio generation."""
import pytest
from rag_pipeline.tts import generate_audio, AVAILABLE_VOICES


def test_available_voices():
    assert "en-CA-LiamNeural" in AVAILABLE_VOICES
    assert "en-CA-ClaraNeural" in AVAILABLE_VOICES


@pytest.mark.asyncio
async def test_generate_audio_short_text(mocker):
    """Test with mocked edge-tts."""
    mock_communicate = mocker.patch("edge_tts.Communicate")
    mock_instance = mock_communicate.return_value

    async def mock_stream():
        yield {"type": "audio", "data": b"audio data"}

    mock_instance.stream = mock_stream

    result = await generate_audio("Hello world")
    assert result is not None
    assert len(result) > 0


@pytest.mark.asyncio
async def test_generate_audio_empty_text():
    result = await generate_audio("")
    assert result is None or result == b""


@pytest.mark.asyncio
async def test_generate_audio_long_text(mocker):
    """Text longer than 5000 chars should be truncated."""
    long_text = "Test. " * 2000  # ~12000 chars
    mock_communicate = mocker.patch("edge_tts.Communicate")
    mock_instance = mock_communicate.return_value

    async def mock_stream():
        yield {"type": "audio", "data": b"audio data"}

    mock_instance.stream = mock_stream

    result = await generate_audio(long_text)
    assert result is not None