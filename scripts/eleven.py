import logging
import os
from pathlib import Path
import importlib
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
importlib.import_module("eleven_utils")
from eleven_utils import raw_text, paragraphs_from_text

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OUTPUT_DIR = Path("elevenlabs-dataset/fake")

elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

paragraphs = paragraphs_from_text(text=raw_text, words_per_paragraph=13)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
request_ids = []
total_character_cost = 0

for idx, paragraph in enumerate(paragraphs, start=1):
    with elevenlabs.text_to_speech.with_raw_response.convert(
        text=paragraph,
        voice_id=os.getenv("GEORGE_VOICE_ID"),
        model_id="eleven_multilingual_v2",
        previous_request_ids=request_ids[-3:],
        output_format="wav_16000",
    ) as response:
        raw = getattr(response, "_response", response)
        headers = getattr(raw, "headers", getattr(response, "headers", {}))
        request_ids.append(headers.get("request-id") if headers else None)
        audio_data = b"".join(chunk for chunk in response.data)

        char_cost = (headers.get("x-character-count") or headers.get("character-cost")) if headers else None
        if char_cost is not None:
            cost = int(char_cost)
            total_character_cost += cost
            logger.info(
                "Paragraph %s: x-character-count=%s, %s bytes",
                idx,
                cost,
                len(audio_data),
            )
        else:
            input_chars = len(paragraph)
            total_character_cost += input_chars
            logger.info(
                "Paragraph %s: ~%s chars (input), %s bytes",
                idx,
                input_chars,
                len(audio_data),
            )

    out_path = OUTPUT_DIR / f"file{idx:03d}.wav"
    out_path.write_bytes(audio_data)
    logger.info("Wrote %s", out_path)

if total_character_cost and paragraphs:
    logger.info("Total cost: %s (chars)", total_character_cost)
