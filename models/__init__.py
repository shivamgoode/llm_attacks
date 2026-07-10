"""models/__init__.py — Exports the list of enabled model clients."""
from models.sarvam import SarvamModel
from models.groq_model import GroqModel
import config

# Add or remove models here to control which are included in benchmarks
MODELS = [
    SarvamModel(),
    GroqModel(config.GROQ_MODEL_1, "Groq-LLaMA3.3-70B"),
    GroqModel(config.GROQ_MODEL_2, "Groq-LLaMA3.1-8B"),
]
