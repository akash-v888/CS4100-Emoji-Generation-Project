"""
Gradio web UI for emoji generation.

Usage:
    python src/demo.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import gradio as gr

# Ensure src/ is on the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import generate_emoji


def process_image(image_path: str) -> tuple:
    if image_path is None:
        return None, "Please upload a face photo."

    emoji_img, info = generate_emoji(image_path)

    # Save to temp file for Gradio
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    emoji_img.save(tmp.name)

    details = "**Predicted Components:**\n"
    for k, v in info["predictions"].items():
        details += f"- {k}: `{v}`\n"
    details += f"\n**Skin Tone:** {info['skin_tone_id']} (RGB {info['skin_tone_rgb']})\n"
    details += "\n**Key Features:**\n"
    for k, v in info["features"].items():
        details += f"- {k}: `{v:.4f}`\n"

    return tmp.name, details


demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath", label="Upload a face photo"),
    outputs=[
        gr.Image(type="filepath", label="Generated Emoji"),
        gr.Markdown(label="Details"),
    ],
    title="Emoji Generator",
    description="Upload a face photo to generate a personalized emoji. The pipeline detects facial features, classifies them, and composites matching OpenMoji parts.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
