"""
Emoji composer: assembles face shape + eyes + eyebrows + mouth into a final emoji.

Usage:
    python src/compose.py --eye_type round --mouth_type smile --brow_type raised --face_shape oval --skin_tone 255,224,189
    python src/compose.py --help
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "outputs" / "openmoji_parts" / "reviewed"
CANVAS_SIZE = 512


class EmojiComposer:
    def __init__(self, assets_dir: str | Path = DEFAULT_ASSETS_DIR):
        self.assets_dir = Path(assets_dir)
        self.registry = self._build_registry()

    def _build_registry(self) -> dict[tuple[str, str], dict]:
        """Scan asset JSONs and build (part_type, descriptor) -> metadata mapping."""
        registry: dict[tuple[str, str], list[dict]] = {}
        for json_path in self.assets_dir.rglob("*.json"):
            with open(json_path) as f:
                meta = json.load(f)

            # Handle both new format (part_type/descriptor at top) and legacy (review/stats)
            if "part_type" in meta:
                part_type = meta["part_type"]
                descriptor = meta["descriptor"]
                info = meta["info"]
            elif "review" in meta:
                part_type = meta["review"].get("part_type", "other")
                descriptor = meta["review"].get("descriptor", "unknown")
                info = meta["stats"]
            else:
                continue

            # Normalize "mouths" -> "mouth"
            if part_type == "mouths":
                part_type = "mouth"

            key = (part_type, descriptor)
            entry = {
                "png_path": str(json_path.with_suffix(".png")),
                "svg_path": str(json_path.with_suffix(".svg")),
                "info": info,
            }
            registry.setdefault(key, []).append(entry)
        return registry

    def _pick_asset(self, part_type: str, descriptor: str) -> dict | None:
        """Find a matching asset, falling back to any asset of that part type."""
        # Exact match
        candidates = self.registry.get((part_type, descriptor))
        if candidates:
            return random.choice(candidates)

        # Fallback: any asset of this part type
        fallbacks = []
        for (pt, _), entries in self.registry.items():
            if pt == part_type:
                fallbacks.extend(entries)
        if fallbacks:
            return random.choice(fallbacks)

        return None

    def _make_face(self, skin_tone_rgb: tuple[int, int, int]) -> Image.Image:
        """Load an OpenMoji circle face asset and recolor yellow fill to skin tone."""
        face_asset = self._pick_asset("face", "round")
        if face_asset is None:
            # Fallback: plain circle
            canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            r = int(CANVAS_SIZE * 0.45)
            cx, cy = CANVAS_SIZE // 2, CANVAS_SIZE // 2
            fill = (*skin_tone_rgb, 255)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=(0, 0, 0, 255), width=3)
            return canvas

        face_img = Image.open(face_asset["png_path"]).convert("RGBA")
        info = face_asset["info"]

        # Resize to fill canvas at the asset's original position
        x1 = int(info["bbox_x1_norm"] * CANVAS_SIZE)
        y1 = int(info["bbox_y1_norm"] * CANVAS_SIZE)
        x2 = int(info["bbox_x2_norm"] * CANVAS_SIZE)
        y2 = int(info["bbox_y2_norm"] * CANVAS_SIZE)
        face_img = face_img.resize((x2 - x1, y2 - y1), Image.LANCZOS)

        # Recolor: replace yellow-ish pixels with skin tone, keep dark outline
        arr = np.array(face_img, dtype=np.float32)
        r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]

        # Yellow pixels: high R, high G, low B, opaque
        is_yellow = (r > 180) & (g > 180) & (b < 120) & (a > 128)

        arr[is_yellow, 0] = skin_tone_rgb[0]
        arr[is_yellow, 1] = skin_tone_rgb[1]
        arr[is_yellow, 2] = skin_tone_rgb[2]

        recolored = Image.fromarray(arr.astype(np.uint8), "RGBA")

        canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
        canvas.paste(recolored, (x1, y1), recolored)
        return canvas

    def _paste_part(self, canvas: Image.Image, asset: dict) -> Image.Image:
        """Paste an asset PNG onto the canvas using its normalized position."""
        info = asset["info"]
        part_img = Image.open(asset["png_path"]).convert("RGBA")

        # Target position and size from normalized coords
        x1 = int(info["bbox_x1_norm"] * CANVAS_SIZE)
        y1 = int(info["bbox_y1_norm"] * CANVAS_SIZE)
        x2 = int(info["bbox_x2_norm"] * CANVAS_SIZE)
        y2 = int(info["bbox_y2_norm"] * CANVAS_SIZE)
        target_w = x2 - x1
        target_h = y2 - y1

        if target_w > 0 and target_h > 0:
            part_img = part_img.resize((target_w, target_h), Image.LANCZOS)
            canvas.paste(part_img, (x1, y1), part_img)

        return canvas

    def compose(self, components: dict, skin_tone_rgb: tuple[int, int, int] = (255, 224, 189)) -> Image.Image:
        """
        Compose an emoji from component selections.

        components: {"face_shape": "oval", "eye_type": "round", "mouth_type": "smile", "brow_type": "raised", ...}
        skin_tone_rgb: RGB tuple for face fill color
        """
        # 1. Face circle recolored to skin tone
        canvas = self._make_face(skin_tone_rgb)

        # 2. Layer parts in order: mouth, eyes, eyebrows (back to front)
        part_order = [
            ("mouth", components.get("mouth_type", "smile")),
            ("eyes", components.get("eye_type", "round")),
            ("eyebrows", components.get("brow_type", "flat")),
        ]

        for part_type, descriptor in part_order:
            asset = self._pick_asset(part_type, descriptor)
            if asset:
                canvas = self._paste_part(canvas, asset)

        return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose an emoji from parts.")
    parser.add_argument("--eye_type", default="round")
    parser.add_argument("--mouth_type", default="smile")
    parser.add_argument("--brow_type", default="flat")
    parser.add_argument("--skin_tone", default="255,224,189", help="R,G,B values")
    parser.add_argument("--assets_dir", default=str(DEFAULT_ASSETS_DIR))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "composed_emoji.png"))
    args = parser.parse_args()

    skin_rgb = tuple(int(v) for v in args.skin_tone.split(","))

    composer = EmojiComposer(args.assets_dir)
    components = {
        "eye_type": args.eye_type,
        "mouth_type": args.mouth_type,
        "brow_type": args.brow_type,
    }

    emoji_img = composer.compose(components, skin_rgb)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    emoji_img.save(args.output)
    print(f"Saved: {args.output}")
    print(f"Components: {components}")
    print(f"Skin tone: {skin_rgb}")


if __name__ == "__main__":
    main()
