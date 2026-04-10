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

            # Skip known bad assets
            source = info.get("source_svg", "")
            if part_type == "mouth" and source == "1F601.svg":
                continue  # teeth-only mouth with no tongue fill

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

        # Recolor black outline to a darker shade of the skin tone
        outline_rgb = tuple(max(0, int(c * 0.55)) for c in skin_tone_rgb)
        is_dark = (r < 60) & (g < 60) & (b < 60) & (a > 128)
        arr[is_dark, 0] = outline_rgb[0]
        arr[is_dark, 1] = outline_rgb[1]
        arr[is_dark, 2] = outline_rgb[2]

        recolored = Image.fromarray(arr.astype(np.uint8), "RGBA")

        canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
        canvas.paste(recolored, (x1, y1), recolored)
        return canvas

    def _paste_part(self, canvas: Image.Image, asset: dict,
                    outline_rgb: tuple[int, int, int] | None = None) -> Image.Image:
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

            # Recolor black strokes to match skin tone outline
            if outline_rgb is not None:
                arr = np.array(part_img, dtype=np.float32)
                r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
                is_dark = (r < 60) & (g < 60) & (b < 60) & (a > 128)

                # Detect red/tongue pixels and find dark pixels adjacent to them
                is_red = (r > 150) & (g < 100) & (b < 100) & (a > 128)
                if is_red.any():
                    from scipy.ndimage import binary_dilation
                    near_red = binary_dilation(is_red, iterations=3)
                    is_separator = is_dark & near_red
                    is_outline = is_dark & ~near_red
                    # Separator: darker shade of the tongue color
                    tongue_r = float(r[is_red].mean())
                    tongue_g = float(g[is_red].mean())
                    tongue_b = float(b[is_red].mean())
                    dark_tongue = (max(0, int(tongue_r * 0.55)),
                                   max(0, int(tongue_g * 0.55)),
                                   max(0, int(tongue_b * 0.55)))
                    arr[is_separator, 0] = dark_tongue[0]
                    arr[is_separator, 1] = dark_tongue[1]
                    arr[is_separator, 2] = dark_tongue[2]
                    arr[is_outline, 0] = outline_rgb[0]
                    arr[is_outline, 1] = outline_rgb[1]
                    arr[is_outline, 2] = outline_rgb[2]
                else:
                    arr[is_dark, 0] = outline_rgb[0]
                    arr[is_dark, 1] = outline_rgb[1]
                    arr[is_dark, 2] = outline_rgb[2]

                part_img = Image.fromarray(arr.astype(np.uint8), "RGBA")

            canvas.paste(part_img, (x1, y1), part_img)

        return canvas

    def compose(self, components: dict, skin_tone_rgb: tuple[int, int, int] = (255, 224, 189)) -> Image.Image:
        """
        Compose an emoji from component selections.

        components: {"eye_type": "round", "mouth_type": "smile", "brow_type": "raised",
                     "hair_style": "short"/"long"/"none", "hair_color_rgb": (r,g,b) or None}
        skin_tone_rgb: RGB tuple for face fill color
        """
        hair_style = components.get("hair_style", "none")
        hair_color = components.get("hair_color_rgb")

        canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))

        # 1. Long hair behind face (side curtains)
        if hair_style == "long" and hair_color:
            behind = self._draw_hair_long_behind(hair_color)
            canvas.paste(behind, (0, 0), behind)

        # 2. Face circle
        face = self._make_face(skin_tone_rgb)
        canvas.paste(face, (0, 0), face)

        # 3. Facial features
        outline_rgb = tuple(max(0, int(c * 0.55)) for c in skin_tone_rgb)
        brow_color = hair_color if hair_color else outline_rgb

        for part_type, descriptor in [
            ("mouth", components.get("mouth_type", "smile")),
            ("eyes", components.get("eye_type", "round")),
            ("eyebrows", components.get("brow_type", "flat")),
        ]:
            asset = self._pick_asset(part_type, descriptor)
            if asset:
                color = brow_color if part_type == "eyebrows" else outline_rgb
                canvas = self._paste_part(canvas, asset, outline_rgb=color)

        # 4. Hair on top
        if hair_color and hair_style != "none":
            if hair_style == "short":
                hair_layer = self._draw_hair_short(hair_color)
            else:
                hair_layer = self._draw_hair_long_top(hair_color)
            canvas.paste(hair_layer, (0, 0), hair_layer)

        return canvas

    # Normalized contour traced from hair asset (0-1 range, relative to bounding box)
    HAIR_SHORT_CONTOUR = [
        (0.1354, 0.3542), (0.0833, 0.5000), (0.0938, 0.8021), (0.1354, 0.7917),
        (0.1458, 0.6250), (0.2083, 0.5729), (0.2500, 0.4688), (0.4375, 0.5521),
        (0.5938, 0.5521), (0.7292, 0.5104), (0.8333, 0.6146), (0.8646, 0.8021),
        (0.8958, 0.7917), (0.8958, 0.4479), (0.8438, 0.3333), (0.7604, 0.2500),
        (0.6042, 0.1875), (0.3854, 0.1875), (0.2292, 0.2500),
    ]

    def _draw_hair_short(self, hair_color: tuple[int, int, int]) -> Image.Image:
        """Draw short hair from traced contour — crisp at any resolution."""
        canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        fill = (*hair_color, 255)

        cx = CANVAS_SIZE // 2
        face_top = int(CANVAS_SIZE * 0.166)
        face_r = int(CANVAS_SIZE * 0.333)

        # Bounding box for the hair: wider than face, positioned on top
        hair_w = int(face_r * 2.4)
        hair_h = int(hair_w * 0.85)
        hair_x = cx - hair_w // 2
        hair_y = face_top - int(hair_h * 0.33)

        # Scale contour points to canvas coordinates
        points = [(int(hair_x + px * hair_w), int(hair_y + py * hair_h))
                  for px, py in self.HAIR_SHORT_CONTOUR]

        draw.polygon(points, fill=fill)
        return canvas

    def _draw_hair_long_top(self, hair_color: tuple[int, int, int]) -> Image.Image:
        """Draw long hair dome on top of head (overlays face)."""
        canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        cx = CANVAS_SIZE // 2
        fill = (*hair_color, 255)

        face_top = int(CANVAS_SIZE * 0.166)
        face_r = int(CANVAS_SIZE * 0.333)
        pad = int(CANVAS_SIZE * 0.04)

        # Dome on top
        draw.ellipse([cx - face_r - pad, face_top - pad,
                       cx + face_r + pad, face_top + face_r],
                      fill=fill)

        # Erase below the parting line
        part_y = face_top + int(face_r * 0.45)
        draw.rectangle([0, part_y, CANVAS_SIZE, CANVAS_SIZE], fill=(0, 0, 0, 0))

        return canvas

    def _draw_hair_long_behind(self, hair_color: tuple[int, int, int]) -> Image.Image:
        """Draw long hair side curtains behind the face."""
        canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        cx = CANVAS_SIZE // 2
        fill = (*hair_color, 255)

        face_top = int(CANVAS_SIZE * 0.166)
        face_r = int(CANVAS_SIZE * 0.333)
        pad = int(CANVAS_SIZE * 0.04)

        # Side curtains
        curtain_w = int(CANVAS_SIZE * 0.11)
        curtain_top = face_top + int(face_r * 0.2)
        curtain_bottom = int(CANVAS_SIZE * 0.90)

        # Left
        draw.rounded_rectangle(
            [cx - face_r - pad, curtain_top,
             cx - face_r + curtain_w, curtain_bottom],
            radius=int(CANVAS_SIZE * 0.05), fill=fill)

        # Right
        draw.rounded_rectangle(
            [cx + face_r - curtain_w, curtain_top,
             cx + face_r + pad, curtain_bottom],
            radius=int(CANVAS_SIZE * 0.05), fill=fill)

        return canvas



def main() -> None:
    parser = argparse.ArgumentParser(description="Compose an emoji from parts.")
    parser.add_argument("--eye_type", default="round")
    parser.add_argument("--mouth_type", default="smile")
    parser.add_argument("--brow_type", default="flat")
    parser.add_argument("--skin_tone", default="255,224,189", help="R,G,B values")
    parser.add_argument("--hair_style", default="none", choices=["none", "short", "long"])
    parser.add_argument("--hair_color", default=None, help="R,G,B values for hair")
    parser.add_argument("--assets_dir", default=str(DEFAULT_ASSETS_DIR))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "composed_emoji.png"))
    args = parser.parse_args()

    skin_rgb = tuple(int(v) for v in args.skin_tone.split(","))
    hair_rgb = tuple(int(v) for v in args.hair_color.split(",")) if args.hair_color else None

    composer = EmojiComposer(args.assets_dir)
    components = {
        "eye_type": args.eye_type,
        "mouth_type": args.mouth_type,
        "brow_type": args.brow_type,
        "hair_style": args.hair_style,
        "hair_color_rgb": hair_rgb,
    }

    emoji_img = composer.compose(components, skin_rgb)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    emoji_img.save(args.output)
    print(f"Saved: {args.output}")
    print(f"Components: {components}")
    print(f"Skin tone: {skin_rgb}")


if __name__ == "__main__":
    main()
