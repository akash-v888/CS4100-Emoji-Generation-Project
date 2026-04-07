"""
Interactive OpenMoji part extractor.

Shows all SVG elements numbered in a grid. You type which numbers belong
to each feature (eyes, eyebrows, mouth). Selected elements are composited
and saved as SVG + PNG + JSON.

Usage:
    python src/extract_openmoji_parts.py --input_dir data/openmojis --output_dir outputs/openmoji_parts
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cairosvg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ── SVG constants ──────────────────────────────────────────────────────────

DRAWABLE_TAGS = {"path", "circle", "ellipse", "rect", "polygon", "polyline", "line"}
CONTAINER_TAGS = {"g"}
SKIP_TAGS = {
    "defs", "metadata", "title", "desc", "clipPath", "mask",
    "linearGradient", "radialGradient", "style", "symbol", "use",
}

PART_TYPES = ["eyes", "eyebrows", "mouth", "nose", "face", "other"]


# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class PartInfo:
    part_type: str
    descriptor: str
    source_svg: str
    element_indices: list[int]
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    bbox_x1_norm: float
    bbox_y1_norm: float
    bbox_x2_norm: float
    bbox_y2_norm: float
    center_x_norm: float
    center_y_norm: float
    width_norm: float
    height_norm: float


# ── SVG helpers ────────────────────────────────────────────────────────────

def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    viewbox = root.attrib.get("viewBox")
    if viewbox:
        parts = [float(x) for x in viewbox.replace(",", " ").split()]
        if len(parts) == 4:
            return parts[0], parts[1], parts[2], parts[3]
    w = root.attrib.get("width", "72")
    h = root.attrib.get("height", "72")
    return 0.0, 0.0, float(w.replace("px", "")), float(h.replace("px", ""))


def render_svg_to_rgba(svg_text: str, render_size: int = 512) -> np.ndarray:
    png_bytes = cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        output_width=render_size,
        output_height=render_size,
    )
    return np.array(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))


def bbox_from_alpha(rgba: np.ndarray, threshold: int = 8) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(rgba[:, :, 3] > threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_to_bbox(rgba: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 10) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = rgba.shape[:2]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
    return rgba[y1:y2 + 1, x1:x2 + 1].copy()


def clone_svg_with_nodes(
    original_root: ET.Element,
    nodes: list[ET.Element],
    width: float,
    height: float,
    min_x: float = 0.0,
    min_y: float = 0.0,
) -> str:
    new_root = ET.Element("svg", attrib={
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": f"{min_x} {min_y} {width} {height}",
        "width": str(width),
        "height": str(height),
    })
    for child in original_root:
        if local_name(child.tag) == "defs":
            new_root.append(copy.deepcopy(child))
    for node in nodes:
        new_root.append(copy.deepcopy(node))
    return ET.tostring(new_root, encoding="unicode")


def sanitize_filename(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text or "unlabeled"


# ── Gather all leaf drawable elements ──────────────────────────────────────

def gather_all_elements(root: ET.Element, max_depth: int = 6) -> list[ET.Element]:
    """Collect every drawable leaf element from the SVG, across all layers."""
    elements: list[ET.Element] = []

    def walk(node: ET.Element, depth: int) -> None:
        if depth > max_depth:
            return
        for child in list(node):
            tag = local_name(child.tag)
            if tag in SKIP_TAGS:
                continue
            if tag in DRAWABLE_TAGS:
                elements.append(child)
            if tag in CONTAINER_TAGS:
                walk(child, depth + 1)

    walk(root, 0)
    return elements


# ── Render & display ───────────────────────────────────────────────────────

def render_element(
    root: ET.Element,
    node: ET.Element,
    width: float,
    height: float,
    min_x: float,
    min_y: float,
    render_size: int,
) -> Optional[np.ndarray]:
    svg_text = clone_svg_with_nodes(root, [node], width, height, min_x, min_y)
    try:
        rgba = render_svg_to_rgba(svg_text, render_size=render_size)
    except Exception:
        return None
    bbox = bbox_from_alpha(rgba)
    if bbox is None:
        return None
    # Check if it has any meaningful content
    if rgba[:, :, 3].sum() < 100:
        return None
    return rgba


def show_numbered_elements(
    original_rgba: np.ndarray,
    element_renders: list[Optional[np.ndarray]],
    svg_name: str,
) -> None:
    """Show the original emoji and all elements in a numbered grid."""
    # Filter to only elements that rendered successfully
    valid = [(i, r) for i, r in enumerate(element_renders) if r is not None]
    if not valid:
        print(f"  No drawable elements found in {svg_name}")
        return

    n = len(valid)
    # Layout: original on left, then element grid
    cols = min(6, n + 1)
    rows = (n // (cols - 1)) + (1 if n % (cols - 1) else 0)
    rows = max(rows, 1)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    # Turn off all axes first
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # Show original in top-left
    axes[0, 0].imshow(original_rgba)
    axes[0, 0].set_title("Original", fontsize=10)

    # Show each element numbered
    idx_in_grid = 0
    for i, rgba in valid:
        row = idx_in_grid // (cols - 1)
        col = (idx_in_grid % (cols - 1)) + 1
        if row < rows and col < cols:
            axes[row, col].imshow(rgba)
            axes[row, col].set_title(f"#{i}", fontsize=10, fontweight="bold")
        idx_in_grid += 1

    fig.suptitle(f"{svg_name} — type element numbers to select", fontsize=12)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def composite_elements(renders: list[np.ndarray]) -> np.ndarray:
    """Alpha-composite multiple RGBA renders together."""
    result = np.zeros_like(renders[0], dtype=np.float64)
    for layer in renders:
        fg = layer.astype(np.float64)
        fg_a = fg[:, :, 3:4] / 255.0
        bg_a = result[:, :, 3:4] / 255.0
        out_a = fg_a + bg_a * (1 - fg_a)
        safe_a = np.where(out_a > 0, out_a, 1.0)
        result[:, :, :3] = (fg[:, :, :3] * fg_a + result[:, :, :3] * bg_a * (1 - fg_a)) / safe_a
        result[:, :, 3:4] = out_a * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Interactive selection ──────────────────────────────────────────────────

def parse_number_list(text: str) -> list[int]:
    """Parse '1,3,5' or '1 3 5' or '1-3,5' into a list of ints."""
    nums: list[int] = []
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                nums.extend(range(int(lo), int(hi) + 1))
            except ValueError:
                pass
        else:
            try:
                nums.append(int(part))
            except ValueError:
                pass
    return nums


def prompt_part_selection(
    part_type: str,
    valid_indices: list[int],
) -> list[int]:
    """Ask user to select element numbers for a part type. Returns indices."""
    while True:
        raw = input(f"  {part_type} — enter element #s (e.g. 2,5,7 or 2-5), or ENTER to skip: ").strip()
        if not raw:
            return []
        nums = parse_number_list(raw)
        invalid = [n for n in nums if n not in valid_indices]
        if invalid:
            print(f"    Invalid element(s): {invalid}. Valid: {valid_indices}")
            continue
        return nums


def prompt_descriptor(part_type: str) -> str:
    raw = input(f"    Descriptor for {part_type} (e.g. smile, wide, round): ").strip()
    return raw if raw else part_type


# ── Save ───────────────────────────────────────────────────────────────────

def save_part(
    output_dir: Path,
    part_type: str,
    descriptor: str,
    svg_text: str,
    crop_rgba: np.ndarray,
    info: PartInfo,
) -> None:
    part_dir = output_dir / "reviewed" / part_type
    part_dir.mkdir(parents=True, exist_ok=True)

    safe_desc = sanitize_filename(descriptor)
    base = f"{Path(info.source_svg).stem}_{safe_desc}"

    svg_path = part_dir / f"{base}.svg"
    png_path = part_dir / f"{base}.png"
    json_path = part_dir / f"{base}.json"

    svg_path.write_text(svg_text, encoding="utf-8")
    Image.fromarray(crop_rgba).save(png_path)

    payload = {
        "part_type": part_type,
        "descriptor": descriptor,
        "info": asdict(info),
        "output_svg": str(svg_path),
        "output_png": str(png_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── Main processing ───────────────────────────────────────────────────────

def process_svg_file(
    svg_path: Path,
    output_dir: Path,
    render_size: int = 512,
) -> None:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    min_x, min_y, width, height = parse_viewbox(root)

    # Render the full original emoji
    original_svg = ET.tostring(root, encoding="unicode")
    original_rgba = render_svg_to_rgba(original_svg, render_size=render_size)

    # Gather and render all drawable elements
    elements = gather_all_elements(root)
    if not elements:
        print(f"  No elements found in {svg_path.name}")
        return

    print(f"\n{'=' * 60}")
    print(f"  {svg_path.name}: {len(elements)} drawable elements")
    print(f"{'=' * 60}")

    element_renders: list[Optional[np.ndarray]] = []
    for elem in elements:
        element_renders.append(
            render_element(root, elem, width, height, min_x, min_y, render_size)
        )

    valid_indices = [i for i, r in enumerate(element_renders) if r is not None]
    if not valid_indices:
        print(f"  No renderable elements in {svg_path.name}")
        return

    # Show the numbered grid
    show_numbered_elements(original_rgba, element_renders, svg_path.name)

    print()
    print(f"  Valid element numbers: {valid_indices}")
    print(f"  Select elements for each part (or ENTER to skip):")
    print()

    # Prompt for each part type
    for part_type in PART_TYPES:
        indices = prompt_part_selection(part_type, valid_indices)
        if not indices:
            continue

        descriptor = prompt_descriptor(part_type)

        # Composite selected elements
        selected_renders = [element_renders[i] for i in indices if element_renders[i] is not None]
        if not selected_renders:
            print(f"    No valid renders for selection, skipping.")
            continue

        composite = composite_elements(selected_renders)
        bbox = bbox_from_alpha(composite)
        if bbox is None:
            print(f"    Empty composite, skipping.")
            continue

        crop = crop_to_bbox(composite, bbox, pad=10)
        h, w = composite.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1 + 1, y2 - y1 + 1

        # Build merged SVG
        selected_nodes = [elements[i] for i in indices]
        merged_svg = clone_svg_with_nodes(root, selected_nodes, width, height, min_x, min_y)

        info = PartInfo(
            part_type=part_type,
            descriptor=descriptor,
            source_svg=svg_path.name,
            element_indices=indices,
            bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
            bbox_x1_norm=x1 / w, bbox_y1_norm=y1 / h,
            bbox_x2_norm=x2 / w, bbox_y2_norm=y2 / h,
            center_x_norm=((x1 + x2) / 2.0) / w,
            center_y_norm=((y1 + y2) / 2.0) / h,
            width_norm=bw / w, height_norm=bh / h,
        )

        save_part(output_dir, part_type, descriptor, merged_svg, crop, info)
        print(f"    Saved {part_type}/{descriptor} (elements {indices})")

    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive OpenMoji part extractor. Shows numbered elements, you pick which belong to each feature."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with OpenMoji SVGs.")
    parser.add_argument("--output_dir", type=str, default="outputs/openmoji_parts", help="Output directory.")
    parser.add_argument("--render_size", type=int, default=512, help="Render resolution.")
    parser.add_argument("--limit", type=int, default=None, help="Max SVGs to process.")
    parser.add_argument("--filter", type=str, nargs="+", default=None,
                        help="Only process these emoji codes (e.g. 1F61B 1F60F). Deletes old reviewed files first.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    svg_files = sorted(input_dir.glob("*.svg"))
    if args.filter:
        codes = [c.upper() for c in args.filter]
        svg_files = [f for f in svg_files if f.stem.upper() in codes]
        # Remove old reviewed files for filtered emojis
        reviewed_dir = output_dir / "reviewed"
        if reviewed_dir.exists():
            for code in codes:
                for old_file in reviewed_dir.rglob(f"{code}*"):
                    old_file.unlink()
                    print(f"Removed old file: {old_file}")
    if args.limit is not None:
        svg_files = svg_files[:args.limit]

    if not svg_files:
        raise FileNotFoundError(f"No SVG files in {input_dir}")

    for svg_path in svg_files:
        try:
            process_svg_file(svg_path, output_dir, render_size=args.render_size)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

    print(f"\nDone. Assets saved to: {output_dir}")


if __name__ == "__main__":
    main()
