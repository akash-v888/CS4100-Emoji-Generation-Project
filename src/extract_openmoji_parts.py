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
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


DRAWABLE_TAGS = {
    "path",
    "circle",
    "ellipse",
    "rect",
    "polygon",
    "polyline",
    "line",
}

CONTAINER_TAGS = {"g"}
SKIP_TAGS = {
    "defs",
    "metadata",
    "title",
    "desc",
    "clipPath",
    "mask",
    "linearGradient",
    "radialGradient",
    "style",
    "symbol",
    "use",
}

# Review shortcuts
# e  -> eyes pair
# 1  -> left eye
# 2  -> right eye
# b  -> eyebrows pair
# 3  -> left eyebrow
# 4  -> right eyebrow
# m  -> mouth
# o  -> other
# x  -> reject
# q  -> quit
REVIEW_KEYMAP = {
    "e": ("eyes", "pair"),
    "1": ("eyes", "left"),
    "2": ("eyes", "right"),
    "b": ("eyebrows", "pair"),
    "3": ("eyebrows", "left"),
    "4": ("eyebrows", "right"),
    "m": ("mouths", "center"),
    "o": ("other", "unspecified"),
    "x": ("reject", "unspecified"),
    "q": ("quit", "unspecified"),
}


@dataclass
class CandidateStats:
    source_svg: str
    candidate_id: str
    xml_tag: str
    xml_depth: int
    auto_guess: str

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
    area_ratio: float
    aspect_ratio: float

    mean_r: float
    mean_g: float
    mean_b: float
    dark_ratio: float
    saturated_ratio: float
    opaque_pixel_count: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2)

    @property
    def is_group(self) -> bool:
        return self.xml_tag == "g"


@dataclass
class ReviewLabel:
    keep: bool
    part_type: str
    placement: str
    descriptor: str
    tags: list[str]
    note: str


@dataclass
class CandidateBundle:
    svg_text: str
    full_rgba: np.ndarray
    crop_rgba: np.ndarray
    stats: CandidateStats


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def has_drawable_descendant(node: ET.Element) -> bool:
    for child in node.iter():
        if local_name(child.tag) in DRAWABLE_TAGS:
            return True
    return False


def parse_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    viewbox = root.attrib.get("viewBox")
    if viewbox:
        parts = [float(x) for x in viewbox.replace(",", " ").split()]
        if len(parts) == 4:
            return parts[0], parts[1], parts[2], parts[3]

    width = root.attrib.get("width", "72")
    height = root.attrib.get("height", "72")

    def parse_dim(s: str) -> float:
        s = s.strip().lower().replace("px", "")
        return float(s)

    return 0.0, 0.0, parse_dim(width), parse_dim(height)


def clone_svg_with_node(
    original_root: ET.Element,
    node: ET.Element,
    width: float,
    height: float,
    min_x: float = 0.0,
    min_y: float = 0.0,
) -> str:
    new_root = ET.Element(
        "svg",
        attrib={
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"{min_x} {min_y} {width} {height}",
            "width": str(width),
            "height": str(height),
        },
    )

    for child in original_root:
        if local_name(child.tag) == "defs":
            new_root.append(copy.deepcopy(child))

    new_root.append(copy.deepcopy(node))
    return ET.tostring(new_root, encoding="unicode")


def render_svg_to_rgba(svg_text: str, render_size: int = 512) -> np.ndarray:
    png_bytes = cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        output_width=render_size,
        output_height=render_size,
    )
    return np.array(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))


def bbox_from_alpha(
    rgba: np.ndarray,
    alpha_threshold: int = 8,
) -> Optional[tuple[int, int, int, int]]:
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > alpha_threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_to_bbox(
    rgba: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad: int = 10,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = rgba.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    return rgba[y1:y2 + 1, x1:x2 + 1].copy()


def compute_visual_stats(
    rgba: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> dict:
    h, w = rgba.shape[:2]
    x1, y1, x2, y2 = bbox

    alpha = rgba[:, :, 3]
    mask = alpha > 8
    pixels = rgba[mask]
    rgb = pixels[:, :3].astype(np.float32)

    mean_r, mean_g, mean_b = rgb.mean(axis=0).tolist()
    brightness = rgb.mean(axis=1)
    dark_ratio = float(np.mean(brightness < 80))

    max_rgb = rgb.max(axis=1)
    min_rgb = rgb.min(axis=1)
    saturated_ratio = float(np.mean((max_rgb - min_rgb) > 40))

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    return {
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "bbox_x1_norm": x1 / w,
        "bbox_y1_norm": y1 / h,
        "bbox_x2_norm": x2 / w,
        "bbox_y2_norm": y2 / h,
        "center_x_norm": ((x1 + x2) / 2.0) / w,
        "center_y_norm": ((y1 + y2) / 2.0) / h,
        "width_norm": bw / w,
        "height_norm": bh / h,
        "area_ratio": float(mask.sum() / (h * w)),
        "aspect_ratio": float(bw / max(bh, 1)),
        "mean_r": float(mean_r),
        "mean_g": float(mean_g),
        "mean_b": float(mean_b),
        "dark_ratio": dark_ratio,
        "saturated_ratio": saturated_ratio,
        "opaque_pixel_count": int(mask.sum()),
    }


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 < x1 or y2 < y1:
        return 0
    return (x2 - x1 + 1) * (y2 - y1 + 1)


def containment_ratio(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    inter = intersection_area(inner, outer)
    return inter / max(bbox_area(inner), 1)


def is_likely_full_face(stats: dict) -> bool:
    large = stats["width_norm"] > 0.48 and stats["height_norm"] > 0.48
    centered = 0.25 <= stats["center_x_norm"] <= 0.75 and 0.25 <= stats["center_y_norm"] <= 0.75
    yellowish = stats["mean_r"] > 160 and stats["mean_g"] > 140 and stats["mean_b"] < 140
    return large and centered and yellowish


def predict_type(stats: dict) -> str:
    cy = stats["center_y_norm"]
    w = stats["width_norm"]
    h = stats["height_norm"]
    area = stats["area_ratio"]
    aspect = stats["aspect_ratio"]
    dark = stats["dark_ratio"]
    saturated = stats["saturated_ratio"]

    if is_likely_full_face(stats):
        return "skip"

    # eyebrows
    if (
        0.04 <= cy <= 0.34
        and h < 0.12
        and w < 0.42
        and aspect > 1.15
        and dark > 0.40
    ):
        return "eyebrows"

    # eyes
    if (
        0.10 <= cy <= 0.50
        and h < 0.22
        and w < 0.55
        and area < 0.07
    ):
        return "eyes"

    # mouths
    if (
        0.42 <= cy <= 0.85
        and w > 0.08
        and h < 0.28
        and area < 0.10
    ):
        return "mouths"

    if (
        0.42 <= cy <= 0.85
        and saturated > 0.20
        and w > 0.08
        and h < 0.28
    ):
        return "mouths"

    return "other"


def gather_candidate_nodes(root: ET.Element, max_depth: int = 4) -> list[tuple[ET.Element, int]]:
    """
    Gather both group and leaf candidates.
    Groups come first because they are more likely to represent whole semantic parts.
    """
    groups: list[tuple[ET.Element, int]] = []
    leaves: list[tuple[ET.Element, int]] = []

    def walk(node: ET.Element, depth: int) -> None:
        if depth > max_depth:
            return

        for child in list(node):
            tag = local_name(child.tag)

            if tag in SKIP_TAGS:
                continue

            if tag in CONTAINER_TAGS and has_drawable_descendant(child):
                groups.append((child, depth + 1))

            if tag in DRAWABLE_TAGS:
                leaves.append((child, depth + 1))

            walk(child, depth + 1)

    walk(root, 0)

    # Prefer shallower groups first, then shallower leaves
    groups.sort(key=lambda x: x[1])
    leaves.sort(key=lambda x: x[1])

    return groups + leaves


def dedupe_candidates(candidates: list[CandidateBundle]) -> list[CandidateBundle]:
    seen = set()
    deduped: list[CandidateBundle] = []

    for cand in candidates:
        s = cand.stats
        key = (
            round(s.bbox_x1_norm, 3),
            round(s.bbox_y1_norm, 3),
            round(s.bbox_x2_norm, 3),
            round(s.bbox_y2_norm, 3),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)

    return deduped


def suppress_subparts(candidates: list[CandidateBundle]) -> list[CandidateBundle]:
    """
    Remove tiny/contained pieces when a larger candidate in the same region likely
    represents the full semantic component.

    Example:
    - mouth outline
    - teeth
    - red fill
    - full mouth group

    Keep the full mouth group and suppress the smaller subparts.
    """
    kept: list[CandidateBundle] = []

    # Process larger grouped candidates first
    ordered = sorted(
        candidates,
        key=lambda c: (
            0 if c.stats.is_group else 1,
            -c.stats.area_ratio,
            c.stats.xml_depth,
        ),
    )

    for cand_i in ordered:
        s_i = cand_i.stats
        bbox_i = s_i.bbox

        is_subpart = False

        for cand_j in kept:
            s_j = cand_j.stats
            bbox_j = s_j.bbox

            same_vertical_region = abs(s_i.center_y_norm - s_j.center_y_norm) < 0.12
            mostly_contained = containment_ratio(bbox_i, bbox_j) > 0.88
            parent_is_larger = s_j.area_ratio > s_i.area_ratio * 1.35

            # if a candidate is almost entirely inside a larger kept candidate
            # in the same face region, it is probably a subpart
            if same_vertical_region and mostly_contained and parent_is_larger:
                is_subpart = True
                break

            # if the bbox is nearly identical and the existing one is a group,
            # prefer the group candidate over a leaf candidate
            nearly_same_bbox = (
                abs(s_i.bbox_x1_norm - s_j.bbox_x1_norm) < 0.01
                and abs(s_i.bbox_y1_norm - s_j.bbox_y1_norm) < 0.01
                and abs(s_i.bbox_x2_norm - s_j.bbox_x2_norm) < 0.01
                and abs(s_i.bbox_y2_norm - s_j.bbox_y2_norm) < 0.01
            )
            if nearly_same_bbox and s_j.is_group and not s_i.is_group:
                is_subpart = True
                break

        if not is_subpart:
            kept.append(cand_i)

    return kept


def sanitize_filename(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text or "unlabeled"


def get_review_key(
    original_rgba: np.ndarray,
    candidate_crop_rgba: np.ndarray,
    stats: CandidateStats,
) -> Optional[str]:
    state = {"key": None}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_rgba)
    axes[0].set_title("Original emoji")
    axes[0].axis("off")

    axes[1].imshow(original_rgba)
    rect = Rectangle(
        (stats.bbox_x1, stats.bbox_y1),
        stats.bbox_x2 - stats.bbox_x1,
        stats.bbox_y2 - stats.bbox_y1,
        fill=False,
        edgecolor="red",
        linewidth=2,
    )
    axes[1].add_patch(rect)
    axes[1].set_title(f"Candidate overlay\nAuto guess: {stats.auto_guess}")
    axes[1].axis("off")

    axes[2].imshow(candidate_crop_rgba)
    axes[2].set_title("Candidate crop")
    axes[2].axis("off")

    help_text = (
        "Keys: e=eyes pair | 1=left eye | 2=right eye | "
        "b=brows pair | 3=left brow | 4=right brow | "
        "m=mouth | o=other | x=reject | q=quit"
    )

    fig.suptitle(
        f"{stats.source_svg} | {stats.candidate_id} | "
        f"tag={stats.xml_tag} depth={stats.xml_depth} | "
        f"cx={stats.center_x_norm:.2f}, cy={stats.center_y_norm:.2f}, "
        f"w={stats.width_norm:.2f}, h={stats.height_norm:.2f}"
    )
    fig.text(0.5, 0.02, help_text, ha="center", fontsize=10)

    def on_key(event):
        if event.key is None:
            return
        key = event.key.lower()
        if key in REVIEW_KEYMAP:
            state["key"] = key
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()

    return state["key"]


def prompt_descriptor(default: str = "") -> str:
    raw = input(f"Descriptor [{default}]: ").strip()
    return raw if raw else default


def prompt_tags() -> list[str]:
    raw = input("Tags (comma-separated, optional): ").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def prompt_note() -> str:
    return input("Optional note: ").strip()


def build_review_label(key: str, auto_guess: str) -> ReviewLabel:
    part_type, placement = REVIEW_KEYMAP[key]

    if part_type == "quit":
        raise KeyboardInterrupt("User chose to quit.")
    if part_type == "reject":
        return ReviewLabel(
            keep=False,
            part_type="reject",
            placement="unspecified",
            descriptor="",
            tags=[],
            note=prompt_note(),
        )

    default_descriptor = ""
    if auto_guess in {"eyes", "eyebrows", "mouths"}:
        default_descriptor = auto_guess[:-1] if auto_guess.endswith("s") else auto_guess

    descriptor = prompt_descriptor(default=default_descriptor)
    tags = prompt_tags()
    note = prompt_note()

    return ReviewLabel(
        keep=True,
        part_type=part_type,
        placement=placement,
        descriptor=descriptor,
        tags=tags,
        note=note,
    )


def save_reviewed_candidate(
    output_dir: Path,
    svg_text: str,
    crop_rgba: np.ndarray,
    stats: CandidateStats,
    review: ReviewLabel,
) -> None:
    review_dir = output_dir / "reviewed" / review.part_type
    review_dir.mkdir(parents=True, exist_ok=True)

    safe_descriptor = sanitize_filename(review.descriptor)
    safe_placement = sanitize_filename(review.placement)
    base_name = f"{stats.candidate_id}_{safe_placement}_{safe_descriptor}"

    svg_path = review_dir / f"{base_name}.svg"
    png_path = review_dir / f"{base_name}.png"
    json_path = review_dir / f"{base_name}.json"

    svg_path.write_text(svg_text, encoding="utf-8")
    Image.fromarray(crop_rgba).save(png_path)

    payload = {
        "review": asdict(review),
        "stats": asdict(stats),
        "output_svg": str(svg_path),
        "output_png": str(png_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_svg_file(
    svg_path: Path,
    output_dir: Path,
    render_size: int = 512,
    max_depth: int = 4,
    min_area_ratio: float = 0.001,
) -> None:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    min_x, min_y, width, height = parse_viewbox(root)

    original_svg_text = ET.tostring(root, encoding="unicode")
    original_rgba = render_svg_to_rgba(original_svg_text, render_size=render_size)

    raw_candidates: list[CandidateBundle] = []

    for idx, (node, depth) in enumerate(gather_candidate_nodes(root, max_depth=max_depth)):
        svg_text = clone_svg_with_node(
            original_root=root,
            node=node,
            width=width,
            height=height,
            min_x=min_x,
            min_y=min_y,
        )

        try:
            candidate_full_rgba = render_svg_to_rgba(svg_text, render_size=render_size)
        except Exception:
            continue

        bbox = bbox_from_alpha(candidate_full_rgba)
        if bbox is None:
            continue

        stats_dict = compute_visual_stats(candidate_full_rgba, bbox)
        if stats_dict["area_ratio"] < min_area_ratio:
            continue

        auto_guess = predict_type(stats_dict)
        if auto_guess == "skip":
            continue

        candidate_crop_rgba = crop_to_bbox(candidate_full_rgba, bbox, pad=10)

        stats = CandidateStats(
            source_svg=svg_path.name,
            candidate_id=f"{svg_path.stem}_{idx:03d}",
            xml_tag=local_name(node.tag),
            xml_depth=depth,
            auto_guess=auto_guess,
            bbox_x1=stats_dict["bbox_x1"],
            bbox_y1=stats_dict["bbox_y1"],
            bbox_x2=stats_dict["bbox_x2"],
            bbox_y2=stats_dict["bbox_y2"],
            bbox_x1_norm=stats_dict["bbox_x1_norm"],
            bbox_y1_norm=stats_dict["bbox_y1_norm"],
            bbox_x2_norm=stats_dict["bbox_x2_norm"],
            bbox_y2_norm=stats_dict["bbox_y2_norm"],
            center_x_norm=stats_dict["center_x_norm"],
            center_y_norm=stats_dict["center_y_norm"],
            width_norm=stats_dict["width_norm"],
            height_norm=stats_dict["height_norm"],
            area_ratio=stats_dict["area_ratio"],
            aspect_ratio=stats_dict["aspect_ratio"],
            mean_r=stats_dict["mean_r"],
            mean_g=stats_dict["mean_g"],
            mean_b=stats_dict["mean_b"],
            dark_ratio=stats_dict["dark_ratio"],
            saturated_ratio=stats_dict["saturated_ratio"],
            opaque_pixel_count=stats_dict["opaque_pixel_count"],
        )

        raw_candidates.append(
            CandidateBundle(
                svg_text=svg_text,
                full_rgba=candidate_full_rgba,
                crop_rgba=candidate_crop_rgba,
                stats=stats,
            )
        )

    candidates = dedupe_candidates(raw_candidates)
    candidates = suppress_subparts(candidates)

    # Review useful parts first
    priority = {"mouths": 0, "eyes": 1, "eyebrows": 2, "other": 3}
    candidates.sort(
        key=lambda c: (
            priority.get(c.stats.auto_guess, 9),
            0 if c.stats.is_group else 1,
            c.stats.center_y_norm,
            c.stats.center_x_norm,
        )
    )

    print(f"\n{svg_path.name}: {len(candidates)} candidates after filtering")

    for cand in candidates:
        key = get_review_key(
            original_rgba=original_rgba,
            candidate_crop_rgba=cand.crop_rgba,
            stats=cand.stats,
        )

        if key is None:
            print(f"Window closed without key press: {cand.stats.candidate_id}")
            continue

        review = build_review_label(key, cand.stats.auto_guess)

        if review.keep:
            save_reviewed_candidate(
                output_dir=output_dir,
                svg_text=cand.svg_text,
                crop_rgba=cand.crop_rgba,
                stats=cand.stats,
                review=review,
            )
            print(f"Saved: {cand.stats.candidate_id} -> {review.part_type} ({review.placement})")
        else:
            print(f"Rejected: {cand.stats.candidate_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive OpenMoji part extractor optimized for eyes, eyebrows, and mouths."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing selected OpenMoji face SVGs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets/openmoji_assets",
        help="Directory where reviewed assets are saved.",
    )
    parser.add_argument(
        "--render_size",
        type=int,
        default=512,
        help="Raster size used for preview rendering.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=4,
        help="How deep into SVG groups to search.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of SVG files to process.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    svg_files = sorted(input_dir.glob("*.svg"))
    if args.limit is not None:
        svg_files = svg_files[: args.limit]

    if not svg_files:
        raise FileNotFoundError(f"No SVG files found in {input_dir}")

    for svg_path in svg_files:
        try:
            process_svg_file(
                svg_path=svg_path,
                output_dir=output_dir,
                render_size=args.render_size,
                max_depth=args.max_depth,
            )
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

    print(f"\nDone. Reviewed assets saved to: {output_dir}")


if __name__ == "__main__":
    main()

    
# python src/extract_openmoji_parts.py --input_dir data/openmojis --output_dir outputs/openmoji_parts