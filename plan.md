# CS4100 Emoji Generation Project - Implementation Plan

## Context

CS4100 (Machine Learning) course project: automate personalized emoji creation from a face photo. The pipeline detects facial landmarks, extracts geometric features, classifies them into emoji component categories, and composites selected OpenMoji parts into a final emoji.

**What exists (~35% done):**
- `src/landmarking.py` (168 lines) - MediaPipe face detection, 478 landmarks, bbox, face crop. Uses `FaceProcessor` class with context manager. Returns `FaceResult` dataclass. **Complete.**
- `src/features.py` (46 lines) - 9 geometric features via `compute_basic_features()`: face_width, face_height, face_ratio, mouth_width, mouth_open, mouth_width_ratio, mouth_open_ratio, left_eye_open, right_eye_open. Note: face_width/face_height/mouth_width/mouth_open are absolute pixel values (not ratios). **Complete but needs expansion.**
- `src/skin_tone.py` (210 lines) - `estimate_skin_tone()` samples cheeks via circular masks, converts BGR→LAB, matches to closest of 5 palette tones. `SkinToneResult` dataclass. `draw_skin_tone_debug()` for visualization. TODO on line 9: calibrate palette to actual asset colors. **Complete, needs palette calibration.**
- `src/test_face.py` (59 lines) - Integration test: runs landmarking → features → skin tone on `sample.jpg`, saves debug images. **Complete.**
- `src/extract_openmoji_parts.py` (800 lines) - Interactive SVG part extractor. Key features:
  - `CandidateBundle` dataclass bundles SVG text, RGBA renders, and `CandidateStats`
  - `gather_candidate_nodes()` collects groups (preferred) then leaves, sorted by depth
  - `suppress_subparts()` removes small pieces contained in larger semantic components (e.g., keeps whole mouth group, drops individual teeth/outline)
  - `predict_type()` auto-classifies candidates by position/size/color into eyes/eyebrows/mouths/other/skip
  - `is_likely_full_face()` filters out large yellowish centered elements (face circles)
  - Interactive matplotlib review with keyboard shortcuts (e/1/2/b/3/4/m/o/x/q)
  - Saves reviewed parts as SVG + PNG + JSON to `<output_dir>/reviewed/<part_type>/`
  - **Complete.**
- `data/openmojis/` - 21 OpenMoji face SVG files (72x72 viewbox): 1F600-1F644, 2639
- `models/face_landmarker.task` - MediaPipe model
- `requirements.txt` - 25 pinned dependencies (mediapipe 0.10.33, opencv 4.13, cairosvg 2.9, matplotlib 3.10, pillow 12.1, numpy 2.4)
- `data/raw/sample.jpg` - 1 test image
- `data/labels/manual_labels.csv` - headers (image_name,face_shape,eye_type,mouth_type,brow_type,nose_type,skin_tone) + 1 incomplete row

**Extracted assets (complete):**
- `outputs/openmoji_parts/reviewed/eyes/` - 21 eye variants (round, squint, wink, wide, closed, eye_roll, excited, etc.)
- `outputs/openmoji_parts/reviewed/eyebrows/` - 4 eyebrow variants (raised, sad, mad)
- `outputs/openmoji_parts/reviewed/mouth/` - 21 mouth variants (smile, tongue_out, smirk, straight, sad, kiss, frown, wide, open, closed, slight_smile, slight_frown)
- `outputs/openmoji_parts/reviewed/mouths/` - 6 legacy mouth subparts from early extraction (1F600, 1F601)
- `outputs/openmoji_parts/reviewed/face/` - 21 face variants
- Each part has `.svg`, `.png`, `.json` (with `review` labels + `stats` including normalized bbox positions)
- All 21 emojis processed.

**What's missing:** expanded features, training data, ML training/comparison, emoji composition, end-to-end pipeline, demo UI.

---

## Phase 1: Finish the Emoji Asset Library (Manual Step)

### 1a. Continue running extractor on remaining emojis
Asset extraction is IN PROGRESS — 5 mouths extracted from 1F600 and 1F601, but 19 emojis remain unprocessed and no eyes/eyebrows have been saved yet.

```
python src/extract_openmoji_parts.py --input_dir data/openmojis --output_dir outputs/openmoji_parts
```

**Result**: All 21 emojis processed. Asset counts:
- Eyes: 21 variants (round, squint, wink, wide, closed, eye_roll, excited)
- Eyebrows: 4 variants (raised, sad, mad)
- Mouths: 21 variants in `mouth/` (smile, tongue_out, smirk, straight, sad, kiss, frown, wide, open, closed, slight_smile, slight_frown) + 6 legacy subparts in `mouths/`
- Faces: 21 variants

**Note**: `--filter` flag was added to re-run extraction on specific emojis (e.g. `--filter 1F61B 1F60F`).

---

## Phase 2: Expand Feature Extraction

**Modify**: `src/features.py`

Add `compute_all_features(landmarks)` returning ~25 scale-invariant ratio features. Keep existing `compute_basic_features` for backward compatibility but the new function is what training/inference will use.

**Note on existing features**: `compute_basic_features()` currently returns a mix of absolute pixel values (`face_width`, `face_height`, `mouth_width`, `mouth_open`, `left_eye_open`, `right_eye_open`) and ratios (`face_ratio`, `mouth_width_ratio`, `mouth_open_ratio`). The new function should return ONLY ratios.

**New features to add (all ratios, scale-invariant):**

Eyebrow features (MediaPipe indices):
- `left_brow_height_ratio` / `right_brow_height_ratio` - brow midpoint to eye center distance / face height (indices: 66,105 for brow; 159,145 for eye center)
- `left_brow_angle` / `right_brow_angle` - angle of brow line (107→70 left, 336→300 right) in radians, via `np.arctan2`

Nose features:
- `nose_width_ratio` - nostril width (48, 278) / face width
- `nose_length_ratio` - bridge to tip (6, 1) / face height

Face shape features:
- `jaw_width_ratio` - jaw corners (172, 397) / cheek width (234, 454)
- `forehead_width_ratio` - forehead points (54, 284) / face width
- `chin_length_ratio` - mouth bottom to chin (17, 152) / face height

Expanded eye features:
- `eye_aspect_ratio_left/right` - openness / width (using 133,33 for left corners; 362,263 for right)
- `eye_width_ratio` - avg eye width / face width
- `inter_eye_distance_ratio` - inner corner distance (133, 362) / face width

Existing features to include (already computed):
- `face_ratio` (face_width / face_height)
- `mouth_width_ratio` (mouth_width / face_width)
- `mouth_open_ratio` (mouth_open / face_height)

**Key**: The new function returns ~20-25 float values, ALL normalized ratios. No absolute pixel values.

---

## Phase 3: Labeling Tool & Training Data

### 3a. Create `src/labeling_tool.py` — **DONE**
CLI tool for manual labeling (backup option). Shows face with landmarks, prompts for labels per category, appends to `data/labels/manual_labels.csv`.

### 3b. Import CelebA dataset and auto-label — **Assigned: Jess**
Uses `src/import_celeba.py` to download a CelebA subset from Kaggle and auto-map its 40 binary attributes to our emoji categories. No manual labeling needed.

**One-time setup:**
1. `pip install kaggle`
2. Go to https://www.kaggle.com/settings → "Create New Token"
3. Save downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
4. `chmod 600 ~/.kaggle/kaggle.json`

**Run:**
```
python src/import_celeba.py --count 500
```

This will:
1. Download CelebA from Kaggle (~1.4GB) to `data/celeba_raw/`
2. Filter out images with glasses/hats/blur
3. Sample 500 images → `data/raw/`
4. Map CelebA attributes to our categories → `data/labels/manual_labels.csv`
   - `Smiling` + `Mouth_Slightly_Open` → mouth_type
   - `Oval_Face`, `Chubby`, `High_Cheekbones` → face_shape
   - `Narrow_Eyes`, `Bags_Under_Eyes` → eye_type
   - `Arched_Eyebrows`, `Bushy_Eyebrows` → brow_type
   - `Big_Nose`, `Pointy_Nose` → nose_type
5. Print label distribution

**After import, build the feature dataset:**
```
python src/build_dataset.py
```
This runs landmarking + `compute_all_features` + skin tone on each image, joins with labels → `data/processed/features_labeled.csv`.

### 3c. Create `src/build_dataset.py` — **DONE**
Batch feature extraction + label join pipeline. Already created.

---

## Phase 4: ML Training & Model Comparison

### Create `src/train.py`
1. Load `features_labeled.csv` → X (numeric features), y_dict (labels per category)
2. **Per-category independent classifiers** - one per trait (face_shape, eye_type, mouth_type, brow_type, nose_type)
3. **Models to compare** (course requirement):
   - KNN (grid search k=3,5,7)
   - Random Forest
   - SVM (RBF kernel)
   - Decision Tree
4. StandardScaler, stratified 5-fold CV
5. Save best model per category to `models/trained/<category>.pkl`
6. Save results to `outputs/training_results.json`

### Create `src/evaluate.py`
- Accuracy bar charts per category per model
- Confusion matrices for best models
- Save plots to `outputs/plots/`

---

## Phase 5: Emoji Composition

### Create `src/compose.py`
Composites the selected OpenMoji parts into a final emoji:

```python
class EmojiComposer:
    def __init__(self, assets_dir: str):
        # Load asset registry from JSON metadata files
        # Maps (part_type, descriptor) -> {"png_path": ..., "stats": ...}
        # Example: ("mouths", "smile") -> {png, svg, bbox positions}

    def compose(self, components: dict, skin_tone_rgb: tuple) -> Image:
        # components = {"face_shape": "oval", "eye_type": "round", "mouth_type": "smile", ...}
        # 1. Create 512x512 RGBA canvas
        # 2. Draw face shape with Pillow (filled ellipse/rounded rect) using skin_tone_rgb
        # 3. Load eyes PNG, paste at position from JSON stats (bbox_x1_norm, bbox_y1_norm)
        # 4. Load eyebrows PNG, paste above eyes using stats positions
        # 5. Load mouth PNG, paste at stats position
        # Return PIL Image
```

**Asset placement strategy**: Each extracted part's JSON has `bbox_x1_norm`, `bbox_y1_norm`, `center_x_norm`, `center_y_norm` relative to the 512x512 render canvas. Since all parts come from the same OpenMoji 72x72 grid rendered at 512x512, their normalized positions are directly usable for placement on the composition canvas.

Example from extracted mouth (`1F600_004`): `center_x_norm=0.499, center_y_norm=0.689, width_norm=0.383, height_norm=0.104` — this tells us exactly where to place the mouth on the canvas.

**Face shape**: Draw with Pillow since all OpenMoji faces are identical yellow circles (no variety to extract). Parameterize by face_shape category:
- `oval`: tall ellipse (width=0.85, height=0.95 of canvas)
- `round`: circle (width=height=0.90)
- `square`: rounded rectangle
- `heart`: wider top, narrower chin
- `oblong`: narrow tall ellipse

**Skin tone**: Fill the face shape with the matched `skin_tone_rgb` from `SkinToneResult.matched_tone_rgb`. Add a slightly darker outline for definition.

---

## Phase 6: End-to-End Pipeline & Demo

### Create `src/pipeline.py`
```python
def generate_emoji(image_path: str, model_dir="models/trained", assets_dir="outputs/openmoji_parts") -> Image:
    # 1. FaceProcessor(model_path="models/face_landmarker.task") -> FaceResult
    # 2. compute_all_features(face_result.landmarks_pixels) -> feature dict
    # 3. estimate_skin_tone(image_bgr, face_result.landmarks_pixels) -> SkinToneResult
    # 4. Load trained .pkl classifiers, predict component categories from features
    # 5. EmojiComposer(assets_dir).compose(components, skin_tone_result.matched_tone_rgb)
    # Return PIL Image
```
CLI: `python src/pipeline.py --image data/raw/sample.jpg --output outputs/emoji.png`

### Create `src/demo.py`
Gradio web UI:
- Upload photo → show generated emoji
- Display detected features, predicted components, skin tone
- ~15 lines with Gradio

---

## Phase 7: Calibrate `skin_tone.py` palette
Update `DEFAULT_SKIN_TONE_PALETTE` (line 10-16) to match the fill colors actually used in the face shape assets.

---

## Implementation Order

| # | Task | Files | Status |
|---|------|-------|--------|
| 1 | Finish extracting assets (eyes, brows, remaining emojis) | Run `extract_openmoji_parts.py` | **DONE** (21 eyes, 4 brows, 27 mouths, 21 faces) |
| 2 | Expand feature extraction | Modify `src/features.py` | **DONE** (16 ratio features) |
| 3 | Build labeling tool | Create `src/labeling_tool.py` | **DONE** |
| 4 | Import CelebA + auto-label (~500 images) | Run `src/import_celeba.py` | TODO — **Assigned: Jess** |
| 5 | Build dataset pipeline | Create `src/build_dataset.py` + run after step 4 | **DONE** (code ready, run after step 4) |
| 6 | Train & compare models | Create `src/train.py`, `src/evaluate.py` | TODO |
| 7 | Build emoji composer | Create `src/compose.py` | TODO |
| 8 | End-to-end pipeline | Create `src/pipeline.py` | TODO |
| 9 | Gradio demo | Create `src/demo.py` | TODO |
| 10 | Calibrate skin tone palette | Modify `src/skin_tone.py` | TODO |

Steps 2-3 can be done in parallel with step 1 (asset extraction is manual/interactive).
Steps 5-6 depend on steps 2-4 being complete.
Step 7 depends on step 1 (need assets to compose).
Steps 8-9 depend on steps 6-7.

---

## Critical Files

| File | Action | Purpose |
|------|--------|---------|
| `src/features.py` | Modify | Expand from 9→~25 ratio-only features |
| `src/extract_openmoji_parts.py` | Done | All 21 emojis extracted; `--filter` flag added for re-extraction |
| `src/labeling_tool.py` | Done | CLI for manual labeling (backup option) |
| `src/import_celeba.py` | Run (Jess) | Download CelebA subset, auto-map attributes to emoji categories |
| `src/build_dataset.py` | Done (run after import) | Batch feature extraction + label join → CSV |
| `src/train.py` | Create | KNN/RF/SVM/DT comparison with stratified CV |
| `src/evaluate.py` | Create | Plots: accuracy bars, confusion matrices |
| `src/compose.py` | Create | Pillow face shape + OpenMoji part overlays |
| `src/pipeline.py` | Create | End-to-end photo → emoji orchestration |
| `src/demo.py` | Create | Gradio web UI |
| `src/skin_tone.py` | Modify | Calibrate palette to face shape fill colors |

## Verification

1. **Assets**: Verify `outputs/openmoji_parts/reviewed/` has 3-5 variants each for eyes, eyebrows, mouths
2. **Features**: Run on `sample.jpg`, verify ~25 features returned, all in [0,1] range (ratios)
3. **Training**: Cross-validation accuracy > random baseline (>20% for 5-class, >25% for 4-class)
4. **Composition**: Generate emoji for known component combos, visually check part alignment
5. **End-to-end**: `python src/pipeline.py --image data/raw/sample.jpg --output outputs/emoji.png`
6. **Demo**: `python src/demo.py`, upload photo in browser, confirm emoji output
