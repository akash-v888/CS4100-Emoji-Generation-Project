# CS4100 Emoji Generation Project

Generate a personalized emoji from a face photo. The pipeline detects facial landmarks, extracts geometric features, classifies them into emoji component categories, and composites matching OpenMoji parts into a final emoji with auto-detected skin tone and hair.

## Pipeline Overview

```
photo → face landmarks (MediaPipe) → 16 ratio features
     → per-category classifiers (face shape, eyes, mouth, brow, nose)
     → skin-tone sampling  +  hair detection
     → OpenMoji part composition → final emoji PNG
```

## Project Structure

```
src/
  landmarking.py            MediaPipe face detection, 478 landmarks
  features.py               16 scale-invariant ratio features
  skin_tone.py              Cheek sampling, Fitzpatrick palette match
  extract_openmoji_parts.py Interactive SVG part extractor
  labeling_tool.py          CLI labeling helper
  import_celeba.py          CelebA download + attribute auto-mapping
  build_dataset.py          Batch features + label join → CSV
  train.py                  KNN / RandomForest / SVM / DecisionTree comparison
  evaluate.py               Accuracy bars, confusion matrices
  compose.py                OpenMoji part compositor, hair drawing, outline recoloring
  pipeline.py               End-to-end: photo → emoji
  demo.py                   Gradio web UI
data/
  openmojis/                21 source OpenMoji SVG faces
  labels/                   manual_labels.csv (CelebA-mapped)
  processed/                features_labeled.csv (499 samples)
models/
  face_landmarker.task      MediaPipe model
  trained/                  Per-category .pkl classifiers
outputs/
  openmoji_parts/reviewed/  Extracted eyes / eyebrows / mouth / face PNGs + SVGs + JSON
  plots/                    accuracy_comparison.png, confusion_matrices.png
  training_results.json     Cross-validation accuracy per model per category
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The MediaPipe model (`models/face_landmarker.task`) and extracted OpenMoji assets are committed, so the pipeline runs without additional downloads.

## Run the Demo

Gradio web UI — upload a face photo, get an emoji:

```bash
python src/demo.py
```

Or run the pipeline from the CLI:

```bash
python src/pipeline.py --image data/raw/sample.jpg --output outputs/emoji.png
```

## Reproducing Training

Rebuilding the dataset from scratch requires a Kaggle API token (`~/.kaggle/kaggle.json`) for the CelebA download.

```bash
python src/import_celeba.py --count 500   # download + auto-label
python src/build_dataset.py               # landmarks + features → CSV
python src/train.py                       # train 4 models per category, save best
python src/evaluate.py                    # plots → outputs/plots/
```

## Models Compared

Per-category independent classifiers, stratified 5-fold CV on 499 CelebA samples:

| Category    | Best Model    | Accuracy |
|-------------|---------------|----------|
| face_shape  | SVM (RBF)     | 37.1%    |
| eye_type    | SVM (RBF)     | 42.9%    |
| mouth_type  | Random Forest | 71.8%    |
| brow_type   | see `outputs/training_results.json` |
| nose_type   | see `outputs/training_results.json` |

All four model families (KNN k=3/5/7, Random Forest, SVM-RBF, Decision Tree) are trained and compared; best model per category is saved to `models/trained/<category>.pkl`.

## Component Post-Processing

- **Eye type**: direct aspect-ratio thresholds (<0.27 squint, 0.27–0.40 round, >0.40 wide) override the weak classifier.
- **Skin tone**: cheek samples in LAB, 25% brightness boost to compensate for shadow.
- **Outlines**: darker shade of the skin tone (55% brightness) replaces black for face/eye/mouth/brow outlines.
- **Hair**: median color above forehead; short vs. long inferred from side-of-jaw pixel similarity; eyebrows recolored to hair color.

## Assets

Face components were extracted from 21 OpenMoji SVGs (`1F600`–`1F644`, `2639`) with an interactive reviewer. Each extracted part has `.svg`, `.png`, and `.json` metadata (normalized bbox / center) that drives placement on the composition canvas.

Counts: 21 eyes, 4 eyebrows, 27 mouths, 21 faces.

## License / Attribution

- OpenMoji assets — CC BY-SA 4.0 (https://openmoji.org/)
- CelebA dataset — research use only (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Short-hair reference shape traced from an Icons8 asset
