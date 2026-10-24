# NTV-AI experiments

Reproduction code and frozen experiment artifacts for **NTV-AI: A High-Resolution Dataset and Benchmark for Power Transmission Tower Inspection**.

This release contains the detector implementations, original hyperparameters, image identities, annotation snapshots, split manifests, predictions for the main benchmark, and results reported in the paper and supplement. It provides two reproduction paths: replay the main benchmark metrics on CPU, or obtain the images and reproduce training and inference.

## Experiments covered

| Study | Training | Evaluation |
| --- | --- | --- |
| Main benchmark | YOLO26-S, D-FINE-S, RF-DETR-Nano; `full640` and `tiles1280`; seeds 42/43/44; **18 runs** | Same 30 training and 8 validation images; **18 test images / 411 objects** |
| External transfer | Four curated external subsets × three detectors × two protocols × three seeds; **72 runs** | Source test and the original **8-image NTV-AI test**; 144 evaluations |
| Combined sources | RF-DETR-Nano; NTV-AI only, external only, combined; two protocols and three seeds; **18 runs** | NTV-AI (8), Furnas (17), STN PLAD (17); 54 evaluations |
| Language-guided detection | Fixed Grounding DINO Tiny and Qwen3-VL-4B-Instruct weights | 16 validation and 5 distinct test inference configurations; each test output evaluated on seven and three classes |
| Annotation detector | RF-DETR-Nano; **68 training / 10 validation images**, seed 43 | Development validation and class-wise proposer combination; not an independent test |

The paper's external experiments use the **frozen curated subsets**, not the complete source datasets. Later full-dataset campaigns and the later 81/20 annotation-detector experiment are not results of this release. The main test covers three captures of the **same physical structure**; it does not establish cross-tower generalization.

## Verify the published main benchmark without a GPU

Python 3.12 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[evaluation,dev]'
python scripts/paper.py verify
python scripts/paper.py replay --output runs/main-metrics-replayed.json
```

On Windows, activate `.venv\Scripts\Activate.ps1`. Metric replay needs neither images nor model weights. It recomputes COCO metrics from the saved predictions and frozen annotations and checks all 18 runs against their published results. Means and **sample** standard deviations are reported on the 0–100 scale. Training and GPU inference use Linux, as in the original implementation.

The RF-DETR-Nano `tiles1280` AP should round to **45.52 ± 3.64**. D-FINE-S and YOLO26-S should round to **35.95 ± 1.87** and **23.62 ± 1.09**, respectively. These are results on the expanded test, not improvements obtained by retraining the models.

## Reproduce training and inference

Follow [REPRODUCING.md](REPRODUCING.md) for image access, exact split preparation, environments, pretrained assets, training, checkpoint evaluation, and VLM runs. `scripts/paper.py list` lists every frozen experiment identifier. Preparation and verification do not start training or inference.

The original trained pilot detector package is available through the [model download](https://drive.google.com/file/d/1_7_dMiDW62qesVG9k1e81qCB1DX4cR03/view?usp=sharing). The release records checkpoint SHA-256 hashes for all 18 main configurations. Large weights and source photographs are not stored in Git.

## Repository layout

```text
src/ntv_ai/          Existing data, detector, training-control and evaluation code
arquiteturas/       Original Conda environments and dependency locks
configs/            Existing recipes, taxonomy, mappings and acquisition definitions
scripts/paper.py    Paper-specific preparation, training, evaluation and metric replay
paper/experiments.json       Frozen paper training and VLM configurations
paper/datasets/             Compressed annotation snapshots, hashes and split lists
paper/results/              Published results and main-test predictions
paper/tables/               English table values from the paper and supplement
paper/provenance.json        Source fingerprints and packaging transformations
tests/                     Software tests; GPU integration tests are opt-in
```

The generic package retains shared utilities beyond the paper. **`paper/experiments.json` is the authoritative list of experiments for this release.** Dataset snapshots preserve labels, boxes, image hashes, groups, and splits. Private machine locators are replaced with portable roots, and reviewer identifiers are pseudonymized. Original snapshot fingerprints remain recorded; packaging does not imply additional human validation.

## Data and third-party resources

NTV-AI images and annotations can be requested from the responsible team; use this repository's issues to contact the maintainers. External image collections remain available from their original providers through the download configuration. The selected image identities and the annotations actually used in these experiments are frozen here. Downloading a full source collection does not change the experimental subsets.

Third-party frameworks, model weights, and source datasets retain their respective licenses and access conditions. No new dataset or third-party license is granted by this repository. See [REPRODUCING.md](REPRODUCING.md) for source links and exact versions.
