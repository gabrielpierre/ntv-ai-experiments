**Table 1 — Power line inspection datasets: classes, tasks, components, and annotation geometry. We count classes with detection or segmentation annotations in the cited versions. In Furnas, ten labels distinguish five components and their conditions; `stockbridge_nok` has no instances. TTPLA excludes the `void` label, which is ignored during evaluation; InsPLAD counts asset categories.**

| Dataset | No. of classes | Tasks | Annotated components | Geometry |
| --- | --- | --- | --- | --- |
| MPID [5] | 3 | Type-specific detection | Glass, porcelain, and composite insulators | Boxes |
| STN PLAD [2] | 5 | Asset detection | Towers, insulators, spacers, signs, and Stockbridge dampers | Boxes |
| PTL-AI Furnas [3] | 10 | Component and condition detection | Insulators, spacers, dampers, marker balls, and nests | Boxes with component/condition categories |
| TTPLA [4] | 4 | Detection and segmentation | Three tower categories and conductors | Pixel-level instances |
| InsPLAD [1] | 17 | Asset detection, defect classification, and anomaly detection | 17 asset categories | Boxes; condition labels on crops |
| NTV-AI | 7 | Component detection | Tower, insulator, two foundation classes, spacer, damper, and signs | Boxes |

**Table 2 — Distribution of NTV-AI annotations by class and capture. Captures 001, 002, and 003 contain 1,412, 1,985, and 1,030 images. The last column counts distinct images with at least one box of the class; an image may contribute to multiple classes.**

| Class | Capture 001 | Capture 002 | Capture 003 | Boxes | Images with class |
| --- | --- | --- | --- | --- | --- |
| Tower | 4,905 | 5,014 | 2,661 | 12,580 | 4,427 |
| Insulator | 7,601 | 3,528 | 5 | 11,134 | 1,890 |
| Concrete foundation | 3,450 | 7,532 | 3,065 | 14,047 | 4,408 |
| Guy-anchor foundation | 6,940 | 8,244 | 4,139 | 19,323 | 4,418 |
| Spacer | 6,453 | 2,467 | 943 | 9,863 | 2,837 |
| Stockbridge damper | 29,252 | 11,771 | 493 | 41,516 | 2,200 |
| Signs | 2,047 | 307 | 100 | 2,454 | 1,502 |
| Total boxes | 60,648 | 38,863 | 11,406 | 110,917 | — |

**Table 3 — Detector hyperparameters for the benchmark and transfer experiments. LR denotes learning rate; the backbone/encoder value is nominal, with additional layer-wise factors in RF-DETR. The reported weight decay applies to regularized groups. The combined-source and annotation models retain the RF-DETR-Nano values.**

| Model | Main LR | LR backbone/encoder | Weight decay | Gradient norm limit |
| --- | --- | --- | --- | --- |
| YOLO26-S | 0.001 | — | 0.0005 | 10.0 |
| D-FINE-S | 0.0002 | 0.0001 | 0.0001 | 0.1 |
| RF-DETR-Nano | 0.0001 | 0.00015 | 0.0001 | 0.1 |

**Table 4 — Seven-class benchmark on the 18 test images containing 411 objects. Mean ± sample standard deviation across three seeds, on a 0–100 scale. AP is averaged over IoUs from 0.50 to 0.95; AR100 uses up to 100 detections per image. Bold: highest mean in each column.**

| Model | Input | AP | AP50 | AP75 | AR100 |
| --- | --- | --- | --- | --- | --- |
| YOLO26-S | full640 | 19.43 ± 0.53 | 38.92 ± 1.53 | 17.28 ± 0.78 | 23.22 ± 1.06 |
| D-FINE-S | full640 | 19.28 ± 1.10 | 37.24 ± 1.03 | 16.11 ± 2.49 | 23.85 ± 1.60 |
| RF-DETR-Nano | full640 | 22.35 ± 0.71 | 42.72 ± 1.39 | 18.27 ± 1.70 | 25.88 ± 0.53 |
| YOLO26-S | tiles1280 | 23.62 ± 1.09 | 41.47 ± 0.97 | 24.27 ± 2.48 | 28.34 ± 2.06 |
| D-FINE-S | tiles1280 | 35.95 ± 1.87 | 58.38 ± 3.02 | 39.05 ± 1.41 | 46.43 ± 3.21 |
| RF-DETR-Nano | tiles1280 | **45.52** ± 3.64 | **68.54** ± 2.90 | **48.96** ± 2.69 | **56.74** ± 2.00 |

**Table 5 — RF-DETR-Nano on three classes: insulator, spacer, and damper; original eight-image NTV-AI test set. Mean AP ± standard deviation across three seeds (0–100). Checkpoints selected on NTV-AI validation. Bold: highest mean per test set and protocol.**

| Input | Training (images) | NTV-AI test (8) | Furnas test (17) | STN PLAD test (17) |
| --- | --- | --- | --- | --- |
| full640 | NTV-AI (30) | **4.23** ± 1.18 | 0.09 ± 0.02 | 0.95 ± 1.11 |
| full640 | External (128) | 0.48 ± 0.22 | 35.98 ± 0.88 | 23.17 ± 1.13 |
| full640 | Combined (158) | 3.12 ± 0.16 | **36.69** ± 2.30 | **23.92** ± 0.80 |
| tiles1280 | NTV-AI (30) | **31.85** ± 1.45 | 2.27 ± 0.56 | 2.25 ± 0.95 |
| tiles1280 | External (128) | 14.54 ± 5.21 | **26.07** ± 3.82 | **57.34** ± 1.92 |
| tiles1280 | Combined (158) | 30.02 ± 1.44 | 24.01 ± 7.68 | 51.00 ± 0.55 |
