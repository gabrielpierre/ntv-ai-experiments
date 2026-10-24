**Table S1 — Sets used in NTV-AI construction and evaluation.**

| Set | Images | Boxes | Purpose |
| --- | --- | --- | --- |
| Pilot | 50 | 968 | Training, validation, and original test |
| Expanded test | 18 | 411 | Main benchmark; images included in the collection |
| Human calibration | 30 | 618 | Annotation detector development |
| Deduplicated union | 78 | 1,547 | Fine-tuning: 68 training / 10 validation |
| Assisted expansion | 4,349 | 109,370 | Model-assisted annotation and human review |
| NTV-AI | 4,427 | 110,917 | Full dataset |

**Table S2 — Per-class results on the 18 test images. Mean ± standard deviation across three seeds. Bold: highest mean per class and metric.**

| Model | Input | Class | AP | AP50 | AR100 |
| --- | --- | --- | --- | --- | --- |
| YOLO26-S | full640 | Tower | 50.13 ± 4.04 | 72.89 ± 2.30 | 54.44 ± 4.59 |
| YOLO26-S | full640 | Insulator | 33.78 ± 2.47 | 62.40 ± 4.05 | 39.86 ± 0.45 |
| YOLO26-S | full640 | Concrete foundation | 19.15 ± 2.58 | 54.26 ± 6.10 | 24.50 ± 3.63 |
| YOLO26-S | full640 | Guy-anchor foundation | 10.90 ± 1.16 | 40.15 ± 0.93 | 17.26 ± 0.65 |
| YOLO26-S | full640 | Spacer | 22.04 ± 4.43 | 42.76 ± 4.20 | 26.51 ± 4.17 |
| YOLO26-S | full640 | Stockbridge damper | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| YOLO26-S | full640 | Signs | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| D-FINE-S | full640 | Tower | 60.66 ± 2.08 | 78.68 ± 5.50 | 64.72 ± 2.74 |
| D-FINE-S | full640 | Insulator | 28.50 ± 4.90 | 54.25 ± 10.23 | 32.39 ± 4.98 |
| D-FINE-S | full640 | Concrete foundation | 14.37 ± 1.78 | 46.46 ± 4.08 | 24.67 ± 1.81 |
| D-FINE-S | full640 | Guy-anchor foundation | 11.53 ± 1.11 | 35.71 ± 1.30 | 20.11 ± 2.80 |
| D-FINE-S | full640 | Spacer | 19.44 ± 2.10 | 42.43 ± 5.77 | 23.25 ± 3.97 |
| D-FINE-S | full640 | Stockbridge damper | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| D-FINE-S | full640 | Signs | 0.46 ± 0.43 | 3.13 ± 2.74 | 1.82 ± 1.82 |
| RF-DETR-Nano | full640 | Tower | **73.56** ± 3.12 | **93.77** ± 3.57 | **77.59** ± 3.22 |
| RF-DETR-Nano | full640 | Insulator | 25.74 ± 1.66 | 51.27 ± 2.29 | 28.70 ± 2.26 |
| RF-DETR-Nano | full640 | Concrete foundation | 18.02 ± 2.78 | 57.10 ± 4.04 | 23.58 ± 2.93 |
| RF-DETR-Nano | full640 | Guy-anchor foundation | 19.69 ± 2.12 | 48.54 ± 1.26 | 24.57 ± 1.07 |
| RF-DETR-Nano | full640 | Spacer | 18.86 ± 3.05 | 44.69 ± 5.94 | 22.06 ± 2.62 |
| RF-DETR-Nano | full640 | Stockbridge damper | 0.00 ± 0.00 | 0.01 ± 0.01 | 0.11 ± 0.15 |
| RF-DETR-Nano | full640 | Signs | 0.60 ± 0.48 | 3.64 ± 2.91 | 4.55 ± 0.91 |
| YOLO26-S | tiles1280 | Tower | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.46 ± 0.58 |
| YOLO26-S | tiles1280 | Insulator | 30.24 ± 5.06 | 70.60 ± 4.40 | 39.86 ± 2.26 |
| YOLO26-S | tiles1280 | Concrete foundation | 34.26 ± 4.53 | 58.16 ± 1.11 | 37.25 ± 4.44 |
| YOLO26-S | tiles1280 | Guy-anchor foundation | 47.21 ± 3.45 | 71.75 ± 5.96 | 56.24 ± 2.00 |
| YOLO26-S | tiles1280 | Spacer | 31.96 ± 2.05 | 56.06 ± 2.51 | 35.87 ± 2.26 |
| YOLO26-S | tiles1280 | Stockbridge damper | 0.42 ± 0.36 | 1.25 ± 1.19 | 0.50 ± 0.52 |
| YOLO26-S | tiles1280 | Signs | 21.25 ± 2.18 | 32.46 ± 6.00 | 28.18 ± 7.27 |
| D-FINE-S | tiles1280 | Tower | 5.77 ± 1.93 | 28.81 ± 4.84 | 22.04 ± 4.12 |
| D-FINE-S | tiles1280 | Insulator | 47.17 ± 1.36 | 67.97 ± 2.08 | 64.13 ± 5.29 |
| D-FINE-S | tiles1280 | Concrete foundation | 44.48 ± 2.32 | 78.35 ± 4.23 | 53.00 ± 2.05 |
| D-FINE-S | tiles1280 | Guy-anchor foundation | 55.15 ± 2.00 | 79.22 ± 2.08 | 67.37 ± 1.31 |
| D-FINE-S | tiles1280 | Spacer | 52.06 ± 3.88 | 81.93 ± 3.11 | 59.44 ± 2.26 |
| D-FINE-S | tiles1280 | Stockbridge damper | 6.52 ± 2.38 | 18.06 ± 5.11 | 16.88 ± 4.75 |
| D-FINE-S | tiles1280 | Signs | 40.51 ± 4.51 | 54.31 ± 8.38 | 42.12 ± 5.01 |
| RF-DETR-Nano | tiles1280 | Tower | 8.13 ± 1.96 | 41.94 ± 6.29 | 23.06 ± 2.20 |
| RF-DETR-Nano | tiles1280 | Insulator | **54.02** ± 3.62 | **75.30** ± 5.70 | **67.10** ± 2.22 |
| RF-DETR-Nano | tiles1280 | Concrete foundation | **65.24** ± 8.09 | **92.89** ± 5.31 | **73.25** ± 7.12 |
| RF-DETR-Nano | tiles1280 | Guy-anchor foundation | **67.14** ± 5.00 | **92.31** ± 2.30 | **76.40** ± 5.37 |
| RF-DETR-Nano | tiles1280 | Spacer | **59.75** ± 7.06 | **88.49** ± 8.05 | **71.11** ± 4.86 |
| RF-DETR-Nano | tiles1280 | Stockbridge damper | **7.46** ± 0.36 | **19.12** ± 4.10 | **21.42** ± 9.73 |
| RF-DETR-Nano | tiles1280 | Signs | **56.90** ± 6.91 | **69.73** ± 10.05 | **64.85** ± 2.29 |

**Table S3 — AP by COCO area range on the 18 test images. Bold: highest mean per area range.**

| Model | Input | Small AP | Medium AP | Large AP |
| --- | --- | --- | --- | --- |
| YOLO26-S | full640 | 4.87 ± 0.62 | 14.59 ± 0.92 | 14.37 ± 0.72 |
| D-FINE-S | full640 | 4.59 ± 0.42 | 13.51 ± 1.23 | 15.10 ± 0.85 |
| RF-DETR-Nano | full640 | 3.77 ± 0.58 | 15.53 ± 0.15 | 21.04 ± 1.30 |
| YOLO26-S | tiles1280 | 22.11 ± 2.83 | 29.77 ± 7.29 | 7.72 ± 0.47 |
| D-FINE-S | tiles1280 | 33.62 ± 1.42 | 50.22 ± 1.56 | 16.91 ± 3.57 |
| RF-DETR-Nano | tiles1280 | **37.42** ± 2.82 | **58.12** ± 2.57 | **57.57** ± 7.97 |

**Table S4 — External splits and class support in the source test set.**

| Source | Training | Validation | Test | Test boxes |
| --- | --- | --- | --- | --- |
| MPID | 65 | 18 | 17 | 40 insulators |
| PTL-AI Furnas | 65 | 18 | 17 | 32 insulators, 4 spacers, 6 dampers, 3 marker balls |
| STN PLAD | 63 | 17 | 17 | 34 towers, 47 insulators, 37 spacers, 238 dampers, 11 signs |
| TTPLA | 58 | 15 | 15 | 2 towers; 13 negative images for the mapped task |

**Table S5 — AP on the training source and transfer to the eight original NTV-AI test images. Each row combines the two evaluations of the same model; mean ± standard deviation across three seeds. Aggregated classes follow Table S4, with the caveat regarding Furnas signage. Bold: highest mean per column within each source.**

| Source | Model | Input | Source AP | NTV-AI AP |
| --- | --- | --- | --- | --- |
| MPID | D-FINE-S | full640 | 22.65 ± 2.99 | 0.01 ± 0.01 |
| MPID | RF-DETR-Nano | full640 | **47.27** ± 3.92 | 0.00 ± 0.00 |
| MPID | YOLO26-S | full640 | 1.78 ± 1.42 | 0.00 ± 0.00 |
| MPID | D-FINE-S | tiles1280 | 9.52 ± 1.44 | 0.44 ± 0.72 |
| MPID | RF-DETR-Nano | tiles1280 | 24.02 ± 3.03 | **17.86** ± 6.44 |
| MPID | YOLO26-S | tiles1280 | 0.02 ± 0.01 | 0.00 ± 0.00 |
| PTL-AI Furnas | D-FINE-S | full640 | 16.37 ± 5.94 | 0.01 ± 0.01 |
| PTL-AI Furnas | RF-DETR-Nano | full640 | **42.56** ± 1.57 | 0.00 ± 0.00 |
| PTL-AI Furnas | YOLO26-S | full640 | 3.50 ± 3.01 | 0.00 ± 0.00 |
| PTL-AI Furnas | D-FINE-S | tiles1280 | 12.09 ± 7.52 | 1.21 ± 2.05 |
| PTL-AI Furnas | RF-DETR-Nano | tiles1280 | 41.37 ± 6.48 | **9.51** ± 0.60 |
| PTL-AI Furnas | YOLO26-S | tiles1280 | 0.38 ± 0.26 | 0.00 ± 0.00 |
| STN PLAD | D-FINE-S | full640 | 35.12 ± 0.28 | 3.32 ± 0.45 |
| STN PLAD | RF-DETR-Nano | full640 | 35.57 ± 0.31 | 7.73 ± 1.43 |
| STN PLAD | YOLO26-S | full640 | 15.67 ± 2.04 | 0.05 ± 0.08 |
| STN PLAD | D-FINE-S | tiles1280 | 40.44 ± 1.83 | 1.09 ± 0.39 |
| STN PLAD | RF-DETR-Nano | tiles1280 | **50.41** ± 0.82 | **8.91** ± 3.37 |
| STN PLAD | YOLO26-S | tiles1280 | 9.16 ± 0.79 | 0.02 ± 0.04 |
| TTPLA | D-FINE-S | full640 | 59.12 ± 30.64 | 1.48 ± 2.52 |
| TTPLA | RF-DETR-Nano | full640 | **90.20** ± 0.21 | **2.09** ± 1.83 |
| TTPLA | YOLO26-S | full640 | 0.05 ± 0.05 | 0.00 ± 0.01 |
| TTPLA | D-FINE-S | tiles1280 | 0.84 ± 1.46 | 0.03 ± 0.04 |
| TTPLA | RF-DETR-Nano | tiles1280 | 19.97 ± 22.08 | 0.90 ± 0.45 |
| TTPLA | YOLO26-S | tiles1280 | 0.16 ± 0.27 | 0.00 ± 0.00 |

**Table S6 — Per-class AP in the combined-source experiment. NTV: training on NTV-AI alone; external: Furnas and STN PLAD; combined: union of all three sources. Mean ± standard deviation across three seeds. Bold: highest nonzero mean per test set, class, and protocol.**

| Input | Training | Class | NTV-AI test | Furnas test | STN PLAD test |
| --- | --- | --- | --- | --- | --- |
| full640 | NTV | Stockbridge damper | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| full640 | NTV | Spacer | 4.77 ± 2.29 | 0.02 ± 0.00 | 2.81 ± 3.33 |
| full640 | NTV | Insulator | **7.94** ± 1.41 | 0.24 ± 0.05 | 0.04 ± 0.01 |
| full640 | External | Stockbridge damper | 0.00 ± 0.00 | **45.57** ± 4.33 | 2.20 ± 0.18 |
| full640 | External | Spacer | 1.18 ± 0.70 | 7.58 ± 4.82 | 26.75 ± 1.87 |
| full640 | External | Insulator | 0.25 ± 0.07 | 54.78 ± 2.27 | 40.58 ± 2.21 |
| full640 | Combined | Stockbridge damper | 0.00 ± 0.00 | 45.03 ± 1.99 | **3.18** ± 0.27 |
| full640 | Combined | Spacer | **6.20** ± 2.29 | **9.65** ± 5.57 | **27.17** ± 1.41 |
| full640 | Combined | Insulator | 3.15 ± 2.03 | **55.38** ± 3.15 | **41.42** ± 0.88 |
| tiles1280 | NTV | Stockbridge damper | 0.35 ± 0.27 | 0.00 ± 0.00 | 0.39 ± 0.24 |
| tiles1280 | NTV | Spacer | **43.26** ± 1.89 | 0.00 ± 0.00 | 5.43 ± 2.27 |
| tiles1280 | NTV | Insulator | **51.93** ± 5.08 | 6.80 ± 1.68 | 0.93 ± 0.41 |
| tiles1280 | External | Stockbridge damper | 0.50 ± 0.40 | **32.31** ± 11.02 | **42.85** ± 0.89 |
| tiles1280 | External | Spacer | 14.23 ± 6.50 | **2.28** ± 3.60 | **64.28** ± 1.19 |
| tiles1280 | External | Insulator | 28.90 ± 10.23 | 43.62 ± 3.64 | **64.88** ± 4.23 |
| tiles1280 | Combined | Stockbridge damper | **0.69** ± 0.63 | 26.84 ± 21.73 | 41.37 ± 2.04 |
| tiles1280 | Combined | Spacer | 38.94 ± 5.51 | 0.52 ± 0.33 | 53.63 ± 5.65 |
| tiles1280 | Combined | Insulator | 50.43 ± 3.31 | **44.67** ± 2.56 | 58.00 ± 2.66 |

**Table S7 — Results of models with fixed weights. Each row corresponds to one run; GD: Grounding DINO Tiny; Qwen: Qwen3-VL-4B-Instruct. Letters identify configurations in Table S8. Bold: highest AP per column.**

| Config. | Model | Input | Prompt | AP (7) | AP (3) | Invalid outputs |
| --- | --- | --- | --- | --- | --- | --- |
| A | GD | full640 | P0 | 2.19 | 0.00 | 0 |
| B | GD | tiles1280 | P0 | 1.81 | 1.94 | 0 |
| C | Qwen | full640 | P0 | **5.71** | 0.00 | 0 |
| D | Qwen | tiles1280 | P0 | 4.50 | **4.44** | 2 |
| E | Qwen | tiles1280 | P1 | 5.52 | 3.72 | 3 |

**Table S8 — Per-class AP for configurations A–E in Table S7. Bold: highest nonzero AP per class.**

| Class | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| Tower | 15.31 | 6.87 | **39.98** | 6.56 | 5.87 |
| Insulator | 0.00 | 5.80 | 0.00 | **13.31** | 11.17 |
| Concrete foundation | 0.00 | 0.00 | 0.00 | 11.60 | **21.59** |
| Guy-anchor foundation | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Spacer | 0.00 | **0.01** | 0.00 | 0.00 | 0.00 |
| Stockbridge damper | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Signs | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Table S9 — Per-class AP on the annotation detector's ten-image validation set, with a minimum confidence of 0.10. Bold: highest AP per class, including ties.**

| Class | Fine-tuned proposer | Combination |
| --- | --- | --- |
| Tower | 8.88 | **61.24** |
| Insulator | **40.80** | **40.80** |
| Concrete foundation | 66.19 | **78.52** |
| Guy-anchor foundation | 80.92 | **99.57** |
| Spacer | 47.32 | **58.18** |
| Stockbridge damper | **30.79** | **30.79** |
| Signs | **75.15** | **75.15** |
