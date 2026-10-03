# Third-party code

Unmodified copies of the files this pipeline uses from two open-source ECG digitizers. Each folder keeps its project's license.

| Folder | Project | Commit | License | Files | Used for |
|---|---|---|---|---|---|
| `open_ecg_digitizer/` | [Ahus-AIM/Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer) | `97a15087d4abcda843da8c58ee74b1d8f47e6f9a` | CC BY-SA 4.0 | `unet.py` (`src/model/unet.py`), `signal_extractor.py` (`src/model/signal_extractor.py`) | trace segmentation network; the `openecg_lines` lead separation |
| `ecgtizer/` | [UMMISCO/ecgtizer](https://github.com/UMMISCO/ecgtizer) | `8cd5b01e8450f023c711364cefd65493ec21d476` | Unlicense (public domain) | `PDF2XML.py`, `extraction_functions.py` (`ecgtizer/`) | track cutting and the `lazy`, `full`, `fragmented` extractions |

The trace network's weights (`unet_weights_07072025.pt`, 90 MB, CC BY-SA 4.0) are not in this repository; `scripts/get_weights.sh` downloads them from the Open-ECG-Digitizer repository.

Open-ECG-Digitizer asks for this citation: Stenhede E, Bjørnstad AM, Ranjbar A. Digitizing Paper ECGs at Scale: An Open-Source Algorithm for Clinical Research. npj Digital Medicine, 2026. doi:10.1038/s41746-025-02327-1.

The text on a page is read by RapidOCR ([RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR), Apache-2.0), installed as a dependency through docling.
