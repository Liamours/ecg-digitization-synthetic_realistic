"""Digitize photographed ECG paper without repairing the image.

Stages (one module each): gridmap (pixels to millimetres from the printed grid), ocr and text (gain, labels,
header and footer zone), pulses (calibration step), mask (trace pixels), leads (row assignment), sample (mV over
time), selftest (pulse, Einthoven, lag, coverage), reverse (draw results on the original image), pipeline (glue).
Every result is a PanelRecord that carries the transforms needed to map it back onto the original image.
"""
