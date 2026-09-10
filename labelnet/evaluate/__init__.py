"""Evaluation of label placement quality.

- ``wms``     : work with the WMS request logs (JSON) and the local map servers
- ``metrics`` : unambiguity, legibility and label score
- ``sweep``   : alpha/beta threshold sweep over the four scale contexts (part 1)
- ``accuracy``: precision/recall/accuracy against ground truth (part 2)
- ``compare`` : per-epoch prediction comparison (old main.py comparison flow)
"""
