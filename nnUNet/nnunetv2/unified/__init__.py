"""Unified benchmark layer on top of nnU-Net v2.

Eleven segmentation methods (ten published architectures plus nnU-Net itself)
behind one command line, five training-set definitions (M0-M4), an optional
second output head, and a top-k connected-component constraint at inference.

Nothing in here changes nnU-Net's defaults for a plain ``nnUNetTrainer`` run.
"""
