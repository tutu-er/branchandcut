#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the standalone UnitPredictor for case30lite."""

from __future__ import annotations

import argparse

from run_unit_predictor_common import (
    UnitPredictorConfig,
    add_common_args,
    run_unit_predictor_training,
)


CONFIG = UnitPredictorConfig(
    case_name="case30lite",
    max_samples=260,
    epochs=170,
    batch_size=64,
    lr=1.0e-3,
    weight_decay=8.0e-5,
    hidden_dims=(256, 128),
    dropout=0.05,
    loss_weight_mse=0.12,
    loss_weight_transition=0.05,
    loss_weight_binarize=0.01,
    loss_weight_tv_floor=0.04,
    tv_floor_scale=0.75,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, CONFIG)
    args = parser.parse_args()
    run_unit_predictor_training(CONFIG, args)


if __name__ == "__main__":
    main()
