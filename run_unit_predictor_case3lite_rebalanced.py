#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the standalone UnitPredictor for rebalanced case3lite data."""

from __future__ import annotations

import argparse

from run_unit_predictor_common import (
    UnitPredictorConfig,
    add_common_args,
    run_unit_predictor_training,
)


CONFIG = UnitPredictorConfig(
    case_name="case3lite",
    max_samples=0,
    epochs=700,
    batch_size=64,
    lr=1.5e-3,
    weight_decay=0.0,
    hidden_dims=(512, 256, 128),
    net_variant="tcn_shared_film",
    tcn_channels=128,
    tcn_depth=8,
    dropout=0.0,
    loss_weight_mse=0.30,
    loss_weight_transition=0.12,
    loss_weight_binarize=0.02,
    loss_weight_tv_floor=0.08,
    tv_floor_scale=0.80,
    scheduler_patience=80,
    scheduler_min_lr=1.0e-8,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, CONFIG)
    args = parser.parse_args()
    run_unit_predictor_training(CONFIG, args)


if __name__ == "__main__":
    main()
