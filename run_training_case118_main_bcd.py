#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import run_training_case118 as case118_cfg


case118_cfg.TRAIN_TARGET = "main_bcd"


if __name__ == "__main__":
    case118_cfg.main()
