#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "0.0.1"

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

del handler, logger, logging 
