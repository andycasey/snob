
""" Snob, 'because it arbitrarily puts things in classes' -- C.S. Wallace. """

import logging

from . import (estimator, gaussian, slf)

__version__ = "0.0.1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

del handler, logger, logging 
