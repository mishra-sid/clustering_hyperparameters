#!/usr/bin/env python
# -*- coding: utf-8 -*-
__requires__ = "model-hyperparameters"
import re
import sys

from clustering_hyperparameters.__main__ import main

if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw?|\.exe)?$", "", sys.argv[0])
    sys.exit(main())