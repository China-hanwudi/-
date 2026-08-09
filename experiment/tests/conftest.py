"""Keep the production package tree closed while pytest imports test modules.

The production CLI intentionally rejects every non-``.py`` payload below
``experiment/src/hva_affect`` before its first package import.  Prevent pytest's
normal collection imports from creating a cache that would make later CLI tests
order-dependent.
"""

import sys


sys.dont_write_bytecode = True
