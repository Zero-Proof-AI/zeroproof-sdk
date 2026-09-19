"""The reader's side of this page.

The page does `import whileai as wai`, so `wai` here is the top-level package,
not `whileai.simulations`. Binding it the other way round would hand the page a
different `wai` than the one it asks for, and the blocks would fail on names
that are actually fine.
"""

from _common import POLICY, TOOLS  # noqa: F401

import whileai as wai  # noqa: F401
