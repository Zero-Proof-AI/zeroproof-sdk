"""The reader's side of docs/reference/style.md.

The style guide writes `import whileai as wai`, so `wai` is the top-level
package here, and its front-page example needs the agent's tools and policy.
"""

from _common import POLICY, TOOLS  # noqa: F401

import whileai as wai  # noqa: F401
