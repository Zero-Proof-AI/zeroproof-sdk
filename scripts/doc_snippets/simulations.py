"""The reader's side of docs/simulations.md.

The page's two blocks are about which model runs a role and about the Hugging
Face round trip, so they write `my_tools` and `my_system_prompt` and leave the
agent to the reader. Both blocks need a key and stop at that check; these
names only have to exist for them to get that far.
"""

import whileai.simulations as wai  # noqa: F401

my_tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]
my_system_prompt = "Help customers with orders. Look an order up before you speak about it."
