"""
Constants for the closest-intent custom component.
"""

DOMAIN = "closest_intent"

# Marker substituted in for `{slot}` placeholders during pattern expansion.
# Matched as a wildcard during scoring; mined out for slot extraction.
SLOT_WILDCARD = "\x00slot\x00"
