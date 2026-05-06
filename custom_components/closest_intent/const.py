"""
Constants for the closest-intent custom component.
"""

DOMAIN = "closest_intent"

CONF_THRESHOLD = "threshold"
CONF_EXPANSION_CAP = "expansion_cap"
CONF_DENYLIST = "denylist"
CONF_INCLUDE_BUILTINS = "include_builtins"
CONF_SLOT_EXTRACTION = "slot_extraction"
CONF_BASE_AGENT = "base_agent"

DEFAULT_THRESHOLD = 70
DEFAULT_EXPANSION_CAP = 16
DEFAULT_INCLUDE_BUILTINS = False
DEFAULT_SLOT_EXTRACTION = True
# Where to forward the canonical sentence after a fuzzy match. Default is
# HA's bundled conversation agent. Be careful not to create a loop...
DEFAULT_BASE_AGENT = "conversation.home_assistant"

# Stash keys in `hass.data[DOMAIN]`.
KEY_CONVERSATION_INTENTS = "_conversation_intents"
KEY_CONVERSATION_LISTS = "_conversation_lists"
KEY_CONVERSATION_EXPANSION_RULES = "_conversation_expansion_rules"
KEY_AGENT_INSTANCES = "_agent_instances"

# Diagnostic service name.
SERVICE_DUMP_CANDIDATES = "dump_candidates"

# Hard ceiling on candidates kept per intent after pattern expansion.
PER_INTENT_CANDIDATE_CAP = 32

# Marker substituted in for `{slot}` placeholders during pattern expansion.
# Matched as a wildcard during scoring; mined out for slot extraction.
SLOT_WILDCARD = "\x00slot\x00"
