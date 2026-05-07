"""
Constants for the closest-intent custom component.
"""

DOMAIN = "closest_intent"

CONF_THRESHOLD = "threshold"
CONF_EXPANSION_CAP = "expansion_cap"
CONF_DENYLIST = "denylist"
CONF_INCLUDE_BUILTINS = "include_builtins"
CONF_SLOT_EXTRACTION = "slot_extraction"
CONF_FALLBACK_AGENT = "fallback_agent"

DEFAULT_THRESHOLD = 70
DEFAULT_EXPANSION_CAP = 16
DEFAULT_INCLUDE_BUILTINS = False
DEFAULT_SLOT_EXTRACTION = True
# Fallback conversation agent, used only when hassil errors or returns no
# intent match. The canonical sentence itself always goes to hassil first.
# Be careful not to create a loop...
DEFAULT_FALLBACK_AGENT = "conversation.home_assistant"

# Stash keys in `hass.data[DOMAIN]`.
KEY_CONVERSATION_INTENTS = "_conversation_intents"
KEY_CONVERSATION_LISTS = "_conversation_lists"
KEY_CONVERSATION_EXPANSION_RULES = "_conversation_expansion_rules"
KEY_AGENT_INSTANCES = "_agent_instances"

SERVICE_DUMP_CANDIDATES = "dump_candidates"
SERVICE_PARSE = "parse_sentence"

# Hard ceiling on candidates kept per intent after pattern expansion.
PER_INTENT_CANDIDATE_CAP = 32

# Marker substituted in for `{slot}` placeholders during pattern expansion.
# Matched as a wildcard during scoring; mined out for slot extraction.
SLOT_WILDCARD = "\x00slot\x00"
