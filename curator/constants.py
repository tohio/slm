"""
curator/constants.py
---------------------
Curator-side constants.
 
These values previously lived here directly. They are now defined once in
config.data_mix and re-exported from this module so existing importers
(`from curator.constants import CHARS_PER_TOKEN, CC_CHARS_PER_SEGMENT`)
continue to work unchanged.
 
Do not add new constants here — put them in config/data_mix.py. This
module will eventually be removed; the re-exports exist only so the
curator refactor is drop-in compatible with old imports.
"""
 
from config import CHARS_PER_TOKEN, CC_CHARS_PER_SEGMENT
 
__all__ = ["CHARS_PER_TOKEN", "CC_CHARS_PER_SEGMENT"]