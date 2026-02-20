"""Table detection, LLM-based reconstruction, and markdown formatting.

Submodules:
  patterns     -- compiled regex patterns and constant tuples
  classifiers  -- paragraph classification helpers
  schema       -- TableStructure Pydantic model
  detection    -- table boundary detection, content extraction, interruption detection
  formatting   -- LLM client, cache, markdown rendering
  pipeline     -- main run() entry point and dict utilities
"""
