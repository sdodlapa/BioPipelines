# BioPipelines Codebase Cleanup Plan

**Created**: December 4, 2025  
**Status**: Phase 1 Complete ✅  
**Author**: AI Assistant (Claude)  
**Last Updated**: December 4, 2025

---

## Completed Cleanups

### ✅ December 4, 2025 - Phase 1 Implemented
1. **Renamed duplicate class** in `arbiter.py`: `UnifiedIntentParser` → `SimpleArbiterParser`
2. **Deleted unused directory**: `web/archive/` (4 files removed)
3. **Verified imports**: All core imports working correctly

---

## Executive Summary

This document outlines the cleanup of redundant, deprecated, and legacy code in the BioPipelines codebase. Each task is verified before implementation to ensure we only remove truly unused code.

---

## Pre-Implementation Verification Checklist

Before removing any code, we verify:
1. ✅ No active imports from the code
2. ✅ No references in tests (or tests are also deprecated)
3. ✅ Functionality is available through replacement code
4. ✅ Documentation exists for migration path

---

## Task 1: Duplicate `UnifiedIntentParser` Class

### Problem
Two classes with identical names exist:
- `agents/intent/unified_parser.py:136` - Full-featured (1100+ lines)
- `agents/intent/arbiter.py:589` - Simpler version (200 lines)

### ✅ Verification Results
- [x] `agents/intent/__init__.py` imports from `unified_parser.py` (line 93-95)
- [x] `UnifiedAgent` uses the one from `unified_parser.py` via `__init__.py`
- [x] No code imports `UnifiedIntentParser` from `arbiter.py` directly
- [x] The arbiter.py version is only used internally within that file

### Proposed Action
Rename the class in `arbiter.py` from `UnifiedIntentParser` to `SimpleArbiterParser` to avoid confusion.
This is **SAFE** - the class is not exported or imported externally.

---

## Task 2: Triple `ProviderType` Enum

### Problem
Three separate definitions with inconsistent values:
- `llm/providers/base.py:44` → `LOCAL="local"`, `CLOUD="cloud"`
- `providers/registry.py:22` → `API="api"`, `LOCAL="local"`
- `models/registry.py:18` → `API="api"`, `LOCAL="local"`

### Verification Steps
- [ ] Map all usages of each ProviderType
- [ ] Determine which enum is the "canonical" one
- [ ] Check if the different values (CLOUD vs API) serve different purposes

### Proposed Action
Keep all three for now - they serve different contexts:
- `llm/providers/base.py` - For LLM routing (local GPU vs cloud API)
- `providers/registry.py` - For general provider registry
- `models/registry.py` - For model configuration

Add documentation clarifying their different purposes rather than consolidating.

### Alternative
If truly redundant after verification, consolidate into `infrastructure/enums.py`.

---

## Task 3: Archived Agent Components (`agents/_archived/`)

### Problem
Directory contains deprecated code:
- `bridge.py` - Legacy AgentBridge
- `router.py` - Legacy AgentRouter  
- `context/` - Legacy context management

### ✅ Verification Results
- [x] Imports exist ONLY in backward-compat layer (`agents/__init__.py` lines 53, 63)
- [x] These imports emit `DeprecationWarning` when accessed
- [x] `AgentRouter` in `test_human_handoff.py` is a DIFFERENT class (handoff routing, not archived)
- [x] Test file `tests/_archived/test_agentic_router.py` is also archived with README
- [x] `UnifiedAgent` provides equivalent functionality

### Proposed Action
**DO NOT DELETE YET** - The backward-compat imports with deprecation warnings are the correct approach.
The code is properly archived and will emit warnings if anyone uses it.

Keep for one more release cycle, then delete both:
- `src/workflow_composer/agents/_archived/`
- `tests/_archived/`

---

## Task 4: Archived Web Components (`web/archive/`)

### Problem
Old web implementations:
- `api.py` - Legacy FastAPI backend (600 lines)
- `app.py` - Legacy Flask UI (713 lines)
- `result_browser.py` - Old result browser
- `unified_workspace.py` - Old workspace manager

### ✅ Verification Results
- [x] NO imports from `web.archive` in any Python file
- [x] `web/__init__.py` only imports from `app.py` and `utils.py` (not archive)
- [x] Code quality report shows these files have unused imports (dead code)
- [x] Gradio `web/app.py` provides all current functionality
- [x] No Docker/deployment configs reference these files

### Proposed Action
**SAFE TO DELETE** - Remove `web/archive/` directory entirely.
These files are truly unused and have clear replacements.

---

## Task 5: Deprecated TOOL_PATTERNS

### Problem
In `agents/tools/base.py:89-107`, patterns marked as deprecated:
```python
# LEGACY TOOL_PATTERNS - DEPRECATED
# These patterns will be removed in a future version.
TOOL_PATTERNS = [...]
```

### ✅ Verification Results
- [x] `TOOL_PATTERNS` is imported in `agents/tools/__init__.py` line 22
- [x] `TOOL_PATTERNS` is re-exported in `__all__` (line 1578)
- [x] The ACTIVE patterns are `ALL_TOOL_PATTERNS` (line 219) - different variable
- [x] `AgentTools.detect_tool()` uses `ALL_TOOL_PATTERNS`, NOT the legacy one (line 910)
- [x] Legacy `TOOL_PATTERNS` appears to be a simpler fallback list

### Proposed Action
**SAFE TO REMOVE** the legacy `TOOL_PATTERNS` constant since:
1. `ALL_TOOL_PATTERNS` is the comprehensive list actually used
2. The import and export can be removed
3. The deprecation comment says "will be removed in future version"

OR keep with deprecation warning if external code might use it.

---

## Task 6: Legacy LLM Adapter Layer

### Problem
Two adapter systems coexist:
- `llm/*.py` - Legacy adapters (OllamaAdapter, OpenAIAdapter, etc.)
- `providers/*.py` - New provider system

### Verification Steps
- [ ] Check usage count of legacy adapters
- [ ] Check usage count of new providers
- [ ] Verify feature parity
- [ ] Check `get_llm()` factory function

### Proposed Action
**Do NOT remove** - These are actively used and marked as "still supported".
Add documentation clarifying:
- Legacy adapters: For direct LLM access
- Providers: For orchestrated/routed access

---

## Task 7: Duplicate `providers/` Directories

### Problem
Two provider directories:
- `llm/providers/` - Unified local/cloud providers
- `providers/` - Individual provider implementations

### Verification Steps
- [ ] Map imports from each
- [ ] Check if they serve different purposes
- [ ] Identify any circular dependencies

### Proposed Action
Keep both - they serve different architectural layers:
- `llm/providers/` - Abstract provider interface for LLM orchestrator
- `providers/` - Concrete implementations for different services

Document the distinction in `README.md` files.

---

## Implementation Order

### Phase 1: Safe Cleanups (Verified Safe)
1. ✅ Task 1: Rename duplicate `UnifiedIntentParser` in arbiter.py → `SimpleArbiterParser` **[DONE]**
2. ✅ Task 4: Delete `web/archive/` directory (no imports found) **[DONE]**
3. ⚠️ Task 5: Remove legacy `TOOL_PATTERNS` (or add deprecation warning) **[OPTIONAL]**

### Phase 2: Keep With Documentation
4. ⏸️ Task 3: Keep `agents/_archived/` - deprecation warnings already in place
5. ⏸️ Task 2: Keep triple `ProviderType` - they serve different purposes

### Phase 3: Documentation Only
6. ⏸️ Task 6: Document adapter layers (no code changes)
7. ⏸️ Task 7: Document provider directories (no code changes)

---

## Verification Commands

```bash
# Find all imports of a module
grep -r "from.*_archived" src/ tests/ --include="*.py"
grep -r "import.*_archived" src/ tests/ --include="*.py"

# Find all usages of a class
grep -r "UnifiedIntentParser" src/ tests/ --include="*.py"

# Find all usages of an enum
grep -r "ProviderType" src/ tests/ --include="*.py"

# Check for web.archive imports
grep -r "from.*web\.archive" src/ tests/ --include="*.py"
grep -r "web/archive" . --include="*.py"
```

---

## Success Criteria

- [ ] All tests pass after cleanup
- [ ] No new deprecation warnings in normal operation
- [ ] Documentation updated for any API changes
- [ ] Git history preserved for reverted code

---

## Appendix: Files to Review

### Definitely Remove (After Verification)
- `web/archive/api.py`
- `web/archive/app.py`
- `web/archive/result_browser.py`
- `web/archive/unified_workspace.py`

### Possibly Rename
- `agents/intent/arbiter.py` class `UnifiedIntentParser` → `ArbiterIntentParser`

### Keep But Document
- `llm/*.py` legacy adapters
- `providers/*.py` 
- `llm/providers/*.py`
- Triple `ProviderType` enums

### Keep But Add Warnings
- `agents/tools/base.py` TOOL_PATTERNS
