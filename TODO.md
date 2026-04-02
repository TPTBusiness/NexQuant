# TODOs

This file tracks global TODOs for the Predix project. Individual TODOs in code should be addressed in their respective modules.

## Current Global TODOs

### Low Priority (Post-v1.0.0)

- [ ] Align naming conventions for files in `components/` and `scenarios/` directories
  - Current mismatch: `coder` in `components/` vs `developer` in `scenarios/`
  - **Status:** Inherited from RD-Agent upstream, works as-is
  - **Priority:** Low - cosmetic only, no functional impact

- [ ] Clean up folder naming inconsistencies
  - Investigate why some scenario-related code is in `experiments/` folders
  - **Status:** Historical structure from RD-Agent
  - **Priority:** Low - works correctly, can refactor later

## Completed

### v1.0.0 Release (2026-04-02)

- [x] Rebrand from RD-Agent to Predix for EUR/USD focus
- [x] Remove Microsoft-specific references
- [x] Update documentation for PredixAI organization
- [x] Translate all code comments to English
- [x] Create comprehensive QWEN.md (development guide)
- [x] Create ATTRIBUTION.md (usage guidelines)
- [x] Update README.md with installation and quick start
- [x] Create CHANGELOG.md with v1.0.0 release notes
- [x] Create changelog/ directory structure (changelog/v1.0.0.md)
- [x] Prepare GitHub Release v1.0.0
- [x] Pass all tests (97/97, 98.77% coverage)
- [x] Complete .gitignore for sensitive files
- [x] Clean up git history (remove logs, test artifacts)
- [x] Add English-only commit message policy to QWEN.md
- [x] Add comprehensive Acknowledgments (RD-Agent, TradingAgents, ai-hedge-fund)

### Future Releases

- [ ] v1.1.0: Live paper-trading integration
- [ ] v1.1.0: Additional ML-based factor types
- [ ] v1.2.0: Docker containerization
- [ ] v1.2.0: CI/CD pipeline
- [ ] v1.3.0: Performance optimization
