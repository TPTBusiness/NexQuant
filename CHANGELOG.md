# Changelog

## [1.4.1](https://github.com/TPTBusiness/Predix/compare/v1.4.0...v1.4.1) (2026-05-03)


### Bug Fixes

* 15 bug fixes across orchestrator, runner, backtest, and infrastructure ([163687d](https://github.com/TPTBusiness/Predix/commit/163687d7e1c278a085d7052a3f958a3edb501e77))
* also catch ValueError in mean_variance for dimension mismatch ([ed73b72](https://github.com/TPTBusiness/Predix/commit/ed73b7253f7dc6459ee30dd81a1ce1194e46e9af))
* close log file handle, fix FTMO equity double-count, remove bare except ([76219a5](https://github.com/TPTBusiness/Predix/commit/76219a53efddaafc2b8bd48a0f76c1d4325e6ea5))
* correct project root paths and subprocess handling in parallel runner and CLI ([9735e3a](https://github.com/TPTBusiness/Predix/commit/9735e3a4d8f01e7b16fb9b185a002396a915cea4))
* filter NaN in max(), remove redundant ternary, handle non-finite vbt results ([f89fbb3](https://github.com/TPTBusiness/Predix/commit/f89fbb3421faf6ccdc8e68a911fd9db2c166120f))
* fix type annotation, remove unused parameter, improve import_class errors ([8b6ab73](https://github.com/TPTBusiness/Predix/commit/8b6ab735c05629bf6b76ddc2fd8b15617600cad7))
* resolve dead code, shell injection risk, mutable defaults, and other bugs ([afff262](https://github.com/TPTBusiness/Predix/commit/afff26287f7c4df7ddfde4e816d280fe845e11eb))
* resolve unbound variable, logger shadowing, withdraw_loop edge case, and other bugs in main scripts ([748cf9b](https://github.com/TPTBusiness/Predix/commit/748cf9b214a3e8447f1289fc4cf1e92ad6cc2f1a))

## [1.4.0](https://github.com/TPTBusiness/Predix/compare/v1.3.11...v1.4.0) (2026-05-01)


### Features

* **optimizer:** add max_positions parameter to Optuna search space ([fdb4be3](https://github.com/TPTBusiness/Predix/commit/fdb4be3b3ebd93325e7821f4251148424184a40d))

## [1.3.11](https://github.com/TPTBusiness/Predix/compare/v1.3.10...v1.3.11) (2026-05-01)


### Bug Fixes

* **ci:** lazy import logger in predix.py and cli.py to avoid ImportError in test env ([60763e8](https://github.com/TPTBusiness/Predix/commit/60763e8eae34f41865ba8e5e65bdfde13b564b4b))

## [1.3.10](https://github.com/TPTBusiness/Predix/compare/v1.3.9...v1.3.10) (2026-05-01)


### Bug Fixes

* **security:** replace remaining assert statements with proper error handling ([928533d](https://github.com/TPTBusiness/Predix/commit/928533d9a81bd5062f07458fbf94d3c7fe347775))

## [1.3.9](https://github.com/TPTBusiness/Predix/compare/v1.3.8...v1.3.9) (2026-05-01)


### Bug Fixes

* **security:** resolve path-injection, B701, B101, B112 Bandit alerts ([20b89a0](https://github.com/TPTBusiness/Predix/commit/20b89a061843b39836e975f158404e8e2d4627cd))

## [1.3.8](https://github.com/TPTBusiness/Predix/compare/v1.3.7...v1.3.8) (2026-04-30)


### Bug Fixes

* **deps:** relax aiohttp constraint to &gt;=3.13.4 for litellm compatibility ([34ab192](https://github.com/TPTBusiness/Predix/commit/34ab1923a887089eb36e5cbad6cb8df16f0333ca))
* **qlib:** correct indentation in except blocks in quant_proposal and factor_runner ([8143451](https://github.com/TPTBusiness/Predix/commit/8143451e8c0ead01c4d86d19669268c7bfb15fac))
* **security:** replace eval() with ast.literal_eval in finetune validator (B307) ([0508caf](https://github.com/TPTBusiness/Predix/commit/0508caf9140d210b823fefefa28ee535ec85a0ae))
* **security:** replace shell=True subprocess calls with list args in env.py (B602) ([2012d5a](https://github.com/TPTBusiness/Predix/commit/2012d5ae4e77cc2f1ab9a48beaaac5a74695d083))
* **security:** resolve path-injection and add nosec for safe temp paths (B108, py/path-injection) ([6727480](https://github.com/TPTBusiness/Predix/commit/67274803bd1d14e5d1df9a063f46b2edb8501a2b))

## [1.3.7](https://github.com/TPTBusiness/Predix/compare/v1.3.6...v1.3.7) (2026-04-30)


### Bug Fixes

* **security:** nosec for B608/B701 false positives in UI and template code ([5eb5d7e](https://github.com/TPTBusiness/Predix/commit/5eb5d7e8fdbe90e0dced83fef4e09f5a33e96b2b))
* **security:** replace eval() with ast.literal_eval and add request timeouts (B307, B113) ([3301ada](https://github.com/TPTBusiness/Predix/commit/3301ada697ca7d3afa1a188d2a76a87ae98b4529))
* **security:** replace shell=True subprocess calls with list args (B602) ([13c08f4](https://github.com/TPTBusiness/Predix/commit/13c08f4ce6813eb7c314087921ec8c0f40074bd7))

## [1.3.6](https://github.com/TPTBusiness/Predix/compare/v1.3.5...v1.3.6) (2026-04-30)


### Bug Fixes

* **security:** real fix for B110 (logging in factor_proposal.py [#746](https://github.com/TPTBusiness/Predix/issues/746)) ([16624e0](https://github.com/TPTBusiness/Predix/commit/16624e0bd966ae4d24c4a3eb42bbc31c11da3136))
* **security:** real fix for B110 (logging in factor_runner.py [#744](https://github.com/TPTBusiness/Predix/issues/744)) ([88cf0fb](https://github.com/TPTBusiness/Predix/commit/88cf0fb8828b11c97f2f3ae2881a4900b020c6f0))
* **security:** real fix for B110 (logging in quant_proposal.py [#741](https://github.com/TPTBusiness/Predix/issues/741)) ([7cf2a64](https://github.com/TPTBusiness/Predix/commit/7cf2a644f553b054bd4b0607ea51e5372e68d90a))
* **security:** real fix for B110 (logging in quant_proposal.py [#741](https://github.com/TPTBusiness/Predix/issues/741)) ([ef985f8](https://github.com/TPTBusiness/Predix/commit/ef985f86035d8dca707c60137e6508349a0c4ae6))
* **security:** real fix for B404/B603 (sys.executable in factor_runner.py [#745](https://github.com/TPTBusiness/Predix/issues/745)) ([819655a](https://github.com/TPTBusiness/Predix/commit/819655aaa3efa76596d60501d0e8ca365df3e5e2))
* **security:** revert broken read_pickle encoding arg in kaggle template (B301) ([3574907](https://github.com/TPTBusiness/Predix/commit/35749073c91e69f63ddaad61dae3f2b799327e63))
* **security:** validate SQL identifiers in _add_column_if_not_exists (B608) ([e10dfa2](https://github.com/TPTBusiness/Predix/commit/e10dfa2576038e911f83595d3b466c261bc0cd54))
* **security:** whitelist-validate metric column in get_top_factors (B608) ([e50519f](https://github.com/TPTBusiness/Predix/commit/e50519fe066e68aec2f19b83df4f643c3c22053d))

## [1.3.5](https://github.com/TPTBusiness/Predix/compare/v1.3.4...v1.3.5) (2026-04-27)


### Bug Fixes

* **auto-fixer:** add five new factor code fixes for groupby/apply errors ([449c8fd](https://github.com/TPTBusiness/Predix/commit/449c8fd70a327e604dcca122e4a134f0cca918e4))
* **auto-fixer:** add four new factor code fixes for common runtime errors ([40484f6](https://github.com/TPTBusiness/Predix/commit/40484f6d300425da481f1edd325da4acbc06ec7d))
* **auto-fixer:** add groupby([level=N,'date']) SyntaxError fix ([ca77c00](https://github.com/TPTBusiness/Predix/commit/ca77c005bea4abdd8854c1de2b0e8d03b7742161))
* **auto-fixer:** disable _fix_min_periods for intraday data ([77b0740](https://github.com/TPTBusiness/Predix/commit/77b0740f059349df7e769a378af728aa33b2070e))
* **auto-fixer:** fix chained groupby(level=N).groupby('date') pattern ([7d5fe32](https://github.com/TPTBusiness/Predix/commit/7d5fe32b31a19ce8b04bd8f5a430720fdb748f7a))
* **auto-fixer:** fix df.loc[instrument] DateParseError on MultiIndex frames ([b7860ea](https://github.com/TPTBusiness/Predix/commit/b7860eafc0ad26384947ce0510ecf4e9f3425807))
* **auto-fixer:** fix df['instrument'] KeyError on MultiIndex frames ([aad6bd1](https://github.com/TPTBusiness/Predix/commit/aad6bd1c7c720b3d486e0cf248337f32394773b1))
* **auto-fixer:** fix two assignment-target bugs in instrument column fixers ([421eedf](https://github.com/TPTBusiness/Predix/commit/421eedffed4b883c24397dc5581c019a3985277f))
* **auto-fixer:** preserve date dimension in groupby(['instrument','date']) fix ([b58fdd8](https://github.com/TPTBusiness/Predix/commit/b58fdd8be43720b5d4363e0f8de9a01591d4d2dc))
* **auto-fixer:** remove ddof from rolling() args, not only from std()/var() ([b0fc328](https://github.com/TPTBusiness/Predix/commit/b0fc328d0d4a041c65d8eeb32cb3f2bb86568406))
* **auto-fixer:** strip spurious .reset_index() after .transform() calls ([8708aae](https://github.com/TPTBusiness/Predix/commit/8708aae6e08728cda1875c775a76dc92e43576f3))
* **loop:** prevent step_idx advance on unhandled exceptions + fix consecutive assistant messages ([5ec4ad1](https://github.com/TPTBusiness/Predix/commit/5ec4ad1b96b5b99ef42bea7bb828cb1ef709a688))

## [1.3.4](https://github.com/TPTBusiness/Predix/compare/v1.3.3...v1.3.4) (2026-04-27)


### Bug Fixes

* **auto-fixer:** add five new factor code fixes for groupby/apply errors ([449c8fd](https://github.com/TPTBusiness/Predix/commit/449c8fd70a327e604dcca122e4a134f0cca918e4))
* **auto-fixer:** add four new factor code fixes for common runtime errors ([40484f6](https://github.com/TPTBusiness/Predix/commit/40484f6d300425da481f1edd325da4acbc06ec7d))
* **auto-fixer:** add groupby([level=N,'date']) SyntaxError fix ([ca77c00](https://github.com/TPTBusiness/Predix/commit/ca77c005bea4abdd8854c1de2b0e8d03b7742161))
* **auto-fixer:** disable _fix_min_periods for intraday data ([77b0740](https://github.com/TPTBusiness/Predix/commit/77b0740f059349df7e769a378af728aa33b2070e))
* **auto-fixer:** fix chained groupby(level=N).groupby('date') pattern ([7d5fe32](https://github.com/TPTBusiness/Predix/commit/7d5fe32b31a19ce8b04bd8f5a430720fdb748f7a))
* **auto-fixer:** fix df.loc[instrument] DateParseError on MultiIndex frames ([b7860ea](https://github.com/TPTBusiness/Predix/commit/b7860eafc0ad26384947ce0510ecf4e9f3425807))
* **auto-fixer:** fix df['instrument'] KeyError on MultiIndex frames ([aad6bd1](https://github.com/TPTBusiness/Predix/commit/aad6bd1c7c720b3d486e0cf248337f32394773b1))
* **auto-fixer:** preserve date dimension in groupby(['instrument','date']) fix ([b58fdd8](https://github.com/TPTBusiness/Predix/commit/b58fdd8be43720b5d4363e0f8de9a01591d4d2dc))
* **auto-fixer:** remove ddof from rolling() args, not only from std()/var() ([b0fc328](https://github.com/TPTBusiness/Predix/commit/b0fc328d0d4a041c65d8eeb32cb3f2bb86568406))
* **backtest:** replace broken MC permutation test with binomial win-rate test ([c38d894](https://github.com/TPTBusiness/Predix/commit/c38d89478f586825bfca5715a96ca70ccd8791a3))
* **factors:** detect and correct look-ahead bias in daily-constant factors ([eb490a4](https://github.com/TPTBusiness/Predix/commit/eb490a461b66cbd815ae53ac5205115754712432))
* **factors:** extend look-ahead rules to session factors and add intraday-factor guidance ([c24c100](https://github.com/TPTBusiness/Predix/commit/c24c100442d6487686c0578de0b32d240fcbf215))
* **loop:** compress old experiment history in proposal prompt to reduce context size ([4bf90a9](https://github.com/TPTBusiness/Predix/commit/4bf90a905ba8b2aba2a818191c19998088cccaaf))
* **loop:** prevent step_idx advance on unhandled exceptions + fix consecutive assistant messages ([5ec4ad1](https://github.com/TPTBusiness/Predix/commit/5ec4ad1b96b5b99ef42bea7bb828cb1ef709a688))

## [1.3.3](https://github.com/TPTBusiness/Predix/compare/v1.3.2...v1.3.3) (2026-04-25)


### Bug Fixes

* **backtest:** replace broken MC permutation test with binomial win-rate test ([c38d894](https://github.com/TPTBusiness/Predix/commit/c38d89478f586825bfca5715a96ca70ccd8791a3))
* **factors:** detect and correct look-ahead bias in daily-constant factors ([eb490a4](https://github.com/TPTBusiness/Predix/commit/eb490a461b66cbd815ae53ac5205115754712432))
* **factors:** extend look-ahead rules to session factors and add intraday-factor guidance ([c24c100](https://github.com/TPTBusiness/Predix/commit/c24c100442d6487686c0578de0b32d240fcbf215))
* **loop:** compress old experiment history in proposal prompt to reduce context size ([4bf90a9](https://github.com/TPTBusiness/Predix/commit/4bf90a905ba8b2aba2a818191c19998088cccaaf))
* **strategies:** guard against None IC in acceptance check, disable slow wf_rolling ([2197f52](https://github.com/TPTBusiness/Predix/commit/2197f52150a50ef38d9e70991d7e48c8c30caec4))
* **strategies:** handle None ic/sharpe/dd in rejected strategy log output ([ad2ad3a](https://github.com/TPTBusiness/Predix/commit/ad2ad3ab3360ea75ed3bbc90c12098b9c5cc0114))

## [1.3.2](https://github.com/TPTBusiness/Predix/compare/v1.3.1...v1.3.2) (2026-04-23)


### Bug Fixes

* **strategies:** guard against None IC in acceptance check, disable slow wf_rolling ([2197f52](https://github.com/TPTBusiness/Predix/commit/2197f52150a50ef38d9e70991d7e48c8c30caec4))
* **strategies:** handle None ic/sharpe/dd in rejected strategy log output ([ad2ad3a](https://github.com/TPTBusiness/Predix/commit/ad2ad3ab3360ea75ed3bbc90c12098b9c5cc0114))

## [1.3.1](https://github.com/TPTBusiness/Predix/compare/v1.3.0...v1.3.1) (2026-04-21)


### Bug Fixes

* **deps:** bump python-dotenv to &gt;=1.2.2 (CVE symlink overwrite) ([126ae7d](https://github.com/TPTBusiness/Predix/commit/126ae7d5fb556b677d09d10221862a0d648d697a))

## [1.3.0](https://github.com/TPTBusiness/Predix/compare/v1.2.2...v1.3.0) (2026-04-21)


### Features

* **backtest:** add rolling walk-forward validation and Monte Carlo trade permutation test ([637a94c](https://github.com/TPTBusiness/Predix/commit/637a94c1d987da763869f4f9b73372a3f37d873c))


### Bug Fixes

* **security:** resolve all 30 Bandit security alerts (B301, B614, B104) ([ce5983d](https://github.com/TPTBusiness/Predix/commit/ce5983d9d59c4c34341fb1ec749e44bbcfc4a1c4))

## [1.2.2](https://github.com/TPTBusiness/Predix/compare/v1.2.1...v1.2.2) (2026-04-19)


### Documentation

* **claude:** auto-merge release-please PR after every push ([f500917](https://github.com/TPTBusiness/Predix/commit/f500917b699ee78dc676e84e01574d49bdc8e796))

## [2.2.0](https://github.com/TPTBusiness/Predix/compare/v2.1.0...v2.2.0) (2026-04-18)


### Features

* add Kronos CLI commands, expand tests, document in README ([f911081](https://github.com/TPTBusiness/Predix/commit/f911081d1763d0dc4dd790b57dd97aae2dc62679))
* **fin_quant:** auto-generate Kronos factor before loop start ([277063f](https://github.com/TPTBusiness/Predix/commit/277063f3e36cd071db859cdc77f69135c1f0763b))
* integrate Kronos-mini OHLCV foundation model (Option A + B) ([4ae3b99](https://github.com/TPTBusiness/Predix/commit/4ae3b99f2450930f72e202a1a470c407bfde3328))


### Bug Fixes

* **kronos:** lazy torch import to fix CI ModuleNotFoundError ([ccc1d27](https://github.com/TPTBusiness/Predix/commit/ccc1d27dbe5ab06a57085a589d456ac7bf49cc08))
* **kronos:** pass actual datetime Series to Kronos predictor timestamps ([dc6e7ce](https://github.com/TPTBusiness/Predix/commit/dc6e7ce207d21fbc21976f2af7691058530fac2f))
* **kronos:** replace rdagent_logger with stdlib logging for CI compatibility ([b4558f2](https://github.com/TPTBusiness/Predix/commit/b4558f2456659c6109bd1b3cf100510491cd3e6c))


### Performance Improvements

* **kronos:** batch GPU inference via predict_batch — 75x faster ([74611d0](https://github.com/TPTBusiness/Predix/commit/74611d071ac123a655eb15d0737bb73b8c1bd2b0))
* **kronos:** batch GPU inference via predict_batch — 75x faster ([2babeb9](https://github.com/TPTBusiness/Predix/commit/2babeb95f42828e13a37dc16166c75538f33fd4b))


### Documentation

* fix duplicate sections, add hardware requirements and data setup guide ([6c771b3](https://github.com/TPTBusiness/Predix/commit/6c771b37e6f88526a896499e86929cfca2c199eb))

## [2.1.0](https://github.com/TPTBusiness/Predix/compare/v2.0.0...v2.1.0) (2026-04-18)


### Features

* add daily log rotation, llama health wait, factor auto-fixer, and README updates ([4ae4d6f](https://github.com/TPTBusiness/Predix/commit/4ae4d6f0f1388d229e44333130306ae05767f2e5))
* Add GitHub infrastructure, CI/CD pipelines, and examples ([a0b5dc4](https://github.com/TPTBusiness/Predix/commit/a0b5dc464eaac831c76bdbf805cf60c9083e7d80))
* **factor-coder:** Add critical rules to prevent common factor implementation errors ([a1edca8](https://github.com/TPTBusiness/Predix/commit/a1edca87dd5e75ee402ea555f1b7a07b45c4b1f0))
* **logging:** write complete LLM prompts and responses to daily JSONL log ([803ef13](https://github.com/TPTBusiness/Predix/commit/803ef13052c645392e71aa5de24874aae83f62a7))
* **strategy:** Continuous optimization with Optuna parameter injection ([4fda5ea](https://github.com/TPTBusiness/Predix/commit/4fda5eaa31bc570e295ad96380ee2c02b82db706))
* unified backtest engine, LLM error handling, strategy refactor ([76b9341](https://github.com/TPTBusiness/Predix/commit/76b9341fe8ef0ff03fd911337c299cf0e8582f37))


### Bug Fixes

* Add critical column name rules to factor generation prompt ([3e74410](https://github.com/TPTBusiness/Predix/commit/3e7441079f0f1c5867829a365c6e45cd7d2071df))
* **ci:** fix closed-source asset check false positives in security workflow ([4b83c2b](https://github.com/TPTBusiness/Predix/commit/4b83c2bfe7e90c0c7a11116f07a1b989035b7a3f))
* **ci:** remove CodeQL workflow (conflicts with default setup), drop duplicate lint job ([a671361](https://github.com/TPTBusiness/Predix/commit/a671361ee4de9a7e00ccc66d8fd5732c2ed1fee9))
* **ci:** set JAVA_TOOL_OPTIONS UTF-8 in Codacy workflow ([e36721c](https://github.com/TPTBusiness/Predix/commit/e36721c765a02a325b8a7dfd3c262b2aca7b1652))
* **deps:** pin aiohttp&gt;=3.13.4 to patch 4 CVEs ([81adddc](https://github.com/TPTBusiness/Predix/commit/81adddcfcd14819a1f85c06288a663e7d222a8fb))
* **optuna:** fix inverted parameter range in Stage 2/3 when signal_bias is negative ([eaf885e](https://github.com/TPTBusiness/Predix/commit/eaf885ec2d20ebd93e34d1e2cb445532d2fb0ed3))
* **security:** Patch 5 CodeQL path injection and clear-text logging alerts ([#22](https://github.com/TPTBusiness/Predix/issues/22)-[#25](https://github.com/TPTBusiness/Predix/issues/25), [#9](https://github.com/TPTBusiness/Predix/issues/9)) ([d386af9](https://github.com/TPTBusiness/Predix/commit/d386af98205722d1ea6d1465f585e89cb8df47de))
* **security:** Patch 5 CodeQL path injection and weak hashing alerts ([#25](https://github.com/TPTBusiness/Predix/issues/25)-[#30](https://github.com/TPTBusiness/Predix/issues/30)) ([0d4c3b7](https://github.com/TPTBusiness/Predix/commit/0d4c3b7d69fdbdaafab00940bf7346c8b664928e))
* **security:** Patch path injection and stack trace exposure (CodeQL [#31](https://github.com/TPTBusiness/Predix/issues/31), [#27](https://github.com/TPTBusiness/Predix/issues/27)) ([b0b8432](https://github.com/TPTBusiness/Predix/commit/b0b84328d13dac5c2ef79961200b011c0b5778f1))
* **security:** replace relative_to() with realpath+startswith for CodeQL sanitization ([6d70f1e](https://github.com/TPTBusiness/Predix/commit/6d70f1ed944180c44d0eb75c0e86b013e5888b60))
* **security:** resolve CodeQL path-injection alerts in UI data loaders ([cced426](https://github.com/TPTBusiness/Predix/commit/cced426916cb726e95ad251dcbc0eb9ab6ec3591))
* **security:** resolve CodeQL path-injection and clear-text-logging alerts ([ec50224](https://github.com/TPTBusiness/Predix/commit/ec50224c3580c5c82ddba02fe77af95efd9667ea))
* **security:** Resolve GitHub Security Scan alerts ([6c85ba8](https://github.com/TPTBusiness/Predix/commit/6c85ba833a48326e39006e0f73c506b29a594bde))
* **security:** Upgrade vllm and transformers to patch 4 CVEs ([6c9ba91](https://github.com/TPTBusiness/Predix/commit/6c9ba91d3bf7ce1ed389e544c68be55262bf4e28))
* **strategy:** Fix template variables, APIBackend import, and JSON extraction ([8220faa](https://github.com/TPTBusiness/Predix/commit/8220faa3de6ea555717ac29ba90a3b68135fbf9e))
* **strategy:** Re-evaluate Optuna-optimized strategies with full OHLCV backtest ([026edce](https://github.com/TPTBusiness/Predix/commit/026edce122284fb1da467e6e9de8a2b9116c7ace))


### Documentation

* Add CLI welcome screenshot to README ([e6f2374](https://github.com/TPTBusiness/Predix/commit/e6f237437595745406c310b58a9bd7214ff914ae))
* Add comprehensive data setup guide to README ([f721d53](https://github.com/TPTBusiness/Predix/commit/f721d53e5681be6997418c13acc3439897168048))
* Add conda requirement to README + fix predix CLI ([df45698](https://github.com/TPTBusiness/Predix/commit/df45698b20e0a3e6e0079decf2b8eecb6983a175))
* Clean changelog of closed-source performance metrics ([a0f6587](https://github.com/TPTBusiness/Predix/commit/a0f6587ab1724293924da07fe18c40891ca612a1))
* improve README badges, fix llama-server flags, clean up structure ([336e1a5](https://github.com/TPTBusiness/Predix/commit/336e1a5afb4933ec13572ef050a3e5a2ca183400))
