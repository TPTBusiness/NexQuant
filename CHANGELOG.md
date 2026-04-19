# Changelog

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
