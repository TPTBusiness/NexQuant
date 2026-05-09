# Changelog

## [1.6.0](https://github.com/TPTBusiness/NexQuant/compare/v1.5.0...v1.6.0) (2026-05-09)


### Features

* auto-post releases to Mastodon and X/Twitter via GitHub Actions ([9e896e5](https://github.com/TPTBusiness/NexQuant/commit/9e896e58550a5bb58756357804028f6b657f5098))
* integrate Kronos foundation model into fin_quant R&D loop ([ff1a9d6](https://github.com/TPTBusiness/NexQuant/commit/ff1a9d68484d3ef741fb66ba7b6f020a79c2fbd0))
* prioritize Kronos foundation model factors in strategy selection ([0c5e046](https://github.com/TPTBusiness/NexQuant/commit/0c5e046a40bd319ee56b3f5a313f1766d7d98df8))
* run Kronos on CPU to avoid GPU conflict with llama-server ([9cab556](https://github.com/TPTBusiness/NexQuant/commit/9cab556bb287287cff022efacd30ec1aa81054cd))
* support Kronos-small and Kronos-base models, auto-select GPU/CPU ([9eca3d4](https://github.com/TPTBusiness/NexQuant/commit/9eca3d468063ea7c63f688c487c7ccfbaa6909b8))


### Bug Fixes

* add hypothesis to test deps and fix missing imports in deep tests ([485e2de](https://github.com/TPTBusiness/NexQuant/commit/485e2de13a68644da75c01b02ce631246a8208c4))
* add missing debug() method to RDAgentLog ([1ce53f2](https://github.com/TPTBusiness/NexQuant/commit/1ce53f22232391ba2c6f268fd94a536ce7c4a3e9))
* prevent LLM retry loop from consecutive assistant message corruption ([c1e278d](https://github.com/TPTBusiness/NexQuant/commit/c1e278df6a7a03a40945bfce5ea6011d9258c625))
* restore KronosPredictor instantiation deleted during refactor ([4fee668](https://github.com/TPTBusiness/NexQuant/commit/4fee668dcb9b275959461eaf4adfe103cc84af04))


### Reverts

* remove automated release social workflow ([67ad8a4](https://github.com/TPTBusiness/NexQuant/commit/67ad8a471996055c78e6ce9e2dda6ff505faf0e2))


### Documentation

* add precise release trigger rules (5+ feats, 10+ fixes, 30+ commits, major feature, security) ([2221f87](https://github.com/TPTBusiness/NexQuant/commit/2221f875da3123ef1785ade80fdd48ada31c10f4))
* update README — Kronos-small, test depth, daemon setup, project structure ([d67b904](https://github.com/TPTBusiness/NexQuant/commit/d67b904d57b2f7361194d3654fed7743b973d43d))

## [0.8.0](https://github.com/TPTBusiness/NexQuant/compare/v1.4.2...v0.8.0) (2026-05-04)


### Features

* [AutoRL-Bench] Update DeepSearchQA split and translate task instructions to English ([#1368](https://github.com/TPTBusiness/NexQuant/issues/1368)) ([ffb9491](https://github.com/TPTBusiness/NexQuant/commit/ffb9491c4703290a5b292baa6328ae06bc520f9b))
* Add 'nexquant evaluate' command to CLI ([4308c25](https://github.com/TPTBusiness/NexQuant/commit/4308c257e7c83ab8ec5ef0a719b040f936bad0b3))
* Add 'nexquant top' command + explain factor evaluation results ([ac3334c](https://github.com/TPTBusiness/NexQuant/commit/ac3334c17d8dce48a5081e45d407ccadedfec713))
* Add 6 new CLI commands - all scripts integrated with local LLM ([e0dd07a](https://github.com/TPTBusiness/NexQuant/commit/e0dd07aa99ce33c2fc050d3d40b4520f245adb90))
* add a rag mcp in proposal ([#1267](https://github.com/TPTBusiness/NexQuant/issues/1267)) ([dc7b732](https://github.com/TPTBusiness/NexQuant/commit/dc7b732b2c428e3cca3373e839a0e724a844c79b))
* add a web UI server ([#1345](https://github.com/TPTBusiness/NexQuant/issues/1345)) ([1439548](https://github.com/TPTBusiness/NexQuant/commit/14395488b9c7ea476022a32211ea46de9925cf11))
* Add advanced ML models (Transformer, TCN, PatchTST, CNN+LSTM) ([44760f8](https://github.com/TPTBusiness/NexQuant/commit/44760f83c3d3d38033f5d94f4ba37dc0c25b7f59))
* Add AI Strategy Builder (StrategyCoSTEER) - Closed Source ([089189d](https://github.com/TPTBusiness/NexQuant/commit/089189d8ec058edefd0b81c2689b54f5180b9052))
* Add beautiful CLI welcome screen for GitHub README ([9e4a97d](https://github.com/TPTBusiness/NexQuant/commit/9e4a97d3d7e6d5328c4ffa39ce833591f10ab731))
* Add CLI model selection (local vs OpenRouter) ([c37935a](https://github.com/TPTBusiness/NexQuant/commit/c37935aa8c108a6bca393bcda274cda148101456))
* Add complete ML pipeline with graceful degradation (closed source) ([ed6b906](https://github.com/TPTBusiness/NexQuant/commit/ed6b906248ac3068a4f188d01bcde403e93abc0c))
* add daily log rotation, llama health wait, factor auto-fixer, and README updates ([2238fed](https://github.com/TPTBusiness/NexQuant/commit/2238fed701bd8a6ab1da1d3614d1c6d501e1ecbc))
* Add factor code and description to saved results ([b6b378d](https://github.com/TPTBusiness/NexQuant/commit/b6b378da8abf6f15be0c91e83508dc21d27b5b14))
* Add GitHub infrastructure, CI/CD pipelines, and examples ([26bd87e](https://github.com/TPTBusiness/NexQuant/commit/26bd87ed0a13da7190c8481356574bb710d00772))
* add improve_mode to MultiProcessEvolvingStrategy for selective task implementation ([#1273](https://github.com/TPTBusiness/NexQuant/issues/1273)) ([03f22dc](https://github.com/TPTBusiness/NexQuant/commit/03f22dc7c72a039ee6f1a0e8d0393f35117ec3e1))
* Add improved local prompt with MultiIndex code examples (v3) ([a729eb7](https://github.com/TPTBusiness/NexQuant/commit/a729eb715353961f71e92ddb679406c3c30b83d3))
* add Kronos CLI commands, expand tests, document in README ([24a51e4](https://github.com/TPTBusiness/NexQuant/commit/24a51e4322ef80d5f882697a930f1d1985aa5779))
* add LLM-finetune scenario ([#1314](https://github.com/TPTBusiness/NexQuant/issues/1314)) ([6e19c9e](https://github.com/TPTBusiness/NexQuant/commit/6e19c9e632cf07059c19993f2d4fbc772fb3cf13))
* add mask inference in debug mode ([#1154](https://github.com/TPTBusiness/NexQuant/issues/1154)) ([b4117cf](https://github.com/TPTBusiness/NexQuant/commit/b4117cf58a5618e1d9e92abb46e1c1dd98af5f13))
* Add model loader system (same as prompts) ([b7e397b](https://github.com/TPTBusiness/NexQuant/commit/b7e397b6f271e2cab5312f597cfbcb9652472298))
* add option to enable hyperparameter tuning only in first eval loop ([#1211](https://github.com/TPTBusiness/NexQuant/issues/1211)) ([f82de4a](https://github.com/TPTBusiness/NexQuant/commit/f82de4a380fa31a04a8494b196a743333aadf096))
* Add P5 ML Training Pipeline with LightGBM and 46 tests ([c934276](https://github.com/TPTBusiness/NexQuant/commit/c9342761ff8ab9adef69b65eb4cd8f206327fc97))
* Add parallel run system with API key distribution ([31fb7d5](https://github.com/TPTBusiness/NexQuant/commit/31fb7d56e3b6530091bef2c16e057a249caf4a93))
* add previous runner loops to runner history ([#1142](https://github.com/TPTBusiness/NexQuant/issues/1142)) ([2426a1d](https://github.com/TPTBusiness/NexQuant/commit/2426a1dc6700cc208360944cead9214a3da04889))
* add reasoning attribute to DSRunnerFeedback for enhanced evaluation context ([#1162](https://github.com/TPTBusiness/NexQuant/issues/1162)) ([bfa4525](https://github.com/TPTBusiness/NexQuant/commit/bfa452541c1422c02f77491e70927ce43f21810c))
* Add RL Trading Agent system with 99 tests ([0c4cb7a](https://github.com/TPTBusiness/NexQuant/commit/0c4cb7ad0c9842dd8fb73454bf554e9bedaf72f5))
* add runtime backtest verification (10 invariant checks in &lt;1ms) + 489 tests + README docs ([26db657](https://github.com/TPTBusiness/NexQuant/commit/26db65736431313bcdc27b6defde625db4133516))
* add show_hard_limit option and update time limit handling in DataScience settings ([#1144](https://github.com/TPTBusiness/NexQuant/issues/1144)) ([8a3e42d](https://github.com/TPTBusiness/NexQuant/commit/8a3e42d7fe8c36324c7578ede661297f2af59a37))
* Add simple factor evaluator with direct IC/Sharpe computation ([c7f23d0](https://github.com/TPTBusiness/NexQuant/commit/c7f23d026419060df3fcb3748740df8cc594bf39))
* Add start_llama and start_loop CLI commands ([c1d1844](https://github.com/TPTBusiness/NexQuant/commit/c1d184442aac79ca69b1e366bff7311973459869))
* add stdout into workspace for easier debugging ([#1236](https://github.com/TPTBusiness/NexQuant/issues/1236)) ([0daeb82](https://github.com/TPTBusiness/NexQuant/commit/0daeb82d6330e46edfeedc6b704b1a1c01d1a111))
* add time ratio limit for hyperparameter tuning in Kaggle settin… ([#1135](https://github.com/TPTBusiness/NexQuant/issues/1135)) ([6a49981](https://github.com/TPTBusiness/NexQuant/commit/6a4998154d000d95d7a5ec7cfb5e59305d4cbd11))
* Add Trading Protection System with 4 protections + comprehensive tests ([a9e0eff](https://github.com/TPTBusiness/NexQuant/commit/a9e0eff35d07c5b5223f64af343f8d2ece8d0053))
* add user interaction in data science scenario ([#1251](https://github.com/TPTBusiness/NexQuant/issues/1251)) ([6e09dc6](https://github.com/TPTBusiness/NexQuant/commit/6e09dc6d692f3ae2fcc0ffddf620e8f3e8dc1bd9))
* Auto-start dashboard for fin_quant ([3441604](https://github.com/TPTBusiness/NexQuant/commit/34416041c122b6a51ce94db1031f315c3639a4a5))
* Auto-start dashboard for fin_quant ([52d2b89](https://github.com/TPTBusiness/NexQuant/commit/52d2b8914815fa97d6b53b7cc7e817828520817e))
* **backtest:** add FTMO-realistic backtest mode with leverage, daily/total loss limits and realistic EUR/USD costs ([c5012e1](https://github.com/TPTBusiness/NexQuant/commit/c5012e1a1c7e5cff6c82bc42bd0ba34affb75c10))
* **backtest:** add rolling walk-forward validation and Monte Carlo trade permutation test ([d284d3e](https://github.com/TPTBusiness/NexQuant/commit/d284d3e74610c5f8ed314fa870cfb7f28a7681d4))
* **backtest:** add walk-forward OOS validation to backtest_signal_ftmo ([329841f](https://github.com/TPTBusiness/NexQuant/commit/329841f05a64ee9cdbaced2c4ec4de9436d3d42a))
* Backtesting Engine + Risk Management + Results Database ([cce889a](https://github.com/TPTBusiness/NexQuant/commit/cce889a1b7ee58f0042bc6c8cf01f5631ad45fa7))
* Backtesting Engine + Risk Management + Results DB ([86ef426](https://github.com/TPTBusiness/NexQuant/commit/86ef4269a350535871cb2f3f80d4d8e9e5c9258f))
* **backtest:** use backtest_signal_ftmo in strategy orchestrator and optuna optimizer ([994080e](https://github.com/TPTBusiness/NexQuant/commit/994080ef36e572f688b1d3cc219170bb340fc175))
* Beautiful CLI dashboard + corrected start command ([c2932cb](https://github.com/TPTBusiness/NexQuant/commit/c2932cb06904b041e1376d534309864d9d0e9122))
* Centralize all prompts in prompts/ directory ([3ff1ef8](https://github.com/TPTBusiness/NexQuant/commit/3ff1ef8557ef41d96b48c43efc2fe5795869fed0))
* CLI Commands for strategy generation (P4 complete) ([1f7ef1b](https://github.com/TPTBusiness/NexQuant/commit/1f7ef1b86f46153ff6e6cbde77e01c1ae08b905f))
* Complete P6-P9 implementation (73 tests) ([6981e91](https://github.com/TPTBusiness/NexQuant/commit/6981e9141d1f1f0951647971c10c1b9db227134a))
* continuous strategy generator (WF, MTF, stability, ML models, auto-ensemble) ([a206a31](https://github.com/TPTBusiness/NexQuant/commit/a206a31dbb831d6deed0492b73a9e246634fe074))
* create Jupyter notebook pipeline file based on main.py file ([#1134](https://github.com/TPTBusiness/NexQuant/issues/1134)) ([f03b1b9](https://github.com/TPTBusiness/NexQuant/commit/f03b1b918d32ec5a0ace1443d9f22e0c0598b2fc))
* Data Loader module with tests (P0 complete) ([af45cdf](https://github.com/TPTBusiness/NexQuant/commit/af45cdf074d7c3df02c535728ac55e69f214f1e3))
* Diverse factor selection + improved prompt v3 ([ea47f75](https://github.com/TPTBusiness/NexQuant/commit/ea47f75eda41398699f376219ec2c883c9d67798))
* enable finetune llm ([#1055](https://github.com/TPTBusiness/NexQuant/issues/1055)) ([35c209b](https://github.com/TPTBusiness/NexQuant/commit/35c209b09295d28d6d835c720fa1d300bdf43d13))
* enable LLM‑based hypothesis selection with time‑aware prompt & colored logging ([#1122](https://github.com/TPTBusiness/NexQuant/issues/1122)) ([90dd2f7](https://github.com/TPTBusiness/NexQuant/commit/90dd2f7b9bf49f5e1620e9d2c2eedf6c21f3e839))
* enable to inject diversity cross async multi-trace ([#1173](https://github.com/TPTBusiness/NexQuant/issues/1173)) ([b05a530](https://github.com/TPTBusiness/NexQuant/commit/b05a53012603c21847803e4709da10c5b868cab6))
* enable walk-forward OOS validation by default in backtest_signal_ftmo ([8853f8e](https://github.com/TPTBusiness/NexQuant/commit/8853f8e8e14ddabe510cb0ca271092f965b5ea81))
* enhance timeout handling in CoSTEER and DataScience scenarios ([#1150](https://github.com/TPTBusiness/NexQuant/issues/1150)) ([811d4e7](https://github.com/TPTBusiness/NexQuant/commit/811d4e7631dc83f228cd96a2a498803db46256a9))
* enhance timeout management and knowledge base handling in CoSTEER components ([#1130](https://github.com/TPTBusiness/NexQuant/issues/1130)) ([305eff1](https://github.com/TPTBusiness/NexQuant/commit/305eff1c5e36f3da5e93dc165105f50ccb990e32))
* EURUSD FX patches - prompts, factor spec, experiment settings ([b6cf687](https://github.com/TPTBusiness/NexQuant/commit/b6cf6874db995ea160457a1628a5691cbc8e5b97))
* EURUSD model experiment setting + model simulator text patched ([9a17b25](https://github.com/TPTBusiness/NexQuant/commit/9a17b25d32729453a28dd36246be4c5fdbd3a667))
* EURUSD Trading-Verbesserungen (Phase 2 & 3) ([05c4e1b](https://github.com/TPTBusiness/NexQuant/commit/05c4e1ba54b9259d6cc5f0af00a177d9295278a9))
* EURUSD Trading-Verbesserungen implementiert (Phase 1) ([b95bbf5](https://github.com/TPTBusiness/NexQuant/commit/b95bbf5900a9e06194ab0e330b662e2b853006ea))
* EURUSD walk-forward splits, bars terminology, README no $factor ([0eae7d0](https://github.com/TPTBusiness/NexQuant/commit/0eae7d0ababb422927dd0123118b97724d066ab0))
* **factor-coder:** Add critical rules to prevent common factor implementation errors ([e5c5d34](https://github.com/TPTBusiness/NexQuant/commit/e5c5d34eb5d38dd4bd18e9cd06026ba0e5a43344))
* fallback to acceptable results ([#1129](https://github.com/TPTBusiness/NexQuant/issues/1129)) ([7fc0916](https://github.com/TPTBusiness/NexQuant/commit/7fc09169bc5a779eeb650b799a43a36b44930a61))
* Fast mode - CoSTEER goes to backtest after 1 iteration ([fc830a2](https://github.com/TPTBusiness/NexQuant/commit/fc830a23bd31a53dab188847b10bf60430d396a8))
* **fin_quant:** auto-generate Kronos factor before loop start ([0daf7a8](https://github.com/TPTBusiness/NexQuant/commit/0daf7a8d2bdddd98a0c7d00959a39d4a38084a21))
* Fix 1min data integration and centralize all prompts ([2e94a4c](https://github.com/TPTBusiness/NexQuant/commit/2e94a4ce72cd9d0a01eef38c40ce70db1d158bb2))
* Fix realistic backtesting (Step 1+2) ([9b88ffb](https://github.com/TPTBusiness/NexQuant/commit/9b88ffbbd695d9486f25631ecf7f92457a23f6fc))
* Full auto strategy generation in fin_quant loop ([6d2990d](https://github.com/TPTBusiness/NexQuant/commit/6d2990dfff103e0cb85c0edd092457333d00c19e))
* Full system integration - RL + Protections + Backtesting + CLI ([60618d9](https://github.com/TPTBusiness/NexQuant/commit/60618d90f730470b7a9c57bf70c6f9fc45c36ad5))
* FX feedback loop, EURUSD ticker examples, bars terminology ([781779a](https://github.com/TPTBusiness/NexQuant/commit/781779a1f8c853eb77253053e23bc10c46dcf402))
* FX Multi-Agent Validator (TradingAgents-inspired) - Session/Macro/Bull-Bear/Trader ([cddfc53](https://github.com/TPTBusiness/NexQuant/commit/cddfc53ab07ca75b2364c30b9c2a794383633c2b))
* improve fallback handling in CoSTEER and add GPU usage guidelin… ([#1165](https://github.com/TPTBusiness/NexQuant/issues/1165)) ([9c190e3](https://github.com/TPTBusiness/NexQuant/commit/9c190e3268b4515945dcf5531dbaa222e843ceef))
* Improve nexquant portfolio command with robust error handling ([5051527](https://github.com/TPTBusiness/NexQuant/commit/505152793fe4a1629fa9ecdd8dc03ceb9bcd5db9))
* Improved LLM prompt + Optuna integration (Step 3+5) ([f72b07c](https://github.com/TPTBusiness/NexQuant/commit/f72b07ca94acd2b004f4a5b99faa8bb9ca1c7c76))
* init pydantic ai agent & context 7 mcp ([#1240](https://github.com/TPTBusiness/NexQuant/issues/1240)) ([5ba5e83](https://github.com/TPTBusiness/NexQuant/commit/5ba5e8356cbacb5e4bd9f24b26d6f9ac01784822))
* Integrate critical features into fin_quant workflow (P0+P1) ([484377b](https://github.com/TPTBusiness/NexQuant/commit/484377bc6dbe3bb216b1ebebb54978db371971cb))
* Integrate factor code/description saving into fin_quant process ([3b502e9](https://github.com/TPTBusiness/NexQuant/commit/3b502e9faeab4c7bbd185c9b107b7026b57330f0))
* integrate Kronos-mini OHLCV foundation model (Option A + B) ([165c156](https://github.com/TPTBusiness/NexQuant/commit/165c15684c7efe3db7de80b67eb301384d926739))
* Intelligent embedding chunking instead of truncation ([2d0584b](https://github.com/TPTBusiness/NexQuant/commit/2d0584b4cd7c1b3d9623acd6e141035d51f535fa))
* **logging:** write complete LLM prompts and responses to daily JSONL log ([1f83410](https://github.com/TPTBusiness/NexQuant/commit/1f83410fdd7e242b6cf4eb3aac045d8e6e6b7c70))
* **mcp:** cache with one-click toggle ([#1269](https://github.com/TPTBusiness/NexQuant/issues/1269)) ([4f493c8](https://github.com/TPTBusiness/NexQuant/commit/4f493c8d637dfda42f84af0dc08f8ecfc0501668))
* mcts policy based on trace scheduler ([#1203](https://github.com/TPTBusiness/NexQuant/issues/1203)) ([ac6d8ed](https://github.com/TPTBusiness/NexQuant/commit/ac6d8edad4366b08b5caf75e9a5ee8da0061a078))
* migrate to 1min EURUSD data (2020-2026) ([b39f2b7](https://github.com/TPTBusiness/NexQuant/commit/b39f2b7e46384c4fc56c1274c9120c470313262b))
* ML Training Pipeline with 46 tests (P5 complete) ([8f2aa83](https://github.com/TPTBusiness/NexQuant/commit/8f2aa8341932327dba5e260645bcf96efd5ed548))
* offline selector ([#1231](https://github.com/TPTBusiness/NexQuant/issues/1231)) ([d4c5399](https://github.com/TPTBusiness/NexQuant/commit/d4c539912abdb60e9d8950e7ea1186fd32bfeef3))
* optimize strategy generator (cache OHLCV, min_sharpe 1.5, nexquant generate-strategies CLI) ([def3975](https://github.com/TPTBusiness/NexQuant/commit/def39755793b16920c877045dd6628cb6a9aa9e8))
* **optimizer:** add max_positions parameter to Optuna search space ([f7b23b9](https://github.com/TPTBusiness/NexQuant/commit/f7b23b950f8f59b1b2efa66664ac2180ce136410))
* Optuna Parameter Optimizer with 60 tests (P3 complete) ([5583bf8](https://github.com/TPTBusiness/NexQuant/commit/5583bf874ed36886fa0d24e3472b8062abbd0b86))
* PDF performance reports for strategies (reportlab) ([b86e412](https://github.com/TPTBusiness/NexQuant/commit/b86e41209cd41e02de4ad3de3281b6558fdad059))
* nexquant.py wrapper for dashboard support ([757c66c](https://github.com/TPTBusiness/NexQuant/commit/757c66cddb18254220db1d571d9b739380c57f44))
* prob-based trace scheduler ([#1131](https://github.com/TPTBusiness/NexQuant/issues/1131)) ([7e15b5e](https://github.com/TPTBusiness/NexQuant/commit/7e15b5e2003628f40be12674a73197a956d86545))
* Realistic backtesting with OHLCV data (P5 continued) ([1506439](https://github.com/TPTBusiness/NexQuant/commit/1506439a1950a2e87cd662dfeec9e8b5fa1baf20))
* Realistic backtesting with OHLCV data and spread costs ([85a1e29](https://github.com/TPTBusiness/NexQuant/commit/85a1e2929acf0ea0f582a66f6261dd697f0260db))
* Redirect RD-Agent workspace to results/ directory ([fd2def0](https://github.com/TPTBusiness/NexQuant/commit/fd2def052a02e0f818a7cc705bdc2caaee2f01d2))
* refactor CoSTEER classes to use DSCoSTEER and update max seconds handling ([#1156](https://github.com/TPTBusiness/NexQuant/issues/1156)) ([c111966](https://github.com/TPTBusiness/NexQuant/commit/c111966d1975a4952c1266fb6d6af1c4f5fe83c1))
* refine the logic of enabling hyperparameter tuning and add criteira ([#1175](https://github.com/TPTBusiness/NexQuant/issues/1175)) ([e77572f](https://github.com/TPTBusiness/NexQuant/commit/e77572fb5347e40506fb7b5b25dd861e5f9ebb2b))
* **rl:** add AutoRL-Bench framework and benchmark integrations ([#1348](https://github.com/TPTBusiness/NexQuant/issues/1348)) ([7cd64a2](https://github.com/TPTBusiness/NexQuant/commit/7cd64a26fd84017042eb163e8eb4d3bd30c16de7))
* Save all factor results to results/factors/ ([2abbec9](https://github.com/TPTBusiness/NexQuant/commit/2abbec9fde67f52bcf1f199e7d18f7d99f04805e))
* Save factor results immediately after each evaluation ([72c5ec5](https://github.com/TPTBusiness/NexQuant/commit/72c5ec55f20964917fe9ed21a77f80e0394f61e8))
* **scripts:** add full file logging to strategy generation and rebacktest scripts ([c629af5](https://github.com/TPTBusiness/NexQuant/commit/c629af5b19df26330a131f510154fb5543709a66))
* show the summarized final difference between the final workspace and the base workspace ([#1281](https://github.com/TPTBusiness/NexQuant/issues/1281)) ([35a7ae5](https://github.com/TPTBusiness/NexQuant/commit/35a7ae5e1ff929b3ee3b77c04cb1f4a684a4b2d7))
* **strategies:** make OOS validation mandatory in strategy generator ([0f4c7c4](https://github.com/TPTBusiness/NexQuant/commit/0f4c7c4f46d4fd2fb8ff7c4b1eea58538c7db1b3))
* Strategy Generator working with local LLM (P0-P4) ([036edee](https://github.com/TPTBusiness/NexQuant/commit/036edeeb77d1a99a0a748a357038c6da3efdd5e7))
* Strategy Orchestrator with 30 tests (P2 complete) ([9af5cdb](https://github.com/TPTBusiness/NexQuant/commit/9af5cdbde4996b05a98e59c5c577e487e2d535bd))
* Strategy performance reports, CLI docs, and README update ([232e918](https://github.com/TPTBusiness/NexQuant/commit/232e918b48eabeed22e3b712048fb96089b99067))
* Strategy Worker module with 41 tests (P1 complete) ([b8acf82](https://github.com/TPTBusiness/NexQuant/commit/b8acf82ed26ffd131ca32bf5272547ff11bd5eef))
* **strategy:** Continuous optimization with Optuna parameter injection ([da90ae2](https://github.com/TPTBusiness/NexQuant/commit/da90ae271e46260910023f8a9e3798365b80b298))
* streamline hyperparameter tuning checks and update evaluation g… ([#1167](https://github.com/TPTBusiness/NexQuant/issues/1167)) ([5866230](https://github.com/TPTBusiness/NexQuant/commit/586623084f5d59d88645e75ceab6d795ec497cab))
* Support 25+ parallel runs with resource warnings ([7a4dd1a](https://github.com/TPTBusiness/NexQuant/commit/7a4dd1aa7454560d84993ee8827e005ee0795c37))
* ui, support disable cache ([#1217](https://github.com/TPTBusiness/NexQuant/issues/1217)) ([70fd91c](https://github.com/TPTBusiness/NexQuant/commit/70fd91cd051b2006df876ef6aa47a616058af95f))
* unified backtest engine, LLM error handling, strategy refactor ([1ddb114](https://github.com/TPTBusiness/NexQuant/commit/1ddb1142a2f21ed3a498292ac8f5af6bbc351e7c))
* update README with latest paper acceptance to NeurIPS 2025 ([#1252](https://github.com/TPTBusiness/NexQuant/issues/1252)) ([12969b4](https://github.com/TPTBusiness/NexQuant/commit/12969b491eafab626ce71f7e530458dab6f43246))
* zentrale data_config.yaml + apply_config.py für dynamische Datenkonfiguration ([b7c1e4d](https://github.com/TPTBusiness/NexQuant/commit/b7c1e4db8e29e960fe28393911d60fc0fd3ca413))


### Bug Fixes

* (to main) litellm's Timeout error is not picklable ([#1294](https://github.com/TPTBusiness/NexQuant/issues/1294)) ([315850e](https://github.com/TPTBusiness/NexQuant/commit/315850ea81761aa2478639ad32302d7a55f8181b))
* 15 bug fixes across orchestrator, runner, backtest, and infrastructure ([5ec4516](https://github.com/TPTBusiness/NexQuant/commit/5ec4516ed7bdc44f2fd7d6e3ec9df0a88fc4fd10))
* add a switch for ensemble_time_upper_bound and fix some bug in main ([#1226](https://github.com/TPTBusiness/NexQuant/issues/1226)) ([fc18942](https://github.com/TPTBusiness/NexQuant/commit/fc18942339b3ca59077ddc903f84b2d54193e5bc))
* Add Bandit security scanning and fix critical vulnerabilities ([f47dcf1](https://github.com/TPTBusiness/NexQuant/commit/f47dcf1c58d33041bba2f705b270a7f9c4e7d572))
* Add critical column name rules to factor generation prompt ([bf73725](https://github.com/TPTBusiness/NexQuant/commit/bf7372533e83da682f1ceefeddc70f142f8ccda2))
* Add get_factor_count() to QuantTrace to prevent parallel run crashes ([a16db77](https://github.com/TPTBusiness/NexQuant/commit/a16db77def1ba7adb7bb6734629086a1b5a901cb))
* add json format response fallback to prompt templates ([#1246](https://github.com/TPTBusiness/NexQuant/issues/1246)) ([694afd8](https://github.com/TPTBusiness/NexQuant/commit/694afd81331227d2be7f780f72023d00c0c9864e))
* add metric in scores.csv and avoid reading sample_submission.csv ([#1152](https://github.com/TPTBusiness/NexQuant/issues/1152)) ([80c953d](https://github.com/TPTBusiness/NexQuant/commit/80c953d4053dff66d12e4cf400b069d0fac16cbd))
* Add missing os import in factor_runner.py ([f201823](https://github.com/TPTBusiness/NexQuant/commit/f201823c44c724867163f3b2d3ecf49f384a8e35))
* Add missing Panel import in nexquant evaluate command ([e21923b](https://github.com/TPTBusiness/NexQuant/commit/e21923bd13eac6236a2c25d550bae0b984575491))
* add missing self parameter to instance methods in DSProposalV2ExpGen ([#1213](https://github.com/TPTBusiness/NexQuant/issues/1213)) ([c8bf617](https://github.com/TPTBusiness/NexQuant/commit/c8bf617aca57ea9c53d4a76d23806cb5ab5173ab))
* add missing sys import and fix undefined acc_rate in factor eval ([34323f3](https://github.com/TPTBusiness/NexQuant/commit/34323f307da6924095efcdaef81f99b95e2820eb))
* Add nosec comments for schema migration SQL in results_db.py ([3626b22](https://github.com/TPTBusiness/NexQuant/commit/3626b22482143466b0dec8b63ea0a4a36af06acf))
* allow prev_out keys to be None in workspace cleanup assertion ([#1214](https://github.com/TPTBusiness/NexQuant/issues/1214)) ([f02dc5f](https://github.com/TPTBusiness/NexQuant/commit/f02dc5f47d5973673bcc314ada89933a5d807d21))
* also catch ValueError in mean_variance for dimension mismatch ([daded85](https://github.com/TPTBusiness/NexQuant/commit/daded853b6370f0df6f83a6d1b3f04c0dd0757f0))
* **auto-fixer:** add five new factor code fixes for groupby/apply errors ([d03bcf3](https://github.com/TPTBusiness/NexQuant/commit/d03bcf3505f1be696e7bddc40f33c4a97b3f7486))
* **auto-fixer:** add four new factor code fixes for common runtime errors ([21ce0de](https://github.com/TPTBusiness/NexQuant/commit/21ce0def2dd8352a315e0688ebafc6d62cf0435e))
* **auto-fixer:** add groupby([level=N,'date']) SyntaxError fix ([d58eba3](https://github.com/TPTBusiness/NexQuant/commit/d58eba364e6ea14513b64e6bc12256c72111669a))
* **auto-fixer:** disable _fix_min_periods for intraday data ([665e490](https://github.com/TPTBusiness/NexQuant/commit/665e4903d8f6f3097a45d07060ab003ebea7f96b))
* **auto-fixer:** fix chained groupby(level=N).groupby('date') pattern ([9869839](https://github.com/TPTBusiness/NexQuant/commit/9869839a2c676ddd83f4218e9ff5e50fb8d2d223))
* **auto-fixer:** fix df.loc[instrument] DateParseError on MultiIndex frames ([87926dc](https://github.com/TPTBusiness/NexQuant/commit/87926dc41d795a3ab0670e585b99cc21dd09ae5f))
* **auto-fixer:** fix df['instrument'] KeyError on MultiIndex frames ([63a348e](https://github.com/TPTBusiness/NexQuant/commit/63a348eb3ec20c209c2d060e086bc69019e92884))
* **auto-fixer:** fix two assignment-target bugs in instrument column fixers ([a44eba9](https://github.com/TPTBusiness/NexQuant/commit/a44eba952e031e364050ee3d27a067d17fa01923))
* **auto-fixer:** preserve date dimension in groupby(['instrument','date']) fix ([37a2f37](https://github.com/TPTBusiness/NexQuant/commit/37a2f37f74118a2707a6b128d55c45ddb89cc48a))
* **auto-fixer:** remove ddof from rolling() args, not only from std()/var() ([daacbfd](https://github.com/TPTBusiness/NexQuant/commit/daacbfd141ae0da99c8c4cb01d5e500528eb7d80))
* **auto-fixer:** replace zero \$volume with price-range proxy for FX data ([7fcec39](https://github.com/TPTBusiness/NexQuant/commit/7fcec39f1d8f0f7668435f51a1a9646abcd9c89f))
* **auto-fixer:** strip spurious .reset_index() after .transform() calls ([c489616](https://github.com/TPTBusiness/NexQuant/commit/c489616d1a2fd71877a203d880e31281bc008cdf))
* avoid triggering errors like "RuntimeError: dictionary changed s… ([#1285](https://github.com/TPTBusiness/NexQuant/issues/1285)) ([b180543](https://github.com/TPTBusiness/NexQuant/commit/b18054371c6ce08c6bc322a7b0de41b67fc60408))
* **backtest:** replace broken MC permutation test with binomial win-rate test ([f284b7a](https://github.com/TPTBusiness/NexQuant/commit/f284b7a9751424201510c5938b4ebf6bd81842b6))
* cancel tasks on resume and kill subprocesses on termination ([#1166](https://github.com/TPTBusiness/NexQuant/issues/1166)) ([0e3f4cf](https://github.com/TPTBusiness/NexQuant/commit/0e3f4cf08f08e27f9c483a5bbe069313d0d8014e))
* change runner prompts ([#1223](https://github.com/TPTBusiness/NexQuant/issues/1223)) ([be3433f](https://github.com/TPTBusiness/NexQuant/commit/be3433f26b04054a482dfdc7cdd5c8c0a756a60c))
* **ci:** fix closed-source asset check false positives in security workflow ([1473085](https://github.com/TPTBusiness/NexQuant/commit/14730856636735c17d704854e057fa6e1aea5940))
* **ci:** lazy import logger in nexquant.py and cli.py to avoid ImportError in test env ([52d9ff0](https://github.com/TPTBusiness/NexQuant/commit/52d9ff0cd41d6fc6978e8af7f970cffd6a46f673))
* **ci:** remove CodeQL workflow (conflicts with default setup), drop duplicate lint job ([ab73425](https://github.com/TPTBusiness/NexQuant/commit/ab734252f356ac97dea4f70477ebe2fdee30509c))
* **ci:** remove env-print step to avoid leaking sensitive environment variables ([#1299](https://github.com/TPTBusiness/NexQuant/issues/1299)) ([c067ea6](https://github.com/TPTBusiness/NexQuant/commit/c067ea640030c67c549e3ca2dbad178f144e8b31))
* **ci:** set JAVA_TOOL_OPTIONS UTF-8 in Codacy workflow ([a9c6ea9](https://github.com/TPTBusiness/NexQuant/commit/a9c6ea99c9ebae2794b1c3f4d1e9da1d4e41376a))
* clear ws_ckp after extraction to reduce workspace object size ([#1137](https://github.com/TPTBusiness/NexQuant/issues/1137)) ([28ceb41](https://github.com/TPTBusiness/NexQuant/commit/28ceb41e1cdb603c4e0bd2fe7b72acef1b29ec47))
* CLI dashboard in separate terminal window ([b72cca9](https://github.com/TPTBusiness/NexQuant/commit/b72cca98680bd8a87393bb4e5f7d17aae47ab5ed))
* close log file handle, fix FTMO equity double-count, remove bare except ([4c76c85](https://github.com/TPTBusiness/NexQuant/commit/4c76c85b6509ddd7bbd5361f0823c5a41329591a))
* **collect_info:** parse package names safely from requirements constraints ([#1313](https://github.com/TPTBusiness/NexQuant/issues/1313)) ([99a71bf](https://github.com/TPTBusiness/NexQuant/commit/99a71bf533211df743b5801f913de788259e64cb))
* correct MaxDD to equity curve in strategy_builder; test: add 8 cross-validation tests for metric correctness ([7be98e8](https://github.com/TPTBusiness/NexQuant/commit/7be98e84c911c9ba08b444b33206553cbe60086d))
* correct project root paths and subprocess handling in parallel runner and CLI ([1c35a22](https://github.com/TPTBusiness/NexQuant/commit/1c35a2277ff601553e4733a8e990217dc9d6f989))
* correct Sharpe/MaxDD/WinRate in direct factor eval (was computing on raw factor, now on strategy returns) ([69122ee](https://github.com/TPTBusiness/NexQuant/commit/69122ee5c1819be6fababd701b88d0dbef993040))
* **deps:** bump python-dotenv to &gt;=1.2.2 (CVE symlink overwrite) ([f69333b](https://github.com/TPTBusiness/NexQuant/commit/f69333b27b9356f09e6cc2748cb45845732335c3))
* **deps:** pin aiohttp&gt;=3.13.4 to patch 4 CVEs ([a0b3b90](https://github.com/TPTBusiness/NexQuant/commit/a0b3b90bfdd1193f5b8be521f563d18ff17dd81c))
* **deps:** relax aiohttp constraint to &gt;=3.13.4 for litellm compatibility ([d3978fe](https://github.com/TPTBusiness/NexQuant/commit/d3978fec1305d7503a37ff576fdf953f75e1cd1d))
* Disable ANSI color codes when not running in TTY ([9db0e59](https://github.com/TPTBusiness/NexQuant/commit/9db0e590a4e94f538712cfec79f6cd470155050c))
* Disable Flask debug mode by default (Security Alert [#2](https://github.com/TPTBusiness/NexQuant/issues/2)) ([48c177f](https://github.com/TPTBusiness/NexQuant/commit/48c177fbafce7b111646c14a5c2e6e414414930b))
* Display litellm messages as info instead of warnings ([bd9d672](https://github.com/TPTBusiness/NexQuant/commit/bd9d672997aff80b5ad5c616b6486c11c2570b80))
* **dockerfile:** install coreutils to resolve timeout command error ([#1260](https://github.com/TPTBusiness/NexQuant/issues/1260)) ([35580cb](https://github.com/TPTBusiness/NexQuant/commit/35580cbdf87347d5d6105b2a9b5ad1694b695820))
* **docs:** update rdagent ui with correct params ([#1249](https://github.com/TPTBusiness/NexQuant/issues/1249)) ([3b9ad11](https://github.com/TPTBusiness/NexQuant/commit/3b9ad1145769862a24cc7533a1828f750f72170d))
* Embedding Context Length Error ([6d6c5ab](https://github.com/TPTBusiness/NexQuant/commit/6d6c5abd4ac7252257f88e13e263ecb2497fde3b))
* enable embedding truncation ([#1188](https://github.com/TPTBusiness/NexQuant/issues/1188)) ([880a6c7](https://github.com/TPTBusiness/NexQuant/commit/880a6c70c41024cb51f9fc4349ac7f1d2dbda434))
* end-timestamp 23:45,  weg, SZ-beispiele weg ([6a9ccd5](https://github.com/TPTBusiness/NexQuant/commit/6a9ccd5ddbf95060a2847bd27bcdae762a46a19d))
* enhance feedback handling in MultiProcessEvolvingStrategy for improved task evolution ([#1274](https://github.com/TPTBusiness/NexQuant/issues/1274)) ([afb575c](https://github.com/TPTBusiness/NexQuant/commit/afb575cc91114dbe41d8f582294dcc3692990695))
* Ensure backtest results save to DB and JSON files ([ae7b35e](https://github.com/TPTBusiness/NexQuant/commit/ae7b35ea2e0c71c76e8e454f7845df461d65b99f))
* evaluator erkennt 15min als valid (nicht daily) ([cf0f634](https://github.com/TPTBusiness/NexQuant/commit/cf0f634c17dce45400cc325ccd3ca45e769c15fd))
* **factors:** detect and correct look-ahead bias in daily-constant factors ([dcad0d1](https://github.com/TPTBusiness/NexQuant/commit/dcad0d1f68608a4db3cfdabb75e66c22490643aa))
* **factors:** extend look-ahead rules to session factors and add intraday-factor guidance ([8811dc0](https://github.com/TPTBusiness/NexQuant/commit/8811dc042a0a7a1ac385c7141ded9f56a434dced))
* filter NaN in max(), remove redundant ternary, handle non-finite vbt results ([1acfe50](https://github.com/TPTBusiness/NexQuant/commit/1acfe508a9c327dce8eba7a2ad1f618052a3e8a5))
* fix bug for hypo_select_with_llm when not support response_schema ([#1208](https://github.com/TPTBusiness/NexQuant/issues/1208)) ([d759ca9](https://github.com/TPTBusiness/NexQuant/commit/d759ca95e714a7a1476839a2a04bb652c0fbb863))
* fix chat_max_tokens calculation method to show true input_max_tokens ([#1241](https://github.com/TPTBusiness/NexQuant/issues/1241)) ([7e99605](https://github.com/TPTBusiness/NexQuant/commit/7e996055f2c7fd37595573ebdb13aa57c425a6cc))
* fix mcts ([#1270](https://github.com/TPTBusiness/NexQuant/issues/1270)) ([5003aff](https://github.com/TPTBusiness/NexQuant/commit/5003affb17505525336e6c30ba9c690b810c252b))
* Fix parallel runner dashboard rendering error ([3e8c07e](https://github.com/TPTBusiness/NexQuant/commit/3e8c07e728076a951528c4eb5b429653a5c77d14))
* fix some bugs in RD-Agent(Q) ([#1143](https://github.com/TPTBusiness/NexQuant/issues/1143)) ([7134a51](https://github.com/TPTBusiness/NexQuant/commit/7134a51afa71ab146b52987c194adace62f8b034))
* fix type annotation, remove unused parameter, improve import_class errors ([1eb5849](https://github.com/TPTBusiness/NexQuant/commit/1eb5849dd44c5953f7198212a5ef0dbe8c8d4881))
* Forward-fill daily factors to 1-min frequency ([20f4c21](https://github.com/TPTBusiness/NexQuant/commit/20f4c2140c397230fb56734b0e887b770db805ac))
* generate.py nutzt rdagent4qlib env für Qlib-Datenzugriff ([b9007f7](https://github.com/TPTBusiness/NexQuant/commit/b9007f754ac682800aaf265c0f24c2028d387d84))
* **graph:** using assignment expression to avoid repeated function call ([#1174](https://github.com/TPTBusiness/NexQuant/issues/1174)) ([b6fae75](https://github.com/TPTBusiness/NexQuant/commit/b6fae75cde256c9c8a84783dbd135a9bcca6ac8d))
* Handle failed experiments in feedback step to prevent crashes ([979ef66](https://github.com/TPTBusiness/NexQuant/commit/979ef66dc612c7f589e097dcdc3a01b742b18970))
* handle mixed str and dict types in code_list ([#1279](https://github.com/TPTBusiness/NexQuant/issues/1279)) ([32ecf92](https://github.com/TPTBusiness/NexQuant/commit/32ecf92afcf647f257b430c748cbe6bb5fa0fac4))
* Handle negative/zero values in performance report charts ([f4a4c65](https://github.com/TPTBusiness/NexQuant/commit/f4a4c65ce9bc1c929526a20a852765b92709011c))
* handle None output and conditional step dump in LoopBase execution ([#1212](https://github.com/TPTBusiness/NexQuant/issues/1212)) ([9de8d60](https://github.com/TPTBusiness/NexQuant/commit/9de8d6066994fcd7037fd03d9339b6590ab2fac9))
* Handle Qlib Docker backtest failures gracefully (SECURITY FIX) ([59f4561](https://github.com/TPTBusiness/NexQuant/commit/59f45618229be08dba028dceda21433cc5d52b9f))
* Handle timeout exceptions safely in nexquant_full_eval.py ([2738263](https://github.com/TPTBusiness/NexQuant/commit/27382635171482be2cee2e29d4793e63d14abce4))
* handle ValueError in stdout shrinking and refactor shrink logic ([#1228](https://github.com/TPTBusiness/NexQuant/issues/1228)) ([6fc3877](https://github.com/TPTBusiness/NexQuant/commit/6fc3877a39baabbf26e0cc1cbd327b0f6e2e325e))
* Harden _safe_resolve to fix CodeQL alert [#3](https://github.com/TPTBusiness/NexQuant/issues/3) ([0ed1a0a](https://github.com/TPTBusiness/NexQuant/commit/0ed1a0aa8faad6df36753a928f40a1cdbd606462))
* Harden path validation in Job Summary UI to fix CodeQL alert [#17](https://github.com/TPTBusiness/NexQuant/issues/17) ([7fe15d4](https://github.com/TPTBusiness/NexQuant/commit/7fe15d46cb2a740b6ec0ee37d29acaf37476e8e6))
* Harden path validation to fix CodeQL alert [#20](https://github.com/TPTBusiness/NexQuant/issues/20) ([59d06f6](https://github.com/TPTBusiness/NexQuant/commit/59d06f6588caadaa207bde1d135828c56169bff8))
* ignore case when checking metric name ([#1160](https://github.com/TPTBusiness/NexQuant/issues/1160)) ([1b84f7b](https://github.com/TPTBusiness/NexQuant/commit/1b84f7b7546a9dee4f27e24e07c49fa8ee3a370d))
* ignore RuntimeError for shared workspace double recovery ([#1140](https://github.com/TPTBusiness/NexQuant/issues/1140)) ([bd8a16d](https://github.com/TPTBusiness/NexQuant/commit/bd8a16d92f9176d835bbc27478f9259f0fe9a827))
* Import pandas in nexquant portfolio_simple command ([2b6de06](https://github.com/TPTBusiness/NexQuant/commit/2b6de06a612c147c414bde3175b6f11af1762f4d))
* Improve path traversal prevention with dedicated helper function ([50dc275](https://github.com/TPTBusiness/NexQuant/commit/50dc27566d886a4aea9ea56eaef2c08e794df770))
* increase retry count in hypothesis_gen decorator to 10 ([#1230](https://github.com/TPTBusiness/NexQuant/issues/1230)) ([86ce4f1](https://github.com/TPTBusiness/NexQuant/commit/86ce4f135d649cfb12f2f88626cd31868cb447e7))
* increase time default not controlled by LLM ([#1196](https://github.com/TPTBusiness/NexQuant/issues/1196)) ([e4bd647](https://github.com/TPTBusiness/NexQuant/commit/e4bd647d1b20cbaa26a00cf23c49bfbc0bc80477))
* Initialize EnvController in QuantTrace.__init__ ([698a17e](https://github.com/TPTBusiness/NexQuant/commit/698a17ea61321c37c7fa0d69849a309d29474f80))
* inject correct MultiIndex template into factor prompt ([49004db](https://github.com/TPTBusiness/NexQuant/commit/49004db027d699bacbb975f267daa95d1957ccd7))
* inject MultiIndex warning into factor interface prompt (YAML valide) ([79e2915](https://github.com/TPTBusiness/NexQuant/commit/79e2915823801d3574920fa197cf9c57965f485f))
* insert await asyncio.sleep(0) to yield control in loop ([#1186](https://github.com/TPTBusiness/NexQuant/issues/1186)) ([e0453e0](https://github.com/TPTBusiness/NexQuant/commit/e0453e0058e2a4ec74feb0b31883f45604a9bf0c))
* jinja problem of enumerate ([#1216](https://github.com/TPTBusiness/NexQuant/issues/1216)) ([6725f15](https://github.com/TPTBusiness/NexQuant/commit/6725f15f30df30a3ce37024fded621354d8114a7))
* kaggle competition metric direction ([#1195](https://github.com/TPTBusiness/NexQuant/issues/1195)) ([04878f9](https://github.com/TPTBusiness/NexQuant/commit/04878f9e703fee9caff9208ab23995586f165c95))
* **kronos:** lazy torch import to fix CI ModuleNotFoundError ([9cd8ab5](https://github.com/TPTBusiness/NexQuant/commit/9cd8ab54656786cc04742695c9d2e650a1b124ae))
* **kronos:** pass actual datetime Series to Kronos predictor timestamps ([7741408](https://github.com/TPTBusiness/NexQuant/commit/7741408c671b6fe943491b39d9fc5cac256b457e))
* **kronos:** replace rdagent_logger with stdlib logging for CI compatibility ([1ee5ea7](https://github.com/TPTBusiness/NexQuant/commit/1ee5ea7792f9ea94ddd26a0828d9744d0e07baa6))
* **loop:** compress old experiment history in proposal prompt to reduce context size ([bde37f0](https://github.com/TPTBusiness/NexQuant/commit/bde37f09d53a4f6582d071ed72d86491889bc573))
* **loop:** prevent step_idx advance on unhandled exceptions + fix consecutive assistant messages ([881ca81](https://github.com/TPTBusiness/NexQuant/commit/881ca819cea90d8a60865296e6f416aab69a18c9))
* merge candidates ([#1254](https://github.com/TPTBusiness/NexQuant/issues/1254)) ([46aad78](https://github.com/TPTBusiness/NexQuant/commit/46aad789ef710d9603e2330788dc66849cb6cab3))
* model/factor experiment filtering in Qlib proposals ([#1257](https://github.com/TPTBusiness/NexQuant/issues/1257)) ([9e34b4e](https://github.com/TPTBusiness/NexQuant/commit/9e34b4e855cbd709cd077f529950b8e1f5c01486))
* move snapshot saving after step index update in loop execution ([#1206](https://github.com/TPTBusiness/NexQuant/issues/1206)) ([774346d](https://github.com/TPTBusiness/NexQuant/commit/774346d92e3d9faa858f935bb2651d0f1aa12a6c))
* move task cancellation to finally block and fix subprocess kill typo ([#1234](https://github.com/TPTBusiness/NexQuant/issues/1234)) ([a984f69](https://github.com/TPTBusiness/NexQuant/commit/a984f69f681dda1c6c58f45e2505d7b0e8d75cf0))
* **optuna:** fix inverted parameter range in Stage 2/3 when signal_bias is negative ([f0be842](https://github.com/TPTBusiness/NexQuant/commit/f0be842a6c03f56cb209d1f8a0c5a0d9fa3baebf))
* Override webshop's Werkzeug dependency to fix CVE-2026-27199 ([3a5aa0b](https://github.com/TPTBusiness/NexQuant/commit/3a5aa0ba43fd644ad1944994f3cd3d49e7ab633c))
* preserve null end_time when rendering dataset segments template ([#1326](https://github.com/TPTBusiness/NexQuant/issues/1326)) ([6196ba3](https://github.com/TPTBusiness/NexQuant/commit/6196ba31f2e43db4761eeb482c3301e2238bc4cf))
* prevent calendar index overflow when signal data ends early ([#1324](https://github.com/TPTBusiness/NexQuant/issues/1324)) ([3dbd703](https://github.com/TPTBusiness/NexQuant/commit/3dbd7038280f21793246e5354f083ba472772a10))
* prevent JSON content from being added multiple times during retries ([#1255](https://github.com/TPTBusiness/NexQuant/issues/1255)) ([31b19de](https://github.com/TPTBusiness/NexQuant/commit/31b19dee80c5006c72a0a9698834a04a3acd4af9))
* Prevent path injection in FT Job Summary UI ([e4393fb](https://github.com/TPTBusiness/NexQuant/commit/e4393fb3b1e95fa53f7d8e972da35e994402def8))
* Prevent path injection in RL Job Summary UI ([b3e8cb8](https://github.com/TPTBusiness/NexQuant/commit/b3e8cb8cfe5fe74c5b893c6d0e401375630ee750))
* Prevent path traversal in autorl_bench server.py ([6634e6e](https://github.com/TPTBusiness/NexQuant/commit/6634e6e5c55c07f41d3a37731d59f6e11b35610e))
* Prevent path traversal in get_job_options() app.py ([7da2e57](https://github.com/TPTBusiness/NexQuant/commit/7da2e5706c7d7da8ffee3f04b42f8d3378af26ad))
* Prevent path traversal in RL UI app.py ([d2c1516](https://github.com/TPTBusiness/NexQuant/commit/d2c1516416dbda6109f6d42245263ce5373ce957))
* Prevent path traversal in Streamlit UI app.py ([0d0fd34](https://github.com/TPTBusiness/NexQuant/commit/0d0fd34573c0695c34431a6e9eb7b5c10a3a91f9))
* **qlib:** correct indentation in except blocks in quant_proposal and factor_runner ([8f67ab6](https://github.com/TPTBusiness/NexQuant/commit/8f67ab61299b7fb7063f5ac363705a6687ecaea1))
* Refactor path validation to fix CodeQL alert [#16](https://github.com/TPTBusiness/NexQuant/issues/16) ([a417ebc](https://github.com/TPTBusiness/NexQuant/commit/a417ebc41db5ad24b89f53e5f3c3ff6e5339ae18))
* refine DSCoSTEER_eval prompts ([#1157](https://github.com/TPTBusiness/NexQuant/issues/1157)) ([5594ab4](https://github.com/TPTBusiness/NexQuant/commit/5594ab418b46422e2f2e2edc08f0aadd0e95af04))
* refine prompts and add additional package info ([#1179](https://github.com/TPTBusiness/NexQuant/issues/1179)) ([5353bd3](https://github.com/TPTBusiness/NexQuant/commit/5353bd31f25a98cba552145709af743cd4e83cf5))
* refine task scheduling logic in MultiProcessEvolvingStrategy for… ([#1275](https://github.com/TPTBusiness/NexQuant/issues/1275)) ([27d38af](https://github.com/TPTBusiness/NexQuant/commit/27d38af7bd7e1fdb73e3617e94435abe7901dd21))
* remove $factor from prompt, update example count to EURUSD ([3adc5bf](https://github.com/TPTBusiness/NexQuant/commit/3adc5bf75e6820328991aa5a5456e6f68ccf8fd7))
* remove all Chinese stock references, replace with EURUSD 1min FX ([44eeb01](https://github.com/TPTBusiness/NexQuant/commit/44eeb01ec4f95271a084e9d285e00959926923f3))
* Remove API key from test_benchmark_api.py config ([16e8631](https://github.com/TPTBusiness/NexQuant/commit/16e86310bdd8d2af1539063957edebde97f88110))
* Remove API key logging from eurusd_llm.py ([3f510be](https://github.com/TPTBusiness/NexQuant/commit/3f510be9daddf0b241925f605898e2e1d3a18cb7))
* Remove API key parameter from generate_api_config() ([e6eeac9](https://github.com/TPTBusiness/NexQuant/commit/e6eeac93614a9d97d119696802c7a08153c70f59))
* Remove API key presence detection from logging ([12b45e5](https://github.com/TPTBusiness/NexQuant/commit/12b45e50f2d7d41881c3028b3f2213e7e7c573d8))
* Remove clear-text storage of API key (CodeQL alert [#8](https://github.com/TPTBusiness/NexQuant/issues/8)) ([4842311](https://github.com/TPTBusiness/NexQuant/commit/4842311d9193d665c27311e7efc9637b9f3e0519))
* Remove hardcoded credentials from test_benchmark_api.py ([2523ee2](https://github.com/TPTBusiness/NexQuant/commit/2523ee213e35c03175da9512619b46f6e9069f88))
* remove unused imports in data science scenario module ([#1136](https://github.com/TPTBusiness/NexQuant/issues/1136)) ([fd6cd39](https://github.com/TPTBusiness/NexQuant/commit/fd6cd3950c4d0463f2d1ccab63fa48be4de41a58))
* Rename loader.py to prompt_loader.py to fix module conflict ([06f0c34](https://github.com/TPTBusiness/NexQuant/commit/06f0c3427c665063513ae097068be71069a733b2))
* replace hardcoded ChromeDriver path with webdriver-manager ([#1271](https://github.com/TPTBusiness/NexQuant/issues/1271)) ([e3d2443](https://github.com/TPTBusiness/NexQuant/commit/e3d24437cf7842623fe27fd9221e36a07457d7f7))
* Resolve 88% empty backtest results + path fixes ([8d1c70e](https://github.com/TPTBusiness/NexQuant/commit/8d1c70e679721b90c024bc747d2544ce9c151adf))
* resolve dead code, shell injection risk, mutable defaults, and other bugs ([4267315](https://github.com/TPTBusiness/NexQuant/commit/4267315783ccbdaa3472c5f7fd4728cf656556c1))
* Resolve FORWARD_BARS NameError in backtest script ([ad7f5e1](https://github.com/TPTBusiness/NexQuant/commit/ad7f5e1388ad2149d0c32a5febfed0b77b05ef47))
* Resolve security vulnerabilities (Dependabot + Code Scanning) ([2c96828](https://github.com/TPTBusiness/NexQuant/commit/2c9682800e4ea30361561affbb747e4f2cc763f6))
* resolve unbound variable, logger shadowing, withdraw_loop edge case, and other bugs in main scripts ([2fd4bc3](https://github.com/TPTBusiness/NexQuant/commit/2fd4bc3741bafc6778008b3ecc49ba01207f22e1))
* revert 2 commits ([#1239](https://github.com/TPTBusiness/NexQuant/issues/1239)) ([2201a47](https://github.com/TPTBusiness/NexQuant/commit/2201a4762343f2cc2deb3dff2b70baf99f102292))
* revert to v10 setting ([#1220](https://github.com/TPTBusiness/NexQuant/issues/1220)) ([51f5bc9](https://github.com/TPTBusiness/NexQuant/commit/51f5bc9e117c6bfcb50c29355d5e73381d40b511))
* **security:** nosec for B608/B701 false positives in UI and template code ([8b73952](https://github.com/TPTBusiness/NexQuant/commit/8b739528e5679cb49989be7e0edd7ac404b5d993))
* **security:** Patch 5 CodeQL path injection and clear-text logging alerts ([#22](https://github.com/TPTBusiness/NexQuant/issues/22)-[#25](https://github.com/TPTBusiness/NexQuant/issues/25), [#9](https://github.com/TPTBusiness/NexQuant/issues/9)) ([5aed2cf](https://github.com/TPTBusiness/NexQuant/commit/5aed2cf58a4a39d515bc81e5fd6835a138198b82))
* **security:** Patch 5 CodeQL path injection and weak hashing alerts ([#25](https://github.com/TPTBusiness/NexQuant/issues/25)-[#30](https://github.com/TPTBusiness/NexQuant/issues/30)) ([e188333](https://github.com/TPTBusiness/NexQuant/commit/e1883331f18e7265aeb13145abaca4b295a15f6e))
* **security:** Patch path injection and stack trace exposure (CodeQL [#31](https://github.com/TPTBusiness/NexQuant/issues/31), [#27](https://github.com/TPTBusiness/NexQuant/issues/27)) ([2b0525f](https://github.com/TPTBusiness/NexQuant/commit/2b0525f9b7ef68ecc04bfddd558184f06640fb0b))
* **security:** real fix for B110 (logging in factor_proposal.py [#746](https://github.com/TPTBusiness/NexQuant/issues/746)) ([61656af](https://github.com/TPTBusiness/NexQuant/commit/61656afda75e77686952d847aec443c28e17b6d6))
* **security:** real fix for B110 (logging in factor_runner.py [#744](https://github.com/TPTBusiness/NexQuant/issues/744)) ([5ac64e6](https://github.com/TPTBusiness/NexQuant/commit/5ac64e60e4e3977364ffd5ad8704fdf0c46bad75))
* **security:** real fix for B110 (logging in quant_proposal.py [#741](https://github.com/TPTBusiness/NexQuant/issues/741)) ([bcfeb32](https://github.com/TPTBusiness/NexQuant/commit/bcfeb32958953ba07e980dce5feaffe5d53963e8))
* **security:** real fix for B110 (logging in quant_proposal.py [#741](https://github.com/TPTBusiness/NexQuant/issues/741)) ([d865c82](https://github.com/TPTBusiness/NexQuant/commit/d865c824c98820b26e3d64b8c193445effb19667))
* **security:** real fix for B404/B603 (sys.executable in factor_runner.py [#745](https://github.com/TPTBusiness/NexQuant/issues/745)) ([7894b8e](https://github.com/TPTBusiness/NexQuant/commit/7894b8e6ed1cb580d8909403eb166a2b418b2dd0))
* **security:** replace eval() with ast.literal_eval and add request timeouts (B307, B113) ([ffb24fd](https://github.com/TPTBusiness/NexQuant/commit/ffb24fd5de724455aa77846c3f98fae35bc80430))
* **security:** replace eval() with ast.literal_eval in finetune validator (B307) ([8d53b81](https://github.com/TPTBusiness/NexQuant/commit/8d53b81633965fd0ae2bf32081dacc91b121b77d))
* **security:** replace os.path.realpath with pathlib.resolve in safe_resolve_path to fix path-injection alerts ([0d7af52](https://github.com/TPTBusiness/NexQuant/commit/0d7af52a2d32f1dbcc366b9f395c43ad47ddabb2))
* **security:** replace relative_to() with realpath+startswith for CodeQL sanitization ([d7e2018](https://github.com/TPTBusiness/NexQuant/commit/d7e2018a7232c59a40d6e740111572a0da0cd384))
* **security:** replace remaining assert statements with proper error handling ([d4d5baf](https://github.com/TPTBusiness/NexQuant/commit/d4d5bafd1eb8330f75917170520408b48d38f8c2))
* **security:** replace shell=True subprocess calls with list args (B602) ([30887ac](https://github.com/TPTBusiness/NexQuant/commit/30887ac244f77a5edabc11dda7805b9bb789667f))
* **security:** replace shell=True subprocess calls with list args in env.py (B602) ([1a4f1cf](https://github.com/TPTBusiness/NexQuant/commit/1a4f1cf6044842939bc5e7ed853c437cab591a26))
* **security:** resolve all 30 Bandit security alerts (B301, B614, B104) ([00f400f](https://github.com/TPTBusiness/NexQuant/commit/00f400fe2efda375884234cd381401583a65f456))
* **security:** resolve CodeQL path-injection alerts in UI data loaders ([7caab95](https://github.com/TPTBusiness/NexQuant/commit/7caab9545bd929909f4c7cae02fbcc2cc3a9893a))
* **security:** resolve CodeQL path-injection and clear-text-logging alerts ([8701b8b](https://github.com/TPTBusiness/NexQuant/commit/8701b8bd75f82ceb326da4f105609f4228961666))
* **security:** Resolve GitHub Security Scan alerts ([5af7f19](https://github.com/TPTBusiness/NexQuant/commit/5af7f19bd1656078991752d298c0f3c953f7af2c))
* **security:** resolve path-injection and add nosec for safe temp paths (B108, py/path-injection) ([4133fff](https://github.com/TPTBusiness/NexQuant/commit/4133fffa7d97bd38beb4b99aa7f3ab3039d78103))
* **security:** resolve path-injection, B701, B101, B112 Bandit alerts ([e87d612](https://github.com/TPTBusiness/NexQuant/commit/e87d61257fa4bb401415b62ff88c7ad75085d89c))
* **security:** revert broken read_pickle encoding arg in kaggle template (B301) ([e16460c](https://github.com/TPTBusiness/NexQuant/commit/e16460c7bc5329c9752cd12b20fcee978b5f232b))
* **security:** Upgrade vllm and transformers to patch 4 CVEs ([85915b3](https://github.com/TPTBusiness/NexQuant/commit/85915b3a20e9ceae6dd854ef4c64a61590a36d84))
* **security:** validate SQL identifiers in _add_column_if_not_exists (B608) ([c40795b](https://github.com/TPTBusiness/NexQuant/commit/c40795bcb0dab5ceff9b56ec019b9be6f9d10203))
* **security:** whitelist-validate metric column in get_top_factors (B608) ([db51417](https://github.com/TPTBusiness/NexQuant/commit/db51417cd4337e3b8b76420c93b1bb1ed3271b13))
* set requires_documentation_search to None to disable feature in eval ([#1245](https://github.com/TPTBusiness/NexQuant/issues/1245)) ([ee8c119](https://github.com/TPTBusiness/NexQuant/commit/ee8c119f31b72de1002e5ad5d30c56d0f4b6c9b9))
* Skip already evaluated factors in nexquant_full_eval.py ([8375213](https://github.com/TPTBusiness/NexQuant/commit/8375213629551605b4c401aa1ce71ed8d9f1e4db))
* skip Kronos factor on GPUs &lt; 20GB to avoid CUDA OOM (shared with llama-server) ([08fea7a](https://github.com/TPTBusiness/NexQuant/commit/08fea7a2809941d2b5f3feb5ba998dba132053bb))
* skip res_ratio check if timer or res_time is None ([#1189](https://github.com/TPTBusiness/NexQuant/issues/1189)) ([dbe2142](https://github.com/TPTBusiness/NexQuant/commit/dbe214282e84f099512eeaf01925c7dee1b780a6))
* **strategies:** guard against None IC in acceptance check, disable slow wf_rolling ([843cd9a](https://github.com/TPTBusiness/NexQuant/commit/843cd9ae017b05365e1bb353b9945e2fbce332dd))
* **strategies:** handle None ic/sharpe/dd in rejected strategy log output ([0121c2c](https://github.com/TPTBusiness/NexQuant/commit/0121c2c1583b752622c69313e78ccbeedf6c8d1b))
* **strategy:** Fix template variables, APIBackend import, and JSON extraction ([f0e813e](https://github.com/TPTBusiness/NexQuant/commit/f0e813ee48ae65e0ee78c27a8b971139dac5b552))
* **strategy:** Re-evaluate Optuna-optimized strategies with full OHLCV backtest ([7da8bad](https://github.com/TPTBusiness/NexQuant/commit/7da8badbc1005bb1866631dc14daa815641b4271))
* summary page bug ([#1219](https://github.com/TPTBusiness/NexQuant/issues/1219)) ([beab473](https://github.com/TPTBusiness/NexQuant/commit/beab473b40714fbd802ebb3b61c0dd3d3ba7d91a))
* Switch to ThreadPoolExecutor for factor evaluation ([d0aa146](https://github.com/TPTBusiness/NexQuant/commit/d0aa1464ea1e3553e4b869c3429e5e394bcebda8))
* Translate remaining German comment in eurusd_macro.py ([02b46d1](https://github.com/TPTBusiness/NexQuant/commit/02b46d1ffc3bfe87033714f71a9d22714a071f09))
* ui bug ([#1192](https://github.com/TPTBusiness/NexQuant/issues/1192)) ([2f8261f](https://github.com/TPTBusiness/NexQuant/commit/2f8261f82bf25ad714eff22be2283c6e645b5314))
* update fallback criterion ([#1210](https://github.com/TPTBusiness/NexQuant/issues/1210)) ([dbbe374](https://github.com/TPTBusiness/NexQuant/commit/dbbe374ac8b0cefcde9145a76b4cd5c0b40b3f92))
* Update LICENSE badge link from main to master branch ([0dbace6](https://github.com/TPTBusiness/NexQuant/commit/0dbace6aa7aa1a7a250e45c96e71591edeed8f55))
* update requirements.txt's streamlit ([#1133](https://github.com/TPTBusiness/NexQuant/issues/1133)) ([600d159](https://github.com/TPTBusiness/NexQuant/commit/600d159e86521cc0498df9df3756921e676e3332))
* Update Werkzeug to 2.3.8 (latest secure 2.x version) ([d68a5ee](https://github.com/TPTBusiness/NexQuant/commit/d68a5ee47cba6f8d2ca0faba1ad89ba65f4fc94b))
* update WF test for new default (wf_rolling=True) ([c906e00](https://github.com/TPTBusiness/NexQuant/commit/c906e00ac9731673f6386f8b3ce38f5d8e817992))
* Use 96-bar forward returns in backtest (matching factor IC horizon) ([19c5b3d](https://github.com/TPTBusiness/NexQuant/commit/19c5b3d70633d5cc622328e57acd122120d47971))
* Use num_api_keys instead of len(api_keys) for round-robin ([c91976e](https://github.com/TPTBusiness/NexQuant/commit/c91976e7968f54a065b4a5ee11228133b48db3e9))
* weg, Timestamps mit Uhrzeit, kein SZ-Beispiel ([e9f6ac4](https://github.com/TPTBusiness/NexQuant/commit/e9f6ac48d97b1b57a0dde14562cd1b6f5d106edd))


### Performance Improvements

* **kronos:** batch GPU inference via predict_batch — 75x faster ([a93f940](https://github.com/TPTBusiness/NexQuant/commit/a93f940485eb92d747d5e6f966acb5c5e8d118c7))
* **kronos:** batch GPU inference via predict_batch — 75x faster ([471b1f9](https://github.com/TPTBusiness/NexQuant/commit/471b1f9a4b22cfd2f473d28285a6c7390fe3d10c))


### Documentation

* Add ATTRIBUTION.md with clear usage guidelines ([c5bf3e4](https://github.com/TPTBusiness/NexQuant/commit/c5bf3e4e2b99074e54645328a399f8f6da0387ea))
* Add CLI welcome screenshot to README ([4103ebe](https://github.com/TPTBusiness/NexQuant/commit/4103ebe1bfdc625af18711cf78ed19c808270227))
* Add comprehensive CHANGELOG.md for v1.0.0 release ([569b72b](https://github.com/TPTBusiness/NexQuant/commit/569b72b2c9a154bf991d03ac078bf020ef1eab16))
* Add comprehensive CLI help and update README with quick start ([8265462](https://github.com/TPTBusiness/NexQuant/commit/8265462cacb4e03c981ead1d6b6393a9070f729e))
* Add comprehensive data setup guide to README ([ca30ed2](https://github.com/TPTBusiness/NexQuant/commit/ca30ed270ab36517604a9eb0f1ace0fdd58a917c))
* Add comprehensive Git commit guidelines to QWEN.md ([d10d3a2](https://github.com/TPTBusiness/NexQuant/commit/d10d3a2c658bb77366baec13e922f0ed924b51d8))
* Add conda requirement to README + fix nexquant CLI ([90e185a](https://github.com/TPTBusiness/NexQuant/commit/90e185a4986ff9a4838bd94cb7b4034fea573f87))
* Add CRITICAL rule - NEVER commit closed-source/private assets ([a0ed4f7](https://github.com/TPTBusiness/NexQuant/commit/a0ed4f712ed4aa49eadaa5ced070c22f0146420a))
* Add CRITICAL rule - NEVER commit trading strategies or JSON files ([cb0cb4c](https://github.com/TPTBusiness/NexQuant/commit/cb0cb4c1122b9aab23f2e2f4feb5b4a99ed05008))
* add documentation for Data Science configurable options ([#1301](https://github.com/TPTBusiness/NexQuant/issues/1301)) ([d603d5a](https://github.com/TPTBusiness/NexQuant/commit/d603d5a5aa86e43cfc0ee3efedc5ab18919809f5))
* add execution environment configuration guide (Docker vs Conda) ([#1288](https://github.com/TPTBusiness/NexQuant/issues/1288)) ([27ed3d1](https://github.com/TPTBusiness/NexQuant/commit/27ed3d1a75b15a5589af84d4f597a8484006e71e))
* Add implementation summary ([649ed0c](https://github.com/TPTBusiness/NexQuant/commit/649ed0c3c0db823fb4fc984b9f6b6e7970d728ff))
* Add live trading system documentation to QWEN.md ([49b15d9](https://github.com/TPTBusiness/NexQuant/commit/49b15d917828a3c1263da1785da5663c67d41b40))
* Add Microsoft RD-Agent acknowledgment to README ([06c0b44](https://github.com/TPTBusiness/NexQuant/commit/06c0b44e4106a725a879932122d871041042ec2b))
* Add professional badges to README header ([91d44dd](https://github.com/TPTBusiness/NexQuant/commit/91d44ddabd4b4cf82cb1e6f53c8f4547f52a50cb))
* Add results/ directory README for storage documentation ([ba4e5d6](https://github.com/TPTBusiness/NexQuant/commit/ba4e5d6ece652e8c1c3b8a713a2e0ea2a0ab225c))
* Add v2.0.0 release changelog ([c5e34ff](https://github.com/TPTBusiness/NexQuant/commit/c5e34ff7aaa2d30a159b05f4e6ecc853b8a4f79e))
* Clean changelog of closed-source performance metrics ([7dc2ecd](https://github.com/TPTBusiness/NexQuant/commit/7dc2ecdc8dbf4ef0a2936ab1f1e0c0469ca95e9c))
* Create changelog/ directory with v1.0.0.md release notes ([ddefcd4](https://github.com/TPTBusiness/NexQuant/commit/ddefcd420a9d98fc6548e14cfc94caffd2068963))
* Final system completion - all 9 phases done ([ab541de](https://github.com/TPTBusiness/NexQuant/commit/ab541de9b3ca4cdf62f14f97d540460fc333fca9))
* fix duplicate sections, add hardware requirements and data setup guide ([cc85cd4](https://github.com/TPTBusiness/NexQuant/commit/cc85cd482ac7169fbe98468539899a2ce561e70d))
* improve README badges, fix llama-server flags, clean up structure ([7981a6a](https://github.com/TPTBusiness/NexQuant/commit/7981a6a4d1517950f4124a78642db3f15fde03ba))
* Remove 'Inspired by' comments and add comprehensive Acknowledgments ([d5dc48a](https://github.com/TPTBusiness/NexQuant/commit/d5dc48a6bdd519d0ce159d21ca9bbc46b7996313))
* Simplify README for git-clone-only installation ([a1e3bb9](https://github.com/TPTBusiness/NexQuant/commit/a1e3bb903c31cea3ea4c5e572bc639352e3215ae))
* Translate all code comments to English ([cff6c2a](https://github.com/TPTBusiness/NexQuant/commit/cff6c2a55e0b465a3f30ab802f02e3b4583025bc))
* Translate data_config.yaml to English ([b5221b7](https://github.com/TPTBusiness/NexQuant/commit/b5221b761f51bcf2b7b14c7bdfabfa2e9629a3b0))
* Translate server.py comments to English ([7fd7592](https://github.com/TPTBusiness/NexQuant/commit/7fd75922f89d6358c1ce48fd886ffbca10537531))
* Translate server.py docstring to English ([d5acaa0](https://github.com/TPTBusiness/NexQuant/commit/d5acaa0c036913776eef6bb01083cce2942dc16c))
* update configuration docs ([#1155](https://github.com/TPTBusiness/NexQuant/issues/1155)) ([56ed919](https://github.com/TPTBusiness/NexQuant/commit/56ed919b2e44f4398ac304a4f6cdf099dd382096))
* update license section from MIT to AGPL-3.0 ([ff441a4](https://github.com/TPTBusiness/NexQuant/commit/ff441a49fe0b45c31b1702b8bd22d5c8edd37abb))
* Update QWEN.md with complete 5-phase architecture and results ([66e1798](https://github.com/TPTBusiness/NexQuant/commit/66e17981fd9241d9ee6f50be05142ee201b761a8))
* Update QWEN.md with detailed Git history correction guide ([a972772](https://github.com/TPTBusiness/NexQuant/commit/a97277298d3d5f122905d7e02b58568224b86b40))
* Update QWEN.md with implementation guide ([23af142](https://github.com/TPTBusiness/NexQuant/commit/23af142af0b127600c61ba3623f3538abf1c881c))
* Update SECURITY.md and CONTRIBUTING.md ([e40f659](https://github.com/TPTBusiness/NexQuant/commit/e40f6594441e195041ccb58072483fe8704eac4c))
* Update TODO.md with v1.0.0 completed items and future roadmap ([2d3ca5b](https://github.com/TPTBusiness/NexQuant/commit/2d3ca5bec66e81b37ce7bf4086f24556f6cad134))


### Miscellaneous Chores

* release 0.8.0 ([8c15238](https://github.com/TPTBusiness/NexQuant/commit/8c1523802c3c0237eae27ebef3e155af2cddd05e))

## [1.4.2](https://github.com/TPTBusiness/NexQuant/compare/v1.4.1...v1.4.2) (2026-05-03)


### Bug Fixes

* add missing sys import and fix undefined acc_rate in factor eval ([c45f990](https://github.com/TPTBusiness/NexQuant/commit/c45f9908ee321400f0a19c57f1482e4cd1394a50))

## [1.4.1](https://github.com/TPTBusiness/NexQuant/compare/v1.4.0...v1.4.1) (2026-05-03)


### Bug Fixes

* 15 bug fixes across orchestrator, runner, backtest, and infrastructure ([163687d](https://github.com/TPTBusiness/NexQuant/commit/163687d7e1c278a085d7052a3f958a3edb501e77))
* also catch ValueError in mean_variance for dimension mismatch ([ed73b72](https://github.com/TPTBusiness/NexQuant/commit/ed73b7253f7dc6459ee30dd81a1ce1194e46e9af))
* close log file handle, fix FTMO equity double-count, remove bare except ([76219a5](https://github.com/TPTBusiness/NexQuant/commit/76219a53efddaafc2b8bd48a0f76c1d4325e6ea5))
* correct project root paths and subprocess handling in parallel runner and CLI ([9735e3a](https://github.com/TPTBusiness/NexQuant/commit/9735e3a4d8f01e7b16fb9b185a002396a915cea4))
* filter NaN in max(), remove redundant ternary, handle non-finite vbt results ([f89fbb3](https://github.com/TPTBusiness/NexQuant/commit/f89fbb3421faf6ccdc8e68a911fd9db2c166120f))
* fix type annotation, remove unused parameter, improve import_class errors ([8b6ab73](https://github.com/TPTBusiness/NexQuant/commit/8b6ab735c05629bf6b76ddc2fd8b15617600cad7))
* resolve dead code, shell injection risk, mutable defaults, and other bugs ([afff262](https://github.com/TPTBusiness/NexQuant/commit/afff26287f7c4df7ddfde4e816d280fe845e11eb))
* resolve unbound variable, logger shadowing, withdraw_loop edge case, and other bugs in main scripts ([748cf9b](https://github.com/TPTBusiness/NexQuant/commit/748cf9b214a3e8447f1289fc4cf1e92ad6cc2f1a))

## [1.4.0](https://github.com/TPTBusiness/NexQuant/compare/v1.3.11...v1.4.0) (2026-05-01)


### Features

* **optimizer:** add max_positions parameter to Optuna search space ([fdb4be3](https://github.com/TPTBusiness/NexQuant/commit/fdb4be3b3ebd93325e7821f4251148424184a40d))

## [1.3.11](https://github.com/TPTBusiness/NexQuant/compare/v1.3.10...v1.3.11) (2026-05-01)


### Bug Fixes

* **ci:** lazy import logger in nexquant.py and cli.py to avoid ImportError in test env ([60763e8](https://github.com/TPTBusiness/NexQuant/commit/60763e8eae34f41865ba8e5e65bdfde13b564b4b))

## [1.3.10](https://github.com/TPTBusiness/NexQuant/compare/v1.3.9...v1.3.10) (2026-05-01)


### Bug Fixes

* **security:** replace remaining assert statements with proper error handling ([928533d](https://github.com/TPTBusiness/NexQuant/commit/928533d9a81bd5062f07458fbf94d3c7fe347775))

## [1.3.9](https://github.com/TPTBusiness/NexQuant/compare/v1.3.8...v1.3.9) (2026-05-01)


### Bug Fixes

* **security:** resolve path-injection, B701, B101, B112 Bandit alerts ([20b89a0](https://github.com/TPTBusiness/NexQuant/commit/20b89a061843b39836e975f158404e8e2d4627cd))

## [1.3.8](https://github.com/TPTBusiness/NexQuant/compare/v1.3.7...v1.3.8) (2026-04-30)


### Bug Fixes

* **deps:** relax aiohttp constraint to &gt;=3.13.4 for litellm compatibility ([34ab192](https://github.com/TPTBusiness/NexQuant/commit/34ab1923a887089eb36e5cbad6cb8df16f0333ca))
* **qlib:** correct indentation in except blocks in quant_proposal and factor_runner ([8143451](https://github.com/TPTBusiness/NexQuant/commit/8143451e8c0ead01c4d86d19669268c7bfb15fac))
* **security:** replace eval() with ast.literal_eval in finetune validator (B307) ([0508caf](https://github.com/TPTBusiness/NexQuant/commit/0508caf9140d210b823fefefa28ee535ec85a0ae))
* **security:** replace shell=True subprocess calls with list args in env.py (B602) ([2012d5a](https://github.com/TPTBusiness/NexQuant/commit/2012d5ae4e77cc2f1ab9a48beaaac5a74695d083))
* **security:** resolve path-injection and add nosec for safe temp paths (B108, py/path-injection) ([6727480](https://github.com/TPTBusiness/NexQuant/commit/67274803bd1d14e5d1df9a063f46b2edb8501a2b))

## [1.3.7](https://github.com/TPTBusiness/NexQuant/compare/v1.3.6...v1.3.7) (2026-04-30)


### Bug Fixes

* **security:** nosec for B608/B701 false positives in UI and template code ([5eb5d7e](https://github.com/TPTBusiness/NexQuant/commit/5eb5d7e8fdbe90e0dced83fef4e09f5a33e96b2b))
* **security:** replace eval() with ast.literal_eval and add request timeouts (B307, B113) ([3301ada](https://github.com/TPTBusiness/NexQuant/commit/3301ada697ca7d3afa1a188d2a76a87ae98b4529))
* **security:** replace shell=True subprocess calls with list args (B602) ([13c08f4](https://github.com/TPTBusiness/NexQuant/commit/13c08f4ce6813eb7c314087921ec8c0f40074bd7))

## [1.3.6](https://github.com/TPTBusiness/NexQuant/compare/v1.3.5...v1.3.6) (2026-04-30)


### Bug Fixes

* **security:** real fix for B110 (logging in factor_proposal.py [#746](https://github.com/TPTBusiness/NexQuant/issues/746)) ([16624e0](https://github.com/TPTBusiness/NexQuant/commit/16624e0bd966ae4d24c4a3eb42bbc31c11da3136))
* **security:** real fix for B110 (logging in factor_runner.py [#744](https://github.com/TPTBusiness/NexQuant/issues/744)) ([88cf0fb](https://github.com/TPTBusiness/NexQuant/commit/88cf0fb8828b11c97f2f3ae2881a4900b020c6f0))
* **security:** real fix for B110 (logging in quant_proposal.py [#741](https://github.com/TPTBusiness/NexQuant/issues/741)) ([7cf2a64](https://github.com/TPTBusiness/NexQuant/commit/7cf2a644f553b054bd4b0607ea51e5372e68d90a))
* **security:** real fix for B110 (logging in quant_proposal.py [#741](https://github.com/TPTBusiness/NexQuant/issues/741)) ([ef985f8](https://github.com/TPTBusiness/NexQuant/commit/ef985f86035d8dca707c60137e6508349a0c4ae6))
* **security:** real fix for B404/B603 (sys.executable in factor_runner.py [#745](https://github.com/TPTBusiness/NexQuant/issues/745)) ([819655a](https://github.com/TPTBusiness/NexQuant/commit/819655aaa3efa76596d60501d0e8ca365df3e5e2))
* **security:** revert broken read_pickle encoding arg in kaggle template (B301) ([3574907](https://github.com/TPTBusiness/NexQuant/commit/35749073c91e69f63ddaad61dae3f2b799327e63))
* **security:** validate SQL identifiers in _add_column_if_not_exists (B608) ([e10dfa2](https://github.com/TPTBusiness/NexQuant/commit/e10dfa2576038e911f83595d3b466c261bc0cd54))
* **security:** whitelist-validate metric column in get_top_factors (B608) ([e50519f](https://github.com/TPTBusiness/NexQuant/commit/e50519fe066e68aec2f19b83df4f643c3c22053d))

## [1.3.5](https://github.com/TPTBusiness/NexQuant/compare/v1.3.4...v1.3.5) (2026-04-27)


### Bug Fixes

* **auto-fixer:** add five new factor code fixes for groupby/apply errors ([449c8fd](https://github.com/TPTBusiness/NexQuant/commit/449c8fd70a327e604dcca122e4a134f0cca918e4))
* **auto-fixer:** add four new factor code fixes for common runtime errors ([40484f6](https://github.com/TPTBusiness/NexQuant/commit/40484f6d300425da481f1edd325da4acbc06ec7d))
* **auto-fixer:** add groupby([level=N,'date']) SyntaxError fix ([ca77c00](https://github.com/TPTBusiness/NexQuant/commit/ca77c005bea4abdd8854c1de2b0e8d03b7742161))
* **auto-fixer:** disable _fix_min_periods for intraday data ([77b0740](https://github.com/TPTBusiness/NexQuant/commit/77b0740f059349df7e769a378af728aa33b2070e))
* **auto-fixer:** fix chained groupby(level=N).groupby('date') pattern ([7d5fe32](https://github.com/TPTBusiness/NexQuant/commit/7d5fe32b31a19ce8b04bd8f5a430720fdb748f7a))
* **auto-fixer:** fix df.loc[instrument] DateParseError on MultiIndex frames ([b7860ea](https://github.com/TPTBusiness/NexQuant/commit/b7860eafc0ad26384947ce0510ecf4e9f3425807))
* **auto-fixer:** fix df['instrument'] KeyError on MultiIndex frames ([aad6bd1](https://github.com/TPTBusiness/NexQuant/commit/aad6bd1c7c720b3d486e0cf248337f32394773b1))
* **auto-fixer:** fix two assignment-target bugs in instrument column fixers ([421eedf](https://github.com/TPTBusiness/NexQuant/commit/421eedffed4b883c24397dc5581c019a3985277f))
* **auto-fixer:** preserve date dimension in groupby(['instrument','date']) fix ([b58fdd8](https://github.com/TPTBusiness/NexQuant/commit/b58fdd8be43720b5d4363e0f8de9a01591d4d2dc))
* **auto-fixer:** remove ddof from rolling() args, not only from std()/var() ([b0fc328](https://github.com/TPTBusiness/NexQuant/commit/b0fc328d0d4a041c65d8eeb32cb3f2bb86568406))
* **auto-fixer:** strip spurious .reset_index() after .transform() calls ([8708aae](https://github.com/TPTBusiness/NexQuant/commit/8708aae6e08728cda1875c775a76dc92e43576f3))
* **loop:** prevent step_idx advance on unhandled exceptions + fix consecutive assistant messages ([5ec4ad1](https://github.com/TPTBusiness/NexQuant/commit/5ec4ad1b96b5b99ef42bea7bb828cb1ef709a688))

## [1.3.4](https://github.com/TPTBusiness/NexQuant/compare/v1.3.3...v1.3.4) (2026-04-27)


### Bug Fixes

* **auto-fixer:** add five new factor code fixes for groupby/apply errors ([449c8fd](https://github.com/TPTBusiness/NexQuant/commit/449c8fd70a327e604dcca122e4a134f0cca918e4))
* **auto-fixer:** add four new factor code fixes for common runtime errors ([40484f6](https://github.com/TPTBusiness/NexQuant/commit/40484f6d300425da481f1edd325da4acbc06ec7d))
* **auto-fixer:** add groupby([level=N,'date']) SyntaxError fix ([ca77c00](https://github.com/TPTBusiness/NexQuant/commit/ca77c005bea4abdd8854c1de2b0e8d03b7742161))
* **auto-fixer:** disable _fix_min_periods for intraday data ([77b0740](https://github.com/TPTBusiness/NexQuant/commit/77b0740f059349df7e769a378af728aa33b2070e))
* **auto-fixer:** fix chained groupby(level=N).groupby('date') pattern ([7d5fe32](https://github.com/TPTBusiness/NexQuant/commit/7d5fe32b31a19ce8b04bd8f5a430720fdb748f7a))
* **auto-fixer:** fix df.loc[instrument] DateParseError on MultiIndex frames ([b7860ea](https://github.com/TPTBusiness/NexQuant/commit/b7860eafc0ad26384947ce0510ecf4e9f3425807))
* **auto-fixer:** fix df['instrument'] KeyError on MultiIndex frames ([aad6bd1](https://github.com/TPTBusiness/NexQuant/commit/aad6bd1c7c720b3d486e0cf248337f32394773b1))
* **auto-fixer:** preserve date dimension in groupby(['instrument','date']) fix ([b58fdd8](https://github.com/TPTBusiness/NexQuant/commit/b58fdd8be43720b5d4363e0f8de9a01591d4d2dc))
* **auto-fixer:** remove ddof from rolling() args, not only from std()/var() ([b0fc328](https://github.com/TPTBusiness/NexQuant/commit/b0fc328d0d4a041c65d8eeb32cb3f2bb86568406))
* **backtest:** replace broken MC permutation test with binomial win-rate test ([c38d894](https://github.com/TPTBusiness/NexQuant/commit/c38d89478f586825bfca5715a96ca70ccd8791a3))
* **factors:** detect and correct look-ahead bias in daily-constant factors ([eb490a4](https://github.com/TPTBusiness/NexQuant/commit/eb490a461b66cbd815ae53ac5205115754712432))
* **factors:** extend look-ahead rules to session factors and add intraday-factor guidance ([c24c100](https://github.com/TPTBusiness/NexQuant/commit/c24c100442d6487686c0578de0b32d240fcbf215))
* **loop:** compress old experiment history in proposal prompt to reduce context size ([4bf90a9](https://github.com/TPTBusiness/NexQuant/commit/4bf90a905ba8b2aba2a818191c19998088cccaaf))
* **loop:** prevent step_idx advance on unhandled exceptions + fix consecutive assistant messages ([5ec4ad1](https://github.com/TPTBusiness/NexQuant/commit/5ec4ad1b96b5b99ef42bea7bb828cb1ef709a688))

## [1.3.3](https://github.com/TPTBusiness/NexQuant/compare/v1.3.2...v1.3.3) (2026-04-25)


### Bug Fixes

* **backtest:** replace broken MC permutation test with binomial win-rate test ([c38d894](https://github.com/TPTBusiness/NexQuant/commit/c38d89478f586825bfca5715a96ca70ccd8791a3))
* **factors:** detect and correct look-ahead bias in daily-constant factors ([eb490a4](https://github.com/TPTBusiness/NexQuant/commit/eb490a461b66cbd815ae53ac5205115754712432))
* **factors:** extend look-ahead rules to session factors and add intraday-factor guidance ([c24c100](https://github.com/TPTBusiness/NexQuant/commit/c24c100442d6487686c0578de0b32d240fcbf215))
* **loop:** compress old experiment history in proposal prompt to reduce context size ([4bf90a9](https://github.com/TPTBusiness/NexQuant/commit/4bf90a905ba8b2aba2a818191c19998088cccaaf))
* **strategies:** guard against None IC in acceptance check, disable slow wf_rolling ([2197f52](https://github.com/TPTBusiness/NexQuant/commit/2197f52150a50ef38d9e70991d7e48c8c30caec4))
* **strategies:** handle None ic/sharpe/dd in rejected strategy log output ([ad2ad3a](https://github.com/TPTBusiness/NexQuant/commit/ad2ad3ab3360ea75ed3bbc90c12098b9c5cc0114))

## [1.3.2](https://github.com/TPTBusiness/NexQuant/compare/v1.3.1...v1.3.2) (2026-04-23)


### Bug Fixes

* **strategies:** guard against None IC in acceptance check, disable slow wf_rolling ([2197f52](https://github.com/TPTBusiness/NexQuant/commit/2197f52150a50ef38d9e70991d7e48c8c30caec4))
* **strategies:** handle None ic/sharpe/dd in rejected strategy log output ([ad2ad3a](https://github.com/TPTBusiness/NexQuant/commit/ad2ad3ab3360ea75ed3bbc90c12098b9c5cc0114))

## [1.3.1](https://github.com/TPTBusiness/NexQuant/compare/v1.3.0...v1.3.1) (2026-04-21)


### Bug Fixes

* **deps:** bump python-dotenv to &gt;=1.2.2 (CVE symlink overwrite) ([126ae7d](https://github.com/TPTBusiness/NexQuant/commit/126ae7d5fb556b677d09d10221862a0d648d697a))

## [1.3.0](https://github.com/TPTBusiness/NexQuant/compare/v1.2.2...v1.3.0) (2026-04-21)


### Features

* **backtest:** add rolling walk-forward validation and Monte Carlo trade permutation test ([637a94c](https://github.com/TPTBusiness/NexQuant/commit/637a94c1d987da763869f4f9b73372a3f37d873c))


### Bug Fixes

* **security:** resolve all 30 Bandit security alerts (B301, B614, B104) ([ce5983d](https://github.com/TPTBusiness/NexQuant/commit/ce5983d9d59c4c34341fb1ec749e44bbcfc4a1c4))

## [1.2.2](https://github.com/TPTBusiness/NexQuant/compare/v1.2.1...v1.2.2) (2026-04-19)


### Documentation

* **claude:** auto-merge release-please PR after every push ([f500917](https://github.com/TPTBusiness/NexQuant/commit/f500917b699ee78dc676e84e01574d49bdc8e796))

## [2.2.0](https://github.com/TPTBusiness/NexQuant/compare/v2.1.0...v2.2.0) (2026-04-18)


### Features

* add Kronos CLI commands, expand tests, document in README ([f911081](https://github.com/TPTBusiness/NexQuant/commit/f911081d1763d0dc4dd790b57dd97aae2dc62679))
* **fin_quant:** auto-generate Kronos factor before loop start ([277063f](https://github.com/TPTBusiness/NexQuant/commit/277063f3e36cd071db859cdc77f69135c1f0763b))
* integrate Kronos-mini OHLCV foundation model (Option A + B) ([4ae3b99](https://github.com/TPTBusiness/NexQuant/commit/4ae3b99f2450930f72e202a1a470c407bfde3328))


### Bug Fixes

* **kronos:** lazy torch import to fix CI ModuleNotFoundError ([ccc1d27](https://github.com/TPTBusiness/NexQuant/commit/ccc1d27dbe5ab06a57085a589d456ac7bf49cc08))
* **kronos:** pass actual datetime Series to Kronos predictor timestamps ([dc6e7ce](https://github.com/TPTBusiness/NexQuant/commit/dc6e7ce207d21fbc21976f2af7691058530fac2f))
* **kronos:** replace rdagent_logger with stdlib logging for CI compatibility ([b4558f2](https://github.com/TPTBusiness/NexQuant/commit/b4558f2456659c6109bd1b3cf100510491cd3e6c))


### Performance Improvements

* **kronos:** batch GPU inference via predict_batch — 75x faster ([74611d0](https://github.com/TPTBusiness/NexQuant/commit/74611d071ac123a655eb15d0737bb73b8c1bd2b0))
* **kronos:** batch GPU inference via predict_batch — 75x faster ([2babeb9](https://github.com/TPTBusiness/NexQuant/commit/2babeb95f42828e13a37dc16166c75538f33fd4b))


### Documentation

* fix duplicate sections, add hardware requirements and data setup guide ([6c771b3](https://github.com/TPTBusiness/NexQuant/commit/6c771b37e6f88526a896499e86929cfca2c199eb))

## [2.1.0](https://github.com/TPTBusiness/NexQuant/compare/v2.0.0...v2.1.0) (2026-04-18)


### Features

* add daily log rotation, llama health wait, factor auto-fixer, and README updates ([4ae4d6f](https://github.com/TPTBusiness/NexQuant/commit/4ae4d6f0f1388d229e44333130306ae05767f2e5))
* Add GitHub infrastructure, CI/CD pipelines, and examples ([a0b5dc4](https://github.com/TPTBusiness/NexQuant/commit/a0b5dc464eaac831c76bdbf805cf60c9083e7d80))
* **factor-coder:** Add critical rules to prevent common factor implementation errors ([a1edca8](https://github.com/TPTBusiness/NexQuant/commit/a1edca87dd5e75ee402ea555f1b7a07b45c4b1f0))
* **logging:** write complete LLM prompts and responses to daily JSONL log ([803ef13](https://github.com/TPTBusiness/NexQuant/commit/803ef13052c645392e71aa5de24874aae83f62a7))
* **strategy:** Continuous optimization with Optuna parameter injection ([4fda5ea](https://github.com/TPTBusiness/NexQuant/commit/4fda5eaa31bc570e295ad96380ee2c02b82db706))
* unified backtest engine, LLM error handling, strategy refactor ([76b9341](https://github.com/TPTBusiness/NexQuant/commit/76b9341fe8ef0ff03fd911337c299cf0e8582f37))


### Bug Fixes

* Add critical column name rules to factor generation prompt ([3e74410](https://github.com/TPTBusiness/NexQuant/commit/3e7441079f0f1c5867829a365c6e45cd7d2071df))
* **ci:** fix closed-source asset check false positives in security workflow ([4b83c2b](https://github.com/TPTBusiness/NexQuant/commit/4b83c2bfe7e90c0c7a11116f07a1b989035b7a3f))
* **ci:** remove CodeQL workflow (conflicts with default setup), drop duplicate lint job ([a671361](https://github.com/TPTBusiness/NexQuant/commit/a671361ee4de9a7e00ccc66d8fd5732c2ed1fee9))
* **ci:** set JAVA_TOOL_OPTIONS UTF-8 in Codacy workflow ([e36721c](https://github.com/TPTBusiness/NexQuant/commit/e36721c765a02a325b8a7dfd3c262b2aca7b1652))
* **deps:** pin aiohttp&gt;=3.13.4 to patch 4 CVEs ([81adddc](https://github.com/TPTBusiness/NexQuant/commit/81adddcfcd14819a1f85c06288a663e7d222a8fb))
* **optuna:** fix inverted parameter range in Stage 2/3 when signal_bias is negative ([eaf885e](https://github.com/TPTBusiness/NexQuant/commit/eaf885ec2d20ebd93e34d1e2cb445532d2fb0ed3))
* **security:** Patch 5 CodeQL path injection and clear-text logging alerts ([#22](https://github.com/TPTBusiness/NexQuant/issues/22)-[#25](https://github.com/TPTBusiness/NexQuant/issues/25), [#9](https://github.com/TPTBusiness/NexQuant/issues/9)) ([d386af9](https://github.com/TPTBusiness/NexQuant/commit/d386af98205722d1ea6d1465f585e89cb8df47de))
* **security:** Patch 5 CodeQL path injection and weak hashing alerts ([#25](https://github.com/TPTBusiness/NexQuant/issues/25)-[#30](https://github.com/TPTBusiness/NexQuant/issues/30)) ([0d4c3b7](https://github.com/TPTBusiness/NexQuant/commit/0d4c3b7d69fdbdaafab00940bf7346c8b664928e))
* **security:** Patch path injection and stack trace exposure (CodeQL [#31](https://github.com/TPTBusiness/NexQuant/issues/31), [#27](https://github.com/TPTBusiness/NexQuant/issues/27)) ([b0b8432](https://github.com/TPTBusiness/NexQuant/commit/b0b84328d13dac5c2ef79961200b011c0b5778f1))
* **security:** replace relative_to() with realpath+startswith for CodeQL sanitization ([6d70f1e](https://github.com/TPTBusiness/NexQuant/commit/6d70f1ed944180c44d0eb75c0e86b013e5888b60))
* **security:** resolve CodeQL path-injection alerts in UI data loaders ([cced426](https://github.com/TPTBusiness/NexQuant/commit/cced426916cb726e95ad251dcbc0eb9ab6ec3591))
* **security:** resolve CodeQL path-injection and clear-text-logging alerts ([ec50224](https://github.com/TPTBusiness/NexQuant/commit/ec50224c3580c5c82ddba02fe77af95efd9667ea))
* **security:** Resolve GitHub Security Scan alerts ([6c85ba8](https://github.com/TPTBusiness/NexQuant/commit/6c85ba833a48326e39006e0f73c506b29a594bde))
* **security:** Upgrade vllm and transformers to patch 4 CVEs ([6c9ba91](https://github.com/TPTBusiness/NexQuant/commit/6c9ba91d3bf7ce1ed389e544c68be55262bf4e28))
* **strategy:** Fix template variables, APIBackend import, and JSON extraction ([8220faa](https://github.com/TPTBusiness/NexQuant/commit/8220faa3de6ea555717ac29ba90a3b68135fbf9e))
* **strategy:** Re-evaluate Optuna-optimized strategies with full OHLCV backtest ([026edce](https://github.com/TPTBusiness/NexQuant/commit/026edce122284fb1da467e6e9de8a2b9116c7ace))


### Documentation

* Add CLI welcome screenshot to README ([e6f2374](https://github.com/TPTBusiness/NexQuant/commit/e6f237437595745406c310b58a9bd7214ff914ae))
* Add comprehensive data setup guide to README ([f721d53](https://github.com/TPTBusiness/NexQuant/commit/f721d53e5681be6997418c13acc3439897168048))
* Add conda requirement to README + fix nexquant CLI ([df45698](https://github.com/TPTBusiness/NexQuant/commit/df45698b20e0a3e6e0079decf2b8eecb6983a175))
* Clean changelog of closed-source performance metrics ([a0f6587](https://github.com/TPTBusiness/NexQuant/commit/a0f6587ab1724293924da07fe18c40891ca612a1))
* improve README badges, fix llama-server flags, clean up structure ([336e1a5](https://github.com/TPTBusiness/NexQuant/commit/336e1a5afb4933ec13572ef050a3e5a2ca183400))
