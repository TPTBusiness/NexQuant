# Security Runbook für Predix

## Bandit Security Scanner

### Konfiguration

Bandit ist als Pre-Commit Hook konfiguriert und scannt automatisch alle Python-Dateien vor jedem Commit.

**Konfigurationsdateien:**
- `.bandit.yml` - Bandit-Einstellungen
- `.pre-commit-config.yaml` - Pre-commit Hooks
- `requirements/lint.txt` - Bandit Dependency

### Scan-Befehle

```bash
# Alle Dateien scannen
bandit -r rdagent/ -c .bandit.yml

# Nur HIGH Severity Issues
bandit -r rdagent/ -c .bandit.yml --severity-level high

# Spezifische Datei scannen
bandit rdagent/components/backtesting/results_db.py -c .bandit.yml

# Mit JSON Output (für CI/CD)
bandit -r rdagent/ -c .bandit.yml -f json -o results/security/bandit-report.json
```

### Gefundene HIGH Severity Issues

#### 1. subprocess mit shell=True (12 Issues)

**Dateien:**
- `rdagent/utils/env.py` (mehrere Stellen)
- `rdagent/components/coder/factor_coder/factor.py`

**Bewertung:** ✅ **Akzeptiert** - Internal Tool
- Alle Commands verwenden hardcodierte Strings, keine User-Inputs
- Risk: Command Injection bei manipulierten Inputs
- Mitigation: Code-Review für alle subprocess-Aufrufe, keine externen Inputs

**Empfohlene Fixes (Future PR):**
```python
# Statt:
subprocess.run(f"conda env list | grep -q '^{env_name} '", shell=True)

# Besser:
subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=True)
# Dann in Python auf env_name prüfen
```

**Priority:** MEDIUM - Refactor in nächster Wartungsphase

---

#### 2. Jinja2 autoescape=False (6 Issues)

**Dateien:**
- `rdagent/components/coder/data_science/ensemble/__init__.py`
- `rdagent/components/coder/data_science/ensemble/eval.py`
- `rdagent/scenarios/kaggle/developer/coder.py` (2x)
- `rdagent/scenarios/qlib/experiment/utils.py`
- `rdagent/utils/agent/tpl.py`

**Bewertung:** ✅ **Akzeptiert** - Template Generation für Code
- Templates generieren Python-Code, nicht HTML
- XSS-Risiko besteht nicht bei Code-Templates
- `StrictUndefined` verhindert undefined variable leaks

**Mitigation:** ✅ Already secure durch `StrictUndefined`

---

#### 3. MD5 Hash (2 Issues)

**Dateien:**
- `rdagent/log/ui/ds_trace.py` (2x)

**Bewertung:** ✅ **Akzeptiert** - Non-Crypto Use Case
- MD5 wird für UI-Caching verwendet, nicht für Security
- `usedforsecurity=False` kann hinzugefügt werden

**Empfohlener Fix (Quick Win):**
```python
# Zeile 226 & 333 in rdagent/log/ui/ds_trace.py
unique_key = hashlib.md5("...".encode(), usedforsecurity=False).hexdigest()
```

**Priority:** LOW - 5 Minuten Fix

---

#### 4. tarfile.extractall ohne Validation (2 Issues)

**Dateien:**
- `rdagent/scenarios/data_science/proposal/exp_gen/select/submit.py`
- `rdagent/scenarios/kaggle/kaggle_crawler.py`

**Bewertung:** ⚠️ **Sollte gefixt werden** - Path Traversal Risk
- Extrahiert externe Archive (Kaggle Datasets)
- Risk: Path Traversal Attacks via `../../../etc/passwd`

**Empfohlener Fix:**
```python
import tarfile
import os

def safe_extractall(tar: tarfile.TarFile, path: str) -> None:
    """Extract tarfile safely, preventing path traversal."""
    def is_within_directory(directory: str, target: str) -> bool:
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        prefix = os.path.commonprefix([abs_directory, abs_target])
        return prefix == abs_directory

    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise ValueError(f"Attempted Path Traversal: {member.name}")
    tar.extractall(path=path)

# Usage:
with tarfile.open(tar_path, mode="r:*") as tar:
    safe_extractall(tar, to_dir)
```

**Priority:** HIGH - Nächster Sprint

---

#### 5. Flask debug=True (1 Issue)

**Datei:**
- `rdagent/log/server/debug_app.py:170`

**Bewertung:** ⚠️ **Sollte gefixt werden** - Debugger Exposure
- `debug=True` ermöglicht arbitrary code execution
- Sollte nur in Development-Umgebung sein

**Empfohlener Fix:**
```python
import os

# Zeile 170
debug_mode = os.getenv("FLASK_ENV") == "development"
app.run(debug=debug_mode, host="0.0.0.0", port=port)
```

**Priority:** HIGH - Quick Fix

---

### Skipped Rules Begründung

| Rule | Begründung | Status |
|------|-----------|--------|
| B101 (assert) | Development/Debug Assertions | ✅ Akzeptiert |
| B311 (random) | Non-Crypto Random Usage | ✅ Akzeptiert |
| B404, B603, B607 (subprocess) | Legitimate System Operations | ⚠️ Monitor |
| B113 (request timeout) | Wird in future PR gefixt | 📋 Planned |
| B608 (SQL injection) | Internal Tool, keine User-Inputs | ⚠️ Monitor |
| B301 (pickle) | Controlled Data Sources | ⚠️ Monitor |
| B701 (jinja2) | Code Templates, nicht HTML | ✅ Secure |
| B201 (flask debug) | Development Only | 📋 Fix Planned |
| B324 (hashlib) | Non-Crypto (Caching) | 📋 Quick Fix |
| B202 (tarfile) | External Archives | 🔴 Fix Required |

---

### Pre-Commit Verhalten

**Blockiert Commit bei:**
- HIGH Severity Issues (standardmäßig aktiv)

**Erlaubt Commit bei:**
- MEDIUM Severity Issues (Informational)
- LOW Severity Issues (Informational)

**Manuelles Überspringen (NOT recommended):**
```bash
# Nur im Notfall!
git commit --no-verify -m "feat: urgent fix"
```

---

### CI/CD Integration

Für GitHub Actions:

```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [master, main]
  pull_request:
    branches: [master, main]

jobs:
  bandit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install bandit

      - name: Run Bandit
        run: |
          bandit -r rdagent/ \
            -c .bandit.yml \
            -f json \
            -o bandit-report.json \
            --exit-zero

      - name: Upload Security Report
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: bandit-report.json
```

---

### Regelmäßige Wartung

**Monatlich:**
```bash
# Bandit-Report generieren
bandit -r rdagent/ -c .bandit.yml -f html -o results/security/bandit-report-$(date +%Y-%m).html

# Trend-Analyse
bandit -r rdagent/ -c .bandit.yml -lll | grep "Total issues"
```

**Quartalsweise:**
- Alle `# nosec` Comments reviewen
- Skipped Rules reevaluieren
- Neue Security-Best-Practices einarbeiten

---

### Kontakt & Eskalation

- **Security Issues melden:** @TPTBusiness
- **False Positives:** Zu `.bandit.yml` hinzufügen mit Begründung
- **Patches:** PR mit Label `security` erstellen

---

### Referenzen

- [Bandit Documentation](https://bandit.readthedocs.io/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Database](https://cwe.mitre.org/)
- [Pre-Commit Hooks](https://pre-commit.com/)
