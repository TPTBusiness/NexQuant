# Pull Request

## Beschreibung

<!-- 
Eine klare und prägnante Beschreibung der Änderungen.
Beziehe dich auf das zugehörige Issue (falls vorhanden).
-->

**Fixes:** #<!-- Issue-Nummer -->

## Typ

<!-- Bitte zutreffendes ankreuzen [x] -->

- [ ] 🐛 Bug Fix
- [ ] ✨ Neue Funktion
- [ ] 📚 Dokumentation
- [ ] 🧹 Code Cleanup/Refactoring
- [ ] ⚡ Performance-Verbesserung
- [ ] 🔧 Konfiguration/Build
- [ ] 🧪 Tests

## Changes

<!-- Welche Dateien wurden geändert und warum? -->

- `Datei1.py`: Beschreibung der Änderung
- `Datei2.py`: Beschreibung der Änderung

## Testing

<!-- Wie wurden die Änderungen getestet? -->

### Tests hinzugefügt/aktualisiert

- [ ] Ja, Unit Tests
- [ ] Ja, Integration Tests
- [ ] Nein, aber manuell getestet
- [ ] Nicht zutreffend

### Testing Notes

<!-- Beschreibe deine Testing-Schritte -->

```bash
# Beispiel: Tests ausführen
pytest test/ -v --cov=rdagent

# Beispiel: CLI Command testen
rdagent COMMAND --help
```

## Checklist

<!-- Bitte alle zutreffenden Punkte ankreuzen [x] -->

- [ ] Meine Änderungen folgen dem [Coding Style](CONTRIBUTING.md)
- [ ] Ich habe [CONTRIBUTING.md](CONTRIBUTING.md) gelesen und befolgt
- [ ] Tests wurden hinzugefügt oder aktualisiert
- [ ] Dokumentation wurde aktualisiert (`docs/` oder README.md)
- [ ] CHANGELOG.md wurde aktualisiert (falls zutreffend)
- [ ] Pre-commit Hooks bestanden (`pre-commit run --all-files`)
- [ ] Keine closed-source Assets committen (siehe unten)

## ⚠️ Closed-Source Check

<!-- 
KRITISCH: Bitte bestätige, dass KEINE der folgenden Dateien committen wurden:
-->

- [ ] `git_ignore_folder/` – Trading-Skripte, OHLCV-Daten, Credentials
- [ ] `results/` – Backtest-Ergebnisse, Strategien, Logs
- [ ] `.env` – API-Keys, Credentials
- [ ] `models/local/` – Eigene verbesserte Modelle
- [ ] `prompts/local/` – Eigene verbesserte Prompts
- [ ] `rdagent/scenarios/qlib/local/` – Closed-Source Komponenten
- [ ] `*.db` – SQLite-Datenbanken
- [ ] `*.log` – Log-Files

## Screenshots (falls relevant)

<!-- Vorher/Nachher-Vergleiche, UI-Änderungen etc. -->

| Vorher | Nachher |
|--------|---------|
| <!-- Screenshot --> | <!-- Screenshot --> |

## Zusätzliche Kontext

<!-- Weitere Informationen zu den Änderungen -->
