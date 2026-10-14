"""Bedienoberfläche für den Windows-Prototyp (läuft lokal im Browser, offline)."""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

from .fit import Person
from .measure import MEASURES
from .quality import CHECKLIST
from .store import Store, compare

GENDERS = {"Männlich": "maennlich", "Weiblich": "weiblich", "Divers": "divers"}
OPTIONAL = ["chest", "waist", "hips", "thigh", "calf", "upperarm", "neck", "inseam"]


def _person(gender: str, age: float, height: float, weight: float, optional: list[float | None]) -> Person:
    measures = {"height": float(height or 0), "weight": float(weight or 0)}
    for key, value in zip(OPTIONAL, optional, strict=True):
        if value:
            measures[key] = float(value)
    return Person(GENDERS.get(gender, gender), float(age or 30), measures)


def _stream(task: Callable[[Callable[[str], None]], object]) -> Iterator[tuple[str, object | None]]:
    """Führt task(log) im Hintergrund aus und liefert laufend das Protokoll."""
    q: queue.Queue = queue.Queue()
    lines: list[str] = []
    box: dict = {}

    def worker():
        try:
            box["result"] = task(lambda s: q.put(s))
        except Exception as exc:  # noqa: BLE001 - Fehler soll in der Oberfläche erscheinen
            box["error"] = exc
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            break
        lines.append(item)
        yield "\n".join(lines[-40:]), None
    if "error" in box:
        lines.append(f"FEHLER: {box['error']}")
        yield "\n".join(lines[-40:]), None
        return
    yield "\n".join(lines[-40:]), box["result"]


def _measure_table(measures: dict[str, float], inputs: dict[str, float] | None = None) -> list[list]:
    rows = []
    for key, (label, unit) in MEASURES.items():
        if key in measures:
            entered = (inputs or {}).get(key)
            rows.append([label, f"{measures[key]:.1f} {unit}", "" if not entered else f"{entered:.1f} {unit}"])
    return rows


def _files(result) -> list[str]:
    return [v for k, v in result.files.items() if k.startswith(("glb", "obj"))]


def run_scan(files, name, gender, age, height, weight, refine, *optional):
    from . import pipeline

    if not files:
        yield "Bitte zuerst ein Video oder Fotos auswählen.", None, None, None, ""
        return
    paths = [Path(f if isinstance(f, str) else f.name) for f in files]
    person = _person(gender, age, height, weight, list(optional))
    problems = person.validate()
    if problems:
        yield "\n".join(problems), None, None, None, ""
        return
    source = paths[0] if len(paths) == 1 else paths
    for log, result in _stream(lambda log: pipeline.scan(source, person, name or "Ich",
                                                         refine=bool(refine), log=log)):
        if result is None:
            yield log, None, None, None, ""
        else:
            report = "\n".join(f"- {s}" for s in (result.report.summary if result.report else []))
            hints = "\n".join(f"- {h}" for h in result.hints) or "- Export entspricht den Cascadeur-Regeln."
            yield (log, _measure_table(result.measures, person.measures), result.files.get("glb"),
                   _files(result), f"**Aufnahme**\n{report}\n\n**Export**\n{hints}")


def run_measures_only(name, gender, age, height, weight, *optional):
    from . import pipeline

    person = _person(gender, age, height, weight, list(optional))
    problems = person.validate()
    if problems:
        yield "\n".join(problems), None, None, None
        return
    for log, result in _stream(lambda log: pipeline.body_from_measurements(person, name or "Ich", log=log)):
        if result is None:
            yield log, None, None, None
        else:
            yield log, _measure_table(result.measures, person.measures), result.files.get("glb"), _files(result)


def person_choices() -> list[str]:
    return [p.name for p in Store().persons()]


def history(name: str):
    store = Store()
    person = store.find(name or "")
    if person is None:
        return [], "Noch keine Scans gespeichert.", None
    scans = store.scans(person.id)
    rows = [[s.id, s.source, s.measures.get("height", ""), s.measures.get("weight", ""),
             s.measures.get("chest", ""), s.measures.get("waist", ""), s.measures.get("hips", "")]
            for s in scans]
    text = "Erst ein Scan vorhanden."
    if len(scans) >= 2:
        diff = compare(scans[-2].measures, scans[-1].measures)
        text = "**Veränderung zum vorherigen Scan**\n\n" + "\n".join(
            f"- {label}: {old:.1f} → {new:.1f} ({d:+.1f})" for label, old, new, d in diff)
    latest = scans[-1].files.get("glb") if scans else None
    return rows, text, latest


def restyle(name: str, weight: float, muscle: float):
    from .pipeline import reexport

    store = Store()
    person = store.find(name or "")
    latest = store.latest(person.id) if person else None
    if latest is None:
        return None, None, "Für diese Person gibt es noch keinen Scan."
    files = reexport(latest, changes={"weight": weight, "muscle": muscle})
    return files.get("glb"), [v for k, v in files.items() if k.startswith(("glb", "obj"))], "Neu exportiert."


def make_fbx(name: str):
    from .blender import convert

    store = Store()
    person = store.find(name or "")
    latest = store.latest(person.id) if person else None
    if latest is None:
        return None, "Für diese Person gibt es noch keinen Scan."
    out = []
    for key in ("glb", "glb_unreal"):
        if key in latest.files:
            glb = Path(latest.files[key])
            try:
                convert(glb, glb.with_suffix(".fbx"))
            except FileNotFoundError as exc:
                return None, str(exc)
            out.append(str(glb.with_suffix(".fbx")))
    return out, "FBX erstellt (Datei *_unreal.fbx ist für Cascadeur und Unreal benannt)."


def run_selftest(quick: bool):
    from . import selftest

    for log, result in _stream(lambda log: selftest.run(quick=quick, log=log)):
        yield (log if result is None else result.text())


def _show(path: str | None) -> str | None:
    return path


def _offline_env() -> None:
    """Gradio soll nichts nach Hause melden und nichts nachladen (FitScan läuft offline)."""
    for key, value in (("GRADIO_ANALYTICS_ENABLED", "False"), ("HF_HUB_OFFLINE", "1"),
                       ("HF_HUB_DISABLE_TELEMETRY", "1"), ("DO_NOT_TRACK", "1")):
        os.environ.setdefault(key, value)


def build():
    _offline_env()
    import gradio as gr

    with gr.Blocks(title="FitScan") as demo:
        gr.Markdown("# FitScan\nRiggbares 1:1-Körpermodell aus Video und Körpermaßen. Läuft komplett offline.")
        with gr.Tab("1. Vorbereitung"):
            gr.Markdown("## So gelingt der Scan\n" + "\n".join(f"- {c}" for c in CHECKLIST))
            gr.Markdown("Live-Prüfung mit der Webcam: `start_windows.bat live` im FitScan-Ordner ausführen. "
                        "Die Ampel zeigt, ob Licht, Abstand und Haltung stimmen; Leertaste nimmt auf.")
        with gr.Tab("2. Scan"):
            with gr.Row():
                with gr.Column():
                    files = gr.File(label="Ein Video oder mehrere Fotos (bei mehreren Videos wird das beste "
                                          "verwendet)", file_count="multiple", file_types=["video", "image"])
                    name = gr.Textbox(label="Name", value="Ich")
                    gender = gr.Dropdown(list(GENDERS), value="Männlich", label="Geschlecht")
                    age = gr.Number(label="Alter (Jahre)", value=30)
                    height = gr.Number(label="Körpergröße (cm, Pflicht)")
                    weight = gr.Number(label="Gewicht (kg, Pflicht)")
                    with gr.Accordion("Weitere Maße (optional, Maßband)", open=False):
                        optional = [gr.Number(label=f"{MEASURES[k][0]} (cm)") for k in OPTIONAL]
                    refine = gr.Checkbox(label="Kanten nachschärfen (nur bei unruhigem Hintergrund)",
                                         value=False)
                    start = gr.Button("Scan starten", variant="primary")
                    glb_path = gr.State()
                with gr.Column():
                    model3d = gr.Model3D(label="Körpermodell")
                    table = gr.Dataframe(headers=["Maß", "Modell", "Eingabe"], label="Maße")
                    report = gr.Markdown()
                    downloads = gr.File(label="Dateien", file_count="multiple")
                    log = gr.Textbox(label="Ablauf", lines=8)
            # Model3D zeigt Werte aus laufenden Generatoren nicht an, daher im Folgeschritt setzen
            start.click(run_scan, [files, name, gender, age, height, weight, refine, *optional],
                        [log, table, glb_path, downloads, report]).then(_show, glb_path, model3d)
        with gr.Tab("3. Nur aus Maßen"):
            with gr.Row():
                with gr.Column():
                    m_name = gr.Textbox(label="Name", value="Ich")
                    m_gender = gr.Dropdown(list(GENDERS), value="Männlich", label="Geschlecht")
                    m_age = gr.Number(label="Alter (Jahre)", value=30)
                    m_height = gr.Number(label="Körpergröße (cm)")
                    m_weight = gr.Number(label="Gewicht (kg)")
                    m_optional = [gr.Number(label=f"{MEASURES[k][0]} (cm)") for k in OPTIONAL]
                    m_start = gr.Button("Körper erstellen", variant="primary")
                    m_glb = gr.State()
                with gr.Column():
                    m_model = gr.Model3D(label="Körpermodell")
                    m_table = gr.Dataframe(headers=["Maß", "Modell", "Eingabe"], label="Maße")
                    m_files = gr.File(label="Dateien", file_count="multiple")
                    m_log = gr.Textbox(label="Ablauf", lines=6)
            m_start.click(run_measures_only, [m_name, m_gender, m_age, m_height, m_weight, *m_optional],
                          [m_log, m_table, m_glb, m_files]).then(_show, m_glb, m_model)
        with gr.Tab("4. Verlauf und Statur"):
            with gr.Row():
                who = gr.Dropdown(person_choices(), label="Person", allow_custom_value=True)
                refresh = gr.Button("Aktualisieren")
            h_table = gr.Dataframe(headers=["Scan", "Quelle", "Größe", "Gewicht", "Brust", "Taille", "Hüfte"])
            h_text = gr.Markdown()
            h_model = gr.Model3D(label="Letzter Scan")
            with gr.Row():
                s_weight = gr.Slider(0, 1, value=0.5, step=0.01, label="Gewicht (dünn ↔ kräftig)")
                s_muscle = gr.Slider(0, 1, value=0.5, step=0.01, label="Muskeln (wenig ↔ athletisch)")
            with gr.Row():
                s_go = gr.Button("Statur ändern und exportieren")
                fbx_go = gr.Button("FBX erstellen (braucht Blender)")
            s_files = gr.File(label="Dateien", file_count="multiple")
            s_msg = gr.Markdown()
            refresh.click(lambda: gr.Dropdown(choices=person_choices()), None, who)
            who.change(history, who, [h_table, h_text, h_model])
            s_go.click(restyle, [who, s_weight, s_muscle], [h_model, s_files, s_msg])
            fbx_go.click(make_fbx, who, [s_files, s_msg])
        with gr.Tab("5. Selbsttest"):
            gr.Markdown("Prüft Installation und Anpassung mit einem künstlichen Testkörper (ohne eigenes Video).")
            quick = gr.Checkbox(label="Schnelltest", value=True)
            t_go = gr.Button("Selbsttest starten")
            t_out = gr.Textbox(label="Ergebnis", lines=18)
            t_go.click(run_selftest, quick, t_out)
    return demo


def launch(port: int = 7860, open_browser: bool = True) -> None:
    demo = build()
    demo.queue().launch(server_name="127.0.0.1", server_port=port, inbrowser=open_browser)
