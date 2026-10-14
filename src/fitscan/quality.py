"""Scan-Assistent: prüft jedes Bild auf Licht, Schärfe, Position, Haltung, Stillhalten,
Hintergrund und Blickwinkel und gibt Hinweise in Alltagssprache.

Die Regeln sind bewusst einfach und als Konstanten sichtbar, damit sie später 1:1 in die
iPhone-App übernommen und an echten Testscans nachjustiert werden können.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np

from .vision import (
    FACE,
    L_ANKLE,
    L_ELBOW,
    L_HEEL,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_ANKLE,
    R_ELBOW,
    R_HEEL,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    FrameObservation,
)

ROT, GELB, GRUEN = "rot", "gelb", "gruen"

LIMITS = {
    "min_person_luma": 70,  # 0..255
    "max_person_luma": 225,
    "max_backlight": 50,  # Hintergrund heller als Person um mehr als ...
    "backlight_person_luma": 110,  # ... und die Person selbst eher dunkel
    "min_sharpness": 25.0,  # Varianz des Laplace-Filters auf 400 px Personenhöhe
    "min_height_fraction": 0.45,
    "max_height_fraction": 0.92,
    "max_center_offset": 0.25,
    "arm_angle_min_deg": 12.0,
    "arm_angle_max_deg": 60.0,
    "min_elbow_deg": 150.0,
    "feet_ratio_min": 0.8,
    "feet_ratio_max": 2.6,
    "max_arm_change_deg": 12.0,
    "max_soft_mask": 0.25,
    "max_background_edges": 0.12,
}

SECTORS = 12  # Blickwinkel in 30°-Schritten


@dataclasses.dataclass
class Hint:
    code: str
    text: str
    level: str


@dataclasses.dataclass
class FrameQuality:
    index: int
    level: str
    hints: list[Hint]
    metrics: dict[str, float]
    yaw: float | None

    @property
    def usable(self) -> bool:
        return self.level != ROT


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def arm_angles(kp: np.ndarray) -> tuple[float, float]:
    down = np.array([0.0, 1.0])
    left = _angle_deg(kp[L_ELBOW, :2] - kp[L_SHOULDER, :2], down)
    right = _angle_deg(kp[R_ELBOW, :2] - kp[R_SHOULDER, :2], down)
    return left, right


def check_frame(rgb: np.ndarray, obs: FrameObservation, prev: FrameObservation | None = None) -> FrameQuality:
    hints: list[Hint] = []
    metrics: dict[str, float] = {}

    def add(code: str, text: str, level: str) -> None:
        hints.append(Hint(code, text, level))

    if obs.keypoints is None:
        add("keine_person", "Keine Person erkannt. Ganz ins Bild stellen.", ROT)
        return FrameQuality(obs.index, ROT, hints, metrics, None)
    if obs.persons > 1:
        add("mehrere_personen", "Mehrere Personen im Bild. Nur die gescannte Person darf zu sehen sein.", ROT)

    h, w = rgb.shape[:2]
    kp = obs.keypoints
    mask = obs.mask if obs.mask is not None else np.zeros((h, w), np.float32)
    person = mask > 0.5
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Licht
    if person.sum() > 500:
        p_luma = float(gray[person].mean())
        bg = ~cv2.dilate(person.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        b_luma = float(gray[bg].mean()) if bg.any() else p_luma
        metrics.update(person_luma=p_luma, background_luma=b_luma)
        if p_luma < LIMITS["min_person_luma"]:
            add("dunkel", "Zu dunkel. Mehr Licht von vorn oder oben.", ROT)
        elif p_luma > LIMITS["max_person_luma"]:
            add("hell", "Überbelichtet. Licht etwas weicher machen oder Abstand zur Lampe vergrößern.", GELB)
        if b_luma - p_luma > LIMITS["max_backlight"] and p_luma < LIMITS["backlight_person_luma"]:
            add("gegenlicht", "Gegenlicht: Hinter dir ist es heller als vorn. Fenster oder Lampe gehören hinter die Kamera, nicht hinter dich.", GELB)

    # Schärfe im Personenbereich
    ys, xs = np.where(person)
    if len(ys) > 500:
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        crop = gray[y0 : y1 + 1, x0 : x1 + 1]
        factor = 400.0 / max(crop.shape[0], 1)
        crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        metrics["sharpness"] = sharp
        if sharp < LIMITS["min_sharpness"]:
            add("unscharf", "Unscharf. Langsamer drehen und das Handy ruhig halten.", ROT)

        # Position und Abstand
        frac = (y1 - y0) / h
        metrics["height_fraction"] = frac
        border = 2
        cut = person[:border].any() or person[-border:].any()
        if cut or kp[NOSE, 1] < 0 or max(kp[L_HEEL, 1], kp[R_HEEL, 1]) > h:
            add("abgeschnitten", "Kopf oder Füße sind abgeschnitten. Einen Schritt zurück.", ROT)
        elif frac > LIMITS["max_height_fraction"]:
            add("zu_nah", "Etwas zu nah. Einen kleinen Schritt zurück.", GELB)
        elif frac < LIMITS["min_height_fraction"]:
            add("zu_weit", "Zu weit weg. Etwas näher kommen, der ganze Körper soll das Bild gut füllen.", GELB)
        offset = abs((x0 + x1) / 2 - w / 2) / w
        metrics["center_offset"] = offset
        if offset > LIMITS["max_center_offset"]:
            add("mitte", "Bitte in die Bildmitte stellen.", GELB)

        # Hintergrund: weiche Maskenränder oder viele Kanten deuten auf Unruhe/wenig Kontrast
        soft = ((mask > 0.2) & (mask < 0.8)).sum() / max(person.sum(), 1)
        metrics["soft_mask"] = float(soft)
        box = np.zeros_like(person)
        pad = int(0.05 * h)
        box[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad] = True
        edges = cv2.Canny(gray, 80, 160) > 0
        outside = ~box
        density = float(edges[outside].mean()) if outside.any() else 0.0
        metrics["background_edges"] = density
        if soft > LIMITS["max_soft_mask"]:
            add("kontrast", "Person hebt sich schlecht vom Hintergrund ab. Ruhigere, andersfarbige Wand wählen.", GELB)
        elif density > LIMITS["max_background_edges"]:
            add("hintergrund", "Hintergrund ist sehr unruhig. Vor eine ruhige Wand stellen.", GELB)

    # Haltung (nur in Front- und Rückenansicht sinnvoll)
    yaw = obs.yaw_estimate
    frontal = yaw is None or abs(math.sin(yaw)) < 0.6
    if frontal and kp[[L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW], 2].min() > 0.5:
        la, ra = arm_angles(kp)
        metrics.update(arm_left_deg=la, arm_right_deg=ra)
        if min(la, ra) < LIMITS["arm_angle_min_deg"]:
            add("arme_anliegend", "Arme etwas vom Körper abspreizen (A-Pose).", GELB)
        elif max(la, ra) > LIMITS["arm_angle_max_deg"]:
            add("arme_hoch", "Arme etwas tiefer halten (A-Pose, etwa 30° vom Körper).", GELB)
        if kp[[L_WRIST, R_WRIST], 2].min() > 0.5:
            le = _angle_deg(kp[L_SHOULDER, :2] - kp[L_ELBOW, :2], kp[L_WRIST, :2] - kp[L_ELBOW, :2])
            re = _angle_deg(kp[R_SHOULDER, :2] - kp[R_ELBOW, :2], kp[R_WRIST, :2] - kp[R_ELBOW, :2])
            metrics.update(elbow_left_deg=le, elbow_right_deg=re)
            if min(le, re) < LIMITS["min_elbow_deg"]:
                add("arme_gebeugt", "Arme gestreckt halten.", GELB)
        if kp[[L_ANKLE, R_ANKLE, L_HIP, R_HIP], 2].min() > 0.5:
            hip = abs(kp[L_HIP, 0] - kp[R_HIP, 0])
            feet = abs(kp[L_ANKLE, 0] - kp[R_ANKLE, 0])
            ratio = feet / max(hip, 1.0)
            metrics["feet_ratio"] = ratio
            if ratio < LIMITS["feet_ratio_min"]:
                add("fuesse_eng", "Füße etwas auseinander, etwa schulterbreit.", GELB)
            elif ratio > LIMITS["feet_ratio_max"]:
                add("fuesse_weit", "Füße nicht so weit auseinander.", GELB)
        if prev is not None and prev.keypoints is not None:
            pla, pra = arm_angles(prev.keypoints)
            change = max(abs(la - pla), abs(ra - pra))
            metrics["arm_change_deg"] = change
            if change > LIMITS["max_arm_change_deg"]:
                add("stillhalten", "Bitte stillhalten, Arme in der gleichen Position lassen.", GELB)

    if np.mean(kp[FACE, 2]) < 0.3 and frontal and yaw is not None and abs(yaw) < 0.5:
        metrics["face_hidden"] = 1.0

    level = ROT if any(x.level == ROT for x in hints) else GELB if hints else GRUEN
    return FrameQuality(obs.index, level, hints, metrics, yaw)


def sector_of(yaw: float) -> int:
    return int(round((yaw % (2 * math.pi)) / (2 * math.pi) * SECTORS)) % SECTORS


def sector_name(sector: int) -> str:
    deg = sector * 360 // SECTORS
    names = {0: "vorn", 90: "rechte Seite", 180: "Rücken", 270: "linke Seite"}
    if deg in names:
        return names[deg]
    base = min(names, key=lambda d: min(abs(d - deg), 360 - abs(d - deg)))
    return f"schräg {names[base]}"


@dataclasses.dataclass
class ScanReport:
    frames: int
    usable: int
    covered: list[bool]
    level: str
    summary: list[str]
    top_hints: list[tuple[str, int]]

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def summarize(qualities: list[FrameQuality], min_sectors: int = 8) -> ScanReport:
    covered = [False] * SECTORS
    counts: dict[str, int] = {}
    for q in qualities:
        for hint in q.hints:
            counts[hint.text] = counts.get(hint.text, 0) + 1
        if q.usable and q.yaw is not None:
            covered[sector_of(q.yaw)] = True
    usable = sum(q.usable for q in qualities)
    summary = []
    n_cov = sum(covered)
    if usable == 0:
        summary.append("Kein Bild war brauchbar. Bitte Checkliste beachten und neu aufnehmen.")
    missing = [sector_name(i) for i, c in enumerate(covered) if not c]
    if missing:
        summary.append("Diese Ansichten fehlen noch oder waren nicht brauchbar: " + ", ".join(missing) + ".")
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    for text, n in top:
        summary.append(f"{text} ({n}× im Video)")
    if usable and n_cov >= min_sectors and not top:
        summary.append("Sehr gute Aufnahme: alle Ansichten sauber erfasst.")
    level = GRUEN if (n_cov >= min_sectors and usable) else GELB if usable else ROT
    return ScanReport(len(qualities), usable, covered, level, summary, top)


CHECKLIST = [
    "Eng anliegende Kleidung, keine weiten Ärmel oder Hosen; lange Haare zusammenbinden",
    "Ruhiger, einfarbiger Hintergrund, der sich farblich von Kleidung und Haut abhebt",
    "Helles, gleichmäßiges Licht von vorn oder oben, kein Fenster oder keine Lampe hinter dir",
    "Etwa 2 m Abstand, Handy hochkant auf Hüfthöhe (Stativ), ganzer Körper mit etwas Rand über Kopf und unter Füßen im Bild",
    "A-Pose: Arme gestreckt, etwa 30° vom Körper, Füße schulterbreit, Blick geradeaus",
    "Langsam einmal komplett auf der Stelle drehen (etwa 20 Sekunden für eine Runde)",
]
