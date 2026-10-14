"""Körpermodell: dünne Schicht über Anny (Apache 2.0, Geometrie CC0).

Koordinaten wie in Anny: Meter, Z nach oben, die Person schaut in Richtung -Y,
ihre linke Seite liegt bei +X.

Eine Person wird beschrieben durch
- Phänotyp-Regler (Geschlecht, Alter, Muskeln, Gewicht, Größe, Proportionen), je 0..1,
- lokale Formregler (z. B. Taillenumfang), je -1..1,
- optionale Verschiebungen pro Vertex (Offsets) für die individuelle Feinform,
- einen Gesamtmaßstab, mit dem die eingegebene Körpergröße exakt getroffen wird.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path

import numpy as np
import torch

PHENOTYPES = ["gender", "age", "muscle", "weight", "height", "proportions"]

# Anny: gender 0 = männlich, 1 = weiblich.
GENDER_VALUES = {"maennlich": 0.0, "weiblich": 1.0, "divers": 0.5}

# Altersanker von Anny/MakeHuman: Regler -> Jahre
_AGE_ANCHORS = [(0.0, 1.0), (1 / 3, 11.0), (2 / 3, 25.0), (1.0, 90.0)]

ARM_BONE_PREFIXES = ("shoulder01", "upperarm", "lowerarm", "wrist", "finger", "metacarpal")


def age_to_param(years: float) -> float:
    years = float(np.clip(years, 1.0, 90.0))
    for (p0, y0), (p1, y1) in zip(_AGE_ANCHORS, _AGE_ANCHORS[1:], strict=False):
        if years <= y1:
            return p0 + (years - y0) / (y1 - y0) * (p1 - p0)
    return 1.0


def param_to_age(p: float) -> float:
    for (p0, y0), (p1, y1) in zip(_AGE_ANCHORS, _AGE_ANCHORS[1:], strict=False):
        if p <= p1:
            return y0 + (p - p0) / (p1 - p0) * (y1 - y0)
    return 90.0


@dataclasses.dataclass
class BodyParams:
    phenotype: dict[str, float] = dataclasses.field(
        default_factory=lambda: {k: 0.5 for k in PHENOTYPES}
    )
    local: dict[str, float] = dataclasses.field(default_factory=dict)
    scale: float = 1.0
    offsets: np.ndarray | None = None  # (V, 3) in Metern, Ruhepose, vor dem Maßstab

    def to_json(self) -> dict:
        return {
            "phenotype": {k: float(v) for k, v in self.phenotype.items()},
            "local": {k: float(v) for k, v in self.local.items() if abs(v) > 1e-6},
            "scale": float(self.scale),
            "offsets": None if self.offsets is None else "offsets.npy",
        }

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "params.json").write_text(json.dumps(self.to_json(), indent=2), "utf-8")
        if self.offsets is not None:
            np.save(folder / "offsets.npy", self.offsets.astype(np.float32))

    @classmethod
    def load(cls, folder: Path) -> BodyParams:
        data = json.loads((folder / "params.json").read_text("utf-8"))
        offsets = None
        if data.get("offsets"):
            offsets = np.load(folder / data["offsets"])
        return cls(
            phenotype=dict(data["phenotype"]),
            local=dict(data.get("local", {})),
            scale=float(data.get("scale", 1.0)),
            offsets=offsets,
        )

    def copy(self) -> BodyParams:
        return BodyParams(
            phenotype=dict(self.phenotype),
            local=dict(self.local),
            scale=self.scale,
            offsets=None if self.offsets is None else self.offsets.copy(),
        )


class BodyModel:
    """Lädt Anny einmal pro Prozess und bietet die Rechenwege, die FitScan braucht."""

    _lock = threading.Lock()
    _cache: dict[str, BodyModel] = {}

    def __init__(self, device: str = "cpu"):
        import anny

        self.device = torch.device(device)
        self.anny = anny.Anny(local_changes="default").to(dtype=torch.float32, device=self.device)
        self.anny.eval()
        self.phenotype_labels: list[str] = list(self.anny.phenotype_labels)
        self.local_labels: list[str] = list(self.anny.local_change_labels)
        self.bone_labels: list[str] = list(self.anny.bone_labels)
        self.bone_parents: list[int] = [int(p) for p in self.anny.bone_parents]
        self.faces: np.ndarray = self.anny.faces.cpu().numpy().astype(np.int64)
        self.vertex_bone_indices = self.anny.vertex_bone_indices.cpu().numpy()
        self.vertex_bone_weights = self.anny.vertex_bone_weights.cpu().numpy()
        self.num_vertices = int(self.anny.template_vertices.shape[-2])
        self._identity_pose = torch.eye(4, device=self.device)[None, None].repeat(
            1, len(self.bone_labels), 1, 1
        )

    @classmethod
    def get(cls, device: str | None = None) -> BodyModel:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        with cls._lock:
            if device not in cls._cache:
                cls._cache[device] = cls(device)
            return cls._cache[device]

    # ----------------------------------------------------------------- Tensoren
    def phenotype_tensor(self, params: BodyParams) -> torch.Tensor:
        vals = [params.phenotype.get(k, 0.5) for k in self.phenotype_labels]
        return torch.tensor([vals], dtype=torch.float32, device=self.device)

    def local_tensor(self, params: BodyParams) -> torch.Tensor:
        vals = [params.local.get(k, 0.0) for k in self.local_labels]
        return torch.tensor([vals], dtype=torch.float32, device=self.device)

    def rest(
        self,
        phenotype: torch.Tensor,
        local: torch.Tensor,
        offsets: torch.Tensor | None = None,
        scale: torch.Tensor | float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Ruhepose (Annys Referenzhaltung) mit Bones. Alles differenzierbar."""
        facial = torch.zeros((1, len(self.anny.facial_action_labels)), device=self.device)
        coeffs = self.anny._get_phenotype_blendshape_coefficients(phenotype, local, facial)
        out = self.anny.get_rest_model(coeffs)
        verts = out["rest_vertices"]
        if offsets is not None:
            verts = verts + offsets[None]
        return {
            "vertices": verts * scale,
            "bone_poses": _scale_poses(out["rest_bone_poses"], scale),
            "bone_heads": out["rest_bone_heads"] * scale,
            "_unscaled_bone_poses": out["rest_bone_poses"],
            "_unscaled_vertices": verts,
        }

    def posed(
        self,
        rest: dict[str, torch.Tensor],
        pose: torch.Tensor | None,
        scale: torch.Tensor | float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Skinnt die Ruhepose mit lokalen Gelenkdrehungen (B, J, 4, 4)."""
        if pose is None:
            pose = self._identity_pose
        rest_poses = rest["_unscaled_bone_poses"]
        bone_transforms, bone_poses = self.anny.get_bone_transforms(pose, rest_poses)
        verts = self.anny.get_skinned_vertices(
            rest["_unscaled_vertices"].expand(bone_transforms.shape[0], -1, -1), bone_transforms
        )
        unscaled_heads = rest["bone_heads"] / scale
        heads, _ = self.anny.get_bone_ends(unscaled_heads, unscaled_heads, rest_poses, bone_poses)
        return {"vertices": verts * scale, "bone_heads": heads * scale}

    # ------------------------------------------------------------ Bequemlichkeit
    @torch.no_grad()
    def rest_numpy(self, params: BodyParams) -> dict[str, np.ndarray]:
        offsets = None
        if params.offsets is not None:
            offsets = torch.as_tensor(params.offsets, dtype=torch.float32, device=self.device)
        out = self.rest(
            self.phenotype_tensor(params), self.local_tensor(params), offsets, params.scale
        )
        return {
            "vertices": out["vertices"][0].cpu().numpy(),
            "bone_poses": out["bone_poses"][0].cpu().numpy(),
            "bone_heads": out["bone_heads"][0].cpu().numpy(),
        }

    def bone(self, name: str) -> int:
        return self.bone_labels.index(name)

    def dominant_bone_mask(self, predicate) -> np.ndarray:
        """Vertices, deren Skinning-Gewicht überwiegend auf Bones liegt, für die predicate(name) gilt."""
        selected = np.array([predicate(n) for n in self.bone_labels], dtype=bool)
        w = np.where(selected[self.vertex_bone_indices], self.vertex_bone_weights, 0.0).sum(1)
        return w > 0.5

    def arm_vertex_mask(self) -> np.ndarray:
        return self.dominant_bone_mask(lambda n: n.startswith(ARM_BONE_PREFIXES))


def _scale_poses(poses: torch.Tensor, scale) -> torch.Tensor:
    if isinstance(scale, float) and scale == 1.0:
        return poses
    scaled = poses.clone()
    scaled[..., :3, 3] = poses[..., :3, 3] * scale
    return scaled


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """(..., 3) Achse*Winkel -> (..., 4, 4) homogene Drehung (Rodrigues), differenzierbar."""
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / theta
    kx, ky, kz = k.unbind(-1)
    zero = torch.zeros_like(kx)
    kmat = torch.stack(
        [zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=-1
    ).reshape(*aa.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(kmat)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    rot = eye + s * kmat + (1 - c) * (kmat @ kmat)
    out = torch.zeros(*aa.shape[:-1], 4, 4, dtype=aa.dtype, device=aa.device)
    out[..., :3, :3] = rot
    out[..., 3, 3] = 1.0
    return out
