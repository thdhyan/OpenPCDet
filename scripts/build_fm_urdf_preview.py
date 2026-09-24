#!/usr/bin/env python3
"""Build the browser-only G1 URDF used by the Foundation Model Debug UI.

The simulator keeps the full physics description under ``assets/``. The UI
only needs a lightweight visual kinematic model, so this removes inertial and
collision data plus the LiDAR mesh and copies only visual meshes.

Run from the repository root:
    python scripts/build_fm_urdf_preview.py
"""
from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO / "assets" / "robot" / "g1_29" / "g1_29dof.urdf"
DEFAULT_OUTPUT = REPO / "fm_debug_ui" / "assets" / "g1_29dof.urdf"
EXCLUDED_VISUAL_MESHES = {"meshes/mid360.stl"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"source URDF not found: {args.source}")

    root = ET.parse(args.source).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {"inertial", "collision", "mujoco"}:
                parent.remove(child)

    # Keep only visual links. Drop the LiDAR mesh because it is irrelevant to
    # an arm-action preview and is the largest asset in the source package.
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is not None and mesh.attrib.get("filename") in EXCLUDED_VISUAL_MESHES:
                link.remove(visual)

    mesh_refs = {
        mesh.attrib["filename"]
        for mesh in root.findall(".//visual/geometry/mesh")
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh_output = args.output.parent / "meshes"
    mesh_output.mkdir(exist_ok=True)
    for old in mesh_output.glob("*.STL"):
        old.unlink()
    for old in mesh_output.glob("*.stl"):
        old.unlink()

    for relative in sorted(mesh_refs):
        source = args.source.parent / relative
        destination = mesh_output / Path(relative).name
        shutil.copy2(source, destination)

    ET.indent(root, space="  ")
    root.set("name", "g1_29dof_preview")
    args.output.write_text(ET.tostring(root, encoding="unicode") + "\n")
    print(f"wrote {args.output} and {len(mesh_refs)} visual meshes")


if __name__ == "__main__":
    main()
