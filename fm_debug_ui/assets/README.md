# G1 visual URDF

`g1_29dof.urdf` is a browser-preview derivative of
`assets/robot/g1_29/g1_29dof.urdf`.

- It is kinematic only: no inertials, collision geometry, or physics.
- The LiDAR visual mesh is omitted.
- Meshes are copied under `meshes/` for the Three.js URDF loader.
- Regenerate from the repository root after changing the source description:
  ```bash
  python scripts/build_fm_urdf_preview.py
  ```
