"""Isaac Sim 6.1 "VLM Scene Caption" (``isaacsim.replicator.caption.core``, IRC)
on the G1's D435 view.

IRC renders the camera view, builds a scene graph from every *semantically
labelled* prim it sees (``Semantics.SemanticsAPI`` class labels - the props in
``g1_sim.environments`` are labelled), and asks an OpenAI-compatible model to
write a caption. Outputs (image, scene graphs, captions JSON) go to
``<out_dir>/``.

Model: ``CAPTION_MODEL_URL`` / ``CAPTION_MODEL_NAME``. Default is OpenAI
``gpt-6-luna`` when ``OPENAI_API_KEY`` is set (~2 s/request on 2026-09-24),
else NVIDIA ``moonshotai/kimi-k3`` (~67 s; ``deepseek-ai/deepseek-v4.1-flash``
>150 s; the extension's own ``deepseek-v4-flash-0731`` is no longer served).
Keys live in the repo's gitignored ``.env`` (loaded by ``.envrc``).

IRC speaks the NVIDIA-NIM dialect: it always prefers ``NVIDIA_API_KEY`` from
the environment, sends ``max_tokens``, ``chat_template_kwargs`` and
``temperature=0.2``/``top_p`` (gpt-6 models reject all of them) and
health-checks with ``max_tokens=1`` (too small for a reasoning model).
:func:`_use_openai` adapts that in-process; :func:`_patch_object_grouping`
fixes IRC merging our props into one scene-graph node.

Side effects to know about: IRC leaves a 1920x1080 render product behind and
``rep.orchestrator.step_async`` pauses the timeline - :func:`poll` re-plays it.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

EXT = "isaacsim.replicator.caption.core"
NVIDIA_URL, NVIDIA_MODEL = "https://integrate.api.nvidia.com/v1", "moonshotai/kimi-k3"
OPENAI_URL, OPENAI_MODEL = "https://api.openai.com/v1", "gpt-6-luna"
MIN_COMPLETION_TOKENS = 64  # reasoning models spend tokens before answering


def _patch_object_grouping() -> None:
    """IRC merges each seen prim into the nearest ancestor that carries a USD
    reference, else into its depth-2 ancestor, and keeps one node per result.
    Every labelled prop under /World/Props (holder Xforms, plain cubes)
    collapsed into "/World/Props" and only one of them survived. Treat a
    labelled prim as its own object, and IRC's "none" (``str(None)``) as no
    label."""
    from isaacsim.replicator.caption.core.utils import Utils

    get_label, capture = Utils.get_prim_semantic_label, Utils.capture_usd_info

    def get_prim_semantic_label(target_prim):
        label = get_label(target_prim)
        return None if label in (None, "none") else label

    def capture_usd_info(prim_path, stage, verbose=False):
        final_prim_path, usd_path = capture(prim_path, stage, verbose)
        if get_prim_semantic_label(stage.GetPrimAtPath(prim_path)):
            return prim_path, usd_path
        return final_prim_path, usd_path

    Utils.get_prim_semantic_label = get_prim_semantic_label
    Utils.capture_usd_info = capture_usd_info


def _use_openai() -> None:
    """Make IRC's requests valid for api.openai.com (this process only)."""
    from isaacsim.replicator.caption.core.sft_autolabeling import scene_graph_sft

    base = scene_graph_sft.OpenAI

    class OpenAICompat(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            create = self.chat.completions.create

            def compat_create(*a, max_tokens=None, extra_body=None, temperature=None, top_p=None, **kw):
                if max_tokens is not None:
                    kw["max_completion_tokens"] = max(max_tokens, MIN_COMPLETION_TOKENS)
                return create(*a, **kw)

            self.chat.completions.create = compat_create

    scene_graph_sft.OpenAI = OpenAICompat
    # IRC would otherwise send the NVIDIA key to OpenAI.
    os.environ.pop("NVIDIA_API_KEY", None)


def enable() -> None:
    import omni.kit.app

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(EXT, True)


def start(camera_prim: str, out_dir: Path) -> asyncio.Future | None:
    """Configure IRC for ``camera_prim`` and schedule one caption run. Returns
    the task, or None when the model endpoint is not usable."""
    import yaml
    from isaacsim.replicator.caption.core.api import CaptionAPI
    from isaacsim.replicator.caption.core.settings import ReplicatorCaptionSettings

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(ReplicatorCaptionSettings.get_default_config_file_path()) as f:
        config = yaml.safe_load(f)
    section = config["isaacsim.replicator.caption.core"]
    section.update(camera_prim_path=camera_prim, scene_path="", output_path=str(out_dir))
    config_path = out_dir / "caption_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    _patch_object_grouping()
    CaptionAPI.load_config_file(str(config_path))
    ReplicatorCaptionSettings.set_target_camera_prim_path(camera_prim)
    ReplicatorCaptionSettings.set_output_folder_path(str(out_dir))

    openai = bool(os.environ.get("OPENAI_API_KEY")) and "CAPTION_MODEL_URL" not in os.environ
    url = os.environ.get("CAPTION_MODEL_URL", OPENAI_URL if openai else NVIDIA_URL)
    name = os.environ.get("CAPTION_MODEL_NAME", OPENAI_MODEL if "api.openai.com" in url else NVIDIA_MODEL)
    if "api.openai.com" in url:
        key = os.environ.get("OPENAI_API_KEY", "")
        _use_openai()
    else:
        # A self-hosted CAPTION_MODEL_URL (vLLM, llama.cpp) usually needs no key.
        key = os.environ.get("NVIDIA_API_KEY") or ("EMPTY" if "CAPTION_MODEL_URL" in os.environ else "")
    if not key:
        print("[CAP] no API key for the caption model (fill .env) - caption skipped")
        return None
    CaptionAPI.set_model_params(url, name, key)
    ok, msg = CaptionAPI.check_model()
    print(f"[CAP] model {name} @ {url}: {'OK' if ok else 'FAILED'} - {msg}")
    if not ok:
        return None
    print(f"[CAP] captioning {camera_prim} -> {out_dir}")
    return asyncio.ensure_future(CaptionAPI.get_captions())


def poll(task: asyncio.Future | None) -> asyncio.Future | None:
    """Call every sim step. Prints the result once and returns None when done."""
    if task is None or not task.done():
        return task
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()
    if task.exception() is not None:
        print(f"[CAP] caption failed: {task.exception()!r}")
    else:
        result = task.result() or {}
        print(f"[CAP] {len(result.get('nodes', {}))} objects; brief caption: {result.get('brief_caption')}")
    return None
