#!/usr/bin/env python3
"""Mirror the Isaac People characters used by --env social_static
into assets/people/, so Spark never resolves the remote Isaac asset root
(the remote warehouse path stalled there).

    python3 scripts/fetch_people_assets.py            # ~130 MB
"""

from __future__ import annotations

import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
PREFIX = "Assets/Isaac/6.0/Isaac/People"
OUT = Path(__file__).resolve().parent.parent / "assets" / "people"

sys.path.insert(0, str(OUT.parent.parent))
from g1_sim.social_environments import STATIC_HUMANS  # noqa: E402


def list_keys(prefix: str) -> list[str]:
    keys, token = [], ""
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}" + (f"&continuation-token={token}" if token else "")
        body = urllib.request.urlopen(url, timeout=60).read().decode()
        keys += re.findall(r"<Key>([^<]*)</Key>", body)
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", body)
        if not m:
            return keys
        token = urllib.parse.quote(m.group(1))


def fetch(key: str) -> None:
    dst = OUT / key[len(PREFIX) + 1:]
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{BUCKET}/{urllib.parse.quote(key)}", dst)
    print(f"  {dst.relative_to(OUT)}")


def main() -> None:
    for character in sorted({c for _, c, *_ in STATIC_HUMANS}):
        for key in list_keys(f"{PREFIX}/Characters/{character}/"):
            if "/.thumbs/" not in key:
                fetch(key)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
