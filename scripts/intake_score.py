#!/usr/bin/env python3
"""Compute life-decision information collection coverage."""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
from typing import Any

STATE_PATH = pathlib.Path.home() / ".life-decision" / "state.json"


def write_state(key: str, entry: dict[str, Any]) -> None:
    """Record gate state so mcts_sim.py can verify prerequisites ran for real."""
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {}
        if STATE_PATH.exists():
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        state[key] = entry
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARN: 凭证状态写入失败: {exc}", file=sys.stderr)

WEIGHTS = {
    "D1": 0.18,
    "D2": 0.12,
    "D3": 0.14,
    "D4": 0.14,
    "D5": 0.12,
    "D6": 0.08,
    "D7": 0.14,
    "D8": 0.08,
}
ITEM_COUNTS = {
    "D1": 7,
    "D2": 6,
    "D3": 9,
    "D4": 7,
    "D5": 6,
    "D6": 4,
    "D7": 5,
    "D8": 5,
}
STATUS_VALUE = {
    "known": 1.0,
    "已知": 1.0,
    "yes": 1.0,
    "partial": 0.5,
    "部分": 0.5,
    "maybe": 0.5,
    "refused": 0.3,
    "拒绝": 0.3,
    "unknown": 0.0,
    "未知": 0.0,
    "no": 0.0,
}
DEPTH_RANK = {
    "none": 0,
    "无": 0,
    "shallow": 1,
    "浅": 1,
    "D-浅": 1,
    "mid": 2,
    "middle": 2,
    "中": 2,
    "D-中": 2,
    "deep": 3,
    "深": 3,
    "D-深": 3,
}


def load_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.file:
        return json.loads(open(args.file, encoding="utf-8").read())
    if args.json:
        return json.loads(args.json)
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return json.loads(data)
    raise SystemExit("Provide --json, --file, or JSON on stdin.")


def value(item: Any) -> float:
    if isinstance(item, (int, float)):
        return max(0.0, min(1.0, float(item)))
    return STATUS_VALUE.get(str(item).strip(), 0.0)


def score_dimension(items: Any, expected_count: int) -> tuple[float, float, int]:
    if isinstance(items, dict):
        vals = [value(v) for v in items.values()]
    elif isinstance(items, list):
        vals = [value(v) for v in items]
    else:
        vals = []
    known_points = sum(vals[:expected_count])
    denominator = expected_count
    return known_points / denominator if denominator else 0.0, known_points, denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="Input JSON payload")
    parser.add_argument("--file", help="Path to JSON payload")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()
    payload = load_payload(args)

    door = payload.get("door", "two_way")
    threshold = 0.85 if door in {"one_way", "single", "单向门", "不可逆"} else 0.70
    dimensions = payload.get("dimensions", {})
    depth = payload.get("depth", {})

    rows = []
    total = 0.0
    for dim, weight in WEIGHTS.items():
        ratio, points, denominator = score_dimension(dimensions.get(dim, []), ITEM_COUNTS[dim])
        weighted = ratio * weight
        total += weighted
        rows.append({
            "dimension": dim,
            "points": points,
            "denominator": denominator,
            "ratio": ratio,
            "weight": weight,
            "weighted": weighted,
            "depth": depth.get(dim, "none"),
        })

    d3_ok = DEPTH_RANK.get(str(depth.get("D3", "none")), 0) >= 3
    d6_ok = DEPTH_RANK.get(str(depth.get("D6", "none")), 0) >= 3
    result = {
        "door": door,
        "threshold": threshold,
        "total": total,
        "pass_collection": total >= threshold,
        "depth_check": {"D3_deep": d3_ok, "D6_deep": d6_ok, "pass": d3_ok and d6_ok},
        "dimensions": rows,
    }

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    passed = total >= threshold and d3_ok and d6_ok
    write_state("intake", {
        "ts": ts,
        "score": round(total, 4),
        "door": door,
        "threshold": threshold,
        "d3_deep": d3_ok,
        "d6_deep": d6_ok,
        "passed": passed,
    })
    stamp = (
        f"[凭证] intake ts={ts} score={total*100:.1f}% door={door} "
        f"threshold={threshold:.0%} D3深={'是' if d3_ok else '否'} "
        f"D6深={'是' if d6_ok else '否'} 允许分析={'是' if passed else '否'}"
    )

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(stamp, file=sys.stderr)
        return

    for row in rows:
        print(
            f"{row['dimension']}: {row['points']:.1f}/{row['denominator']} "
            f"({row['ratio']*100:.1f}%) weight={row['weight']:.0%} "
            f"depth={row['depth']}"
        )
    print("-" * 48)
    print(f"加权总收集率: {total*100:.1f}% / 门槛 {threshold*100:.0f}%")
    print(f"D3深度达标: {'是' if d3_ok else '否'}")
    print(f"D6深度达标: {'是' if d6_ok else '否'}")
    print(f"是否允许分析: {'是' if passed else '否'}")
    print(stamp)


if __name__ == "__main__":
    main()
