#!/usr/bin/env python3
"""Run three-scenario budget runway simulation for life-decision."""
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


def money_sum(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return sum(money_sum(v) for v in value.values())
    if isinstance(value, list):
        return sum(money_sum(v) for v in value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def light(months: float) -> str:
    if months != float("inf") and months < 2:
        return "红灯"
    if months != float("inf") and months < 6:
        return "黄灯"
    return "绿灯"


def advice(color: str) -> str:
    if color == "红灯":
        return "必须先找现金流，不允许裸辞/全职创业/长期备考。"
    if color == "黄灯":
        return "可以小成本试错，但必须有周级止损点和最低收入来源。"
    return "有试错窗口，但仍要定复盘点和止损线。"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="Input JSON payload")
    parser.add_argument("--file", help="Path to JSON payload")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()
    b = load_payload(args)

    cash = money_sum(b.get("可用现金")) + money_sum(b.get("可快速变现资产", b.get("可变现资产")))
    income = money_sum(b.get("月稳定收入", b.get("月固定收入")))
    necessary = money_sum(b.get("月必要支出"))
    compressible = money_sum(b.get("月可压缩支出"))
    one_off = money_sum(b.get("一次性即将发生支出"))
    family_obj = b.get("家人支持", {}) if isinstance(b.get("家人支持", {}), dict) else {}
    family = money_sum(family_obj.get("每月支持"))

    scenarios = b.get("scenarios") or {
        "乐观": {"income_factor": 1.0, "expense_factor": 0.9, "family_factor": 1.0, "shock": 0},
        "正常": {"income_factor": 1.0, "expense_factor": 1.0, "family_factor": 0.8, "shock": one_off},
        "保守": {"income_factor": 0.7, "expense_factor": 1.2, "family_factor": 0.3, "shock": one_off + 1000},
    }

    rows = []
    for name, s in scenarios.items():
        monthly_income = income * float(s.get("income_factor", 1.0)) + family * float(s.get("family_factor", 1.0))
        monthly_expense = necessary * float(s.get("expense_factor", 1.0)) + compressible * float(s.get("compressible_factor", 0.5))
        burn = max(0.0, monthly_expense - monthly_income)
        usable_cash = max(0.0, cash - money_sum(s.get("shock")))
        months = float("inf") if burn == 0 else usable_cash / burn
        color = light(months)
        rows.append({
            "scenario": name,
            "usable_cash": usable_cash,
            "monthly_income": monthly_income,
            "monthly_expense": monthly_expense,
            "monthly_burn": burn,
            "runway_months": None if months == float("inf") else months,
            "light": color,
            "advice": advice(color),
        })

    result = {
        "base": {
            "cash_and_liquid_assets": cash,
            "monthly_stable_income": income,
            "monthly_family_support": family,
            "monthly_necessary_expense": necessary,
            "monthly_compressible_expense": compressible,
            "one_off_expense": one_off,
        },
        "scenarios": rows,
    }

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    lights = {row["scenario"]: row["light"] for row in rows}
    normal_row = next((r for r in rows if r["scenario"] in ("正常", "normal")), rows[0] if rows else None)
    normal_runway = normal_row["runway_months"] if normal_row else None
    write_state("budget", {
        "ts": ts,
        "normal_runway_months": normal_runway,
        "lights": lights,
    })
    runway_txt = "不烧钱" if normal_runway is None else f"{normal_runway:.1f}个月"
    stamp = f"[凭证] budget ts={ts} 正常档跑道={runway_txt} 灯={'/'.join(f'{k}:{v}' for k, v in lights.items())}"

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(stamp, file=sys.stderr)
        return

    base = result["base"]
    print("[基础账]")
    print(f"可用现金+可快速变现资产: {base['cash_and_liquid_assets']:.0f}元")
    print(f"月稳定收入: {base['monthly_stable_income']:.0f}元")
    print(f"家人月支持: {base['monthly_family_support']:.0f}元")
    print(f"月必要支出: {base['monthly_necessary_expense']:.0f}元")
    print(f"月可压缩支出: {base['monthly_compressible_expense']:.0f}元")
    print(f"一次性即将发生支出: {base['one_off_expense']:.0f}元")
    print("\n[三档压力测试]")
    for row in rows:
        months = "无限/不烧钱" if row["runway_months"] is None else f"{row['runway_months']:.1f}个月"
        print(
            f"{row['scenario']}: 可用现金{row['usable_cash']:.0f}元, "
            f"月净消耗{row['monthly_burn']:.0f}元, 跑道{months}, {row['light']}"
        )
        print(f"  -> {row['advice']}")
    print(stamp)


if __name__ == "__main__":
    main()
