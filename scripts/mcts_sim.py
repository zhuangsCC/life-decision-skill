#!/usr/bin/env python3
"""Run Life MCTS and personality simulation from a JSON case payload."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import pathlib
import random
import sys
from typing import Any

STATE_PATH = pathlib.Path.home() / ".life-decision" / "state.json"
GATE_MAX_AGE_HOURS = 24


def read_state() -> dict[str, Any]:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def entry_age_hours(entry: dict[str, Any]) -> float | None:
    try:
        ts = datetime.datetime.fromisoformat(str(entry.get("ts")))
        return (datetime.datetime.now() - ts).total_seconds() / 3600
    except (TypeError, ValueError):
        return None


def check_gates(skip_budget_reason: str | None) -> tuple[list[str], str]:
    """Return (blocking problems, gate provenance summary for the stamp)."""
    state = read_state()
    problems: list[str] = []
    prov_parts: list[str] = []

    intake = state.get("intake")
    if not intake:
        problems.append(
            "没有信息收集凭证：必须先运行 scripts/intake_score.py 并达标。"
            "现在就去按维度整理已收集信息并运行它，不要跳过。"
        )
    else:
        age = entry_age_hours(intake)
        if age is None or age > GATE_MAX_AGE_HOURS:
            problems.append(
                f"信息收集凭证已过期（超过{GATE_MAX_AGE_HOURS}小时）：请基于当前对话重新运行 intake_score.py。"
            )
        elif not intake.get("passed"):
            problems.append(
                f"信息收集未达标（score={intake.get('score', 0)*100:.1f}% / "
                f"threshold={intake.get('threshold', 0):.0%}，D3深={intake.get('d3_deep')}，"
                f"D6深={intake.get('d6_deep')}）：回到第一阶段继续追问，禁止在未达标时推演。"
            )
        else:
            prov_parts.append(f"intake {intake.get('score', 0)*100:.1f}%@{intake.get('ts')}")

    budget = state.get("budget")
    if budget:
        age = entry_age_hours(budget)
        if age is not None and age <= GATE_MAX_AGE_HOURS:
            runway = budget.get("normal_runway_months")
            runway_txt = "不烧钱" if runway is None else f"{runway:.1f}个月"
            prov_parts.append(f"budget {runway_txt}@{budget.get('ts')}")
        else:
            budget = None
    if not budget:
        if skip_budget_reason:
            prov_parts.append(f"budget 跳过(原因:{skip_budget_reason})")
        else:
            problems.append(
                "没有预算跑道凭证：涉及钱/工作/城市/创业/转行/学习投入的决策必须先运行 "
                "scripts/budget_sim.py。确实与财务完全无关的决策（如纯感情选择），"
                "用 --skip-budget \"具体原因\" 显式声明，原因会写进凭证由用户监督。"
            )

    return problems, ", ".join(prov_parts)


def clip(x: float) -> float:
    return max(0.0, min(100.0, x))


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(len(sorted_values) * q)))
    return sorted_values[idx]


def prob_value(spec: Any, params: dict[str, float]) -> float | None:
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, dict):
        if "value" in spec:
            return float(spec["value"])
        base = float(spec.get("base", 0.0))
        param = spec.get("param")
        if param:
            center = float(spec.get("center", 50.0))
            scale = float(spec.get("scale", 100.0)) or 100.0
            sign = float(spec.get("sign", 1.0))
            raw = base + sign * (float(params.get(param, center)) - center) / scale
        else:
            raw = base
        lo = float(spec.get("min", 0.02))
        hi = float(spec.get("max", 0.98))
        return max(lo, min(hi, raw))
    raise ValueError(f"Unsupported probability spec: {spec!r}")


class Node:
    __slots__ = ("year", "track", "N", "kids", "astat", "actions")

    def __init__(self, year: int, track: str, engine: "LifeMCTS") -> None:
        self.year = year
        self.track = track
        self.N = 0
        self.kids: dict[tuple[str, str], Node] = {}
        self.astat: dict[str, tuple[int, float]] = {}
        self.actions = engine.actions_of(track, year)


class LifeMCTS:
    def __init__(self, case: dict[str, Any]) -> None:
        self.c = case
        self.DIMS = case["dims"]
        self.W = case["weights"]
        self.H = int(case["horizon"])
        self.P0 = case["params"]
        self.tracks = case["tracks"]
        self.trans = case["transitions"]
        self.noise = float(case.get("noise", 5))
        self.couples = case.get("couplings", [])
        self.sl = case.get("stop_loss_line")
        self.rec = case.get("recovery_track")

    def comp(self, dims: dict[str, float]) -> float:
        return sum(float(dims[k]) * float(self.W[k]) for k in self.DIMS)

    def actions_of(self, track: str, year: int) -> list[str]:
        if year >= self.H:
            return []
        return list(self.tracks.get(track, {}).get("choices", []))

    def feasible(self, verbose: bool = False) -> list[str]:
        start = self.c["start"]
        hard = self.c.get("hard_limits", {})
        req = self.c.get("requires", {})
        choices = self.tracks.get(start, {}).get("choices", [])
        ok = []
        for opt in choices:
            need = req.get(opt, {})
            can = all(
                (hard.get(k, 0) >= v) if isinstance(v, (int, float)) else (hard.get(k) == v)
                for k, v in need.items()
            )
            if can:
                ok.append(opt)
            elif verbose:
                print(f"   [可行性筛掉] {opt}  缺条件:{need}（你只有 {hard}）")
        return ok

    def sample(self, action: str, params: dict[str, float]) -> dict[str, Any]:
        if action not in self.trans:
            raise ValueError(f"Missing transition for action: {action}")
        outs = self.trans[action]
        probs: list[float | None] = []
        fixed_sum = 0.0
        none_count = 0
        for outcome in outs:
            p = prob_value(outcome.get("p"), params) if "p" in outcome else None
            if p is None:
                none_count += 1
            else:
                fixed_sum += max(0.0, p)
            probs.append(p)
        residual = max(0.0, 1.0 - fixed_sum)
        filled = [
            (residual / none_count if p is None and none_count else max(0.0, p or 0.0))
            for p in probs
        ]
        total = sum(filled) or 1.0
        normalized = [p / total for p in filled]
        r = random.random()
        acc = 0.0
        for outcome, p in zip(outs, normalized):
            acc += p
            if r <= acc:
                return outcome
        return outs[-1]

    def step(self, dims: dict[str, float], impact: dict[str, float]) -> dict[str, float]:
        new_dims = {
            k: float(dims[k]) + random.gauss(float(impact.get(k, 0)), self.noise)
            for k in self.DIMS
        }
        for rule in self.couples:
            if new_dims[rule["if"]] < float(rule["below"]):
                for k, v in rule["then"].items():
                    new_dims[k] += random.gauss(float(v), 2)
        return {k: clip(v) for k, v in new_dims.items()}

    def transit(
        self, dims: dict[str, float], outcome: dict[str, Any], params: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float], str, bool]:
        new_dims = self.step(dims, outcome.get("impact", {}))
        new_params = dict(params)
        for k, v in outcome.get("on_outcome", {}).items():
            new_params[k] = clip(float(new_params.get(k, 50)) + float(v))
        to = outcome["to"]
        stop_loss_fired = False
        if self.sl is not None and self.comp(new_dims) < float(self.sl) and self.rec:
            to = self.rec
            stop_loss_fired = True
        return new_dims, new_params, to, stop_loss_fired

    def pick(self, node: Node) -> str | None:
        if not node.actions:
            return None
        for action in node.actions:
            if action not in node.astat:
                return action
        return max(
            node.actions,
            key=lambda a: (
                node.astat[a][1] / node.astat[a][0]
                + 1.4 * math.sqrt(max(0.0, math.log(max(1, node.N))) / node.astat[a][0])
            ),
        )

    def run(
        self,
        node: Node,
        dims: dict[str, float],
        params: dict[str, float],
        forced: str | None = None,
    ) -> tuple[float, dict[str, float]]:
        if node.year >= self.H or not node.actions:
            return self.comp(dims) / 100, dims
        node.N += 1
        action = forced or self.pick(node)
        if action is None:
            return self.comp(dims) / 100, dims
        outcome = self.sample(action, params)
        new_dims, new_params, to, _ = self.transit(dims, outcome, params)
        key = (action, outcome["label"])
        if key not in node.kids:
            node.kids[key] = Node(node.year + 1, to, self)
        reward, terminal = self.run(node.kids[key], new_dims, new_params)
        n, w = node.astat.get(action, (0, 0.0))
        node.astat[action] = (n + 1, w + reward)
        return reward, terminal

    def validate(self, strict: bool = True) -> list[str]:
        issues = []
        if abs(sum(float(self.W[k]) for k in self.DIMS) - 1.0) > 0.01:
            issues.append("weights must sum to approximately 1.0")
        if self.c["start"] not in self.tracks:
            issues.append(f"start track missing: {self.c['start']}")
        for dim in self.DIMS:
            if dim not in self.W:
                issues.append(f"weight missing for dimension: {dim}")
            if dim not in self.c.get("init", {}):
                issues.append(f"initial score missing for dimension: {dim}")
        all_actions = [a for tr in self.tracks.values() for a in tr.get("choices", [])]
        for action in all_actions:
            if action not in self.trans:
                issues.append(f"transition missing for action: {action}")
        for action, outcomes in self.trans.items():
            if not isinstance(outcomes, list) or not outcomes:
                issues.append(f"transition must have outcomes: {action}")
                continue
            for idx, outcome in enumerate(outcomes):
                for key in ("label", "to"):
                    if key not in outcome:
                        issues.append(f"{action}[{idx}] missing {key}")
                for dim in outcome.get("impact", {}):
                    if dim not in self.DIMS:
                        issues.append(f"{action}[{idx}] impact references unknown dimension: {dim}")
        if strict and issues:
            raise ValueError("; ".join(issues))
        return issues

    def decide(self, iters: int = 12000, verbose: bool = True) -> dict[str, Any]:
        self.validate(strict=True)
        opts = self.feasible(verbose=verbose)
        if not opts:
            raise ValueError("No feasible start options. Check hard_limits/requires.")
        root = Node(0, self.c["start"], self)
        root.actions = opts
        res = {o: {"c": [], "d": {k: [] for k in self.DIMS}} for o in opts}
        for i in range(iters):
            a0 = opts[i % len(opts)]
            reward, terminal = self.run(root, dict(self.c["init"]), dict(self.P0), forced=a0)
            res[a0]["c"].append(reward * 100)
            for k in self.DIMS:
                res[a0]["d"][k].append(terminal[k])

        names = self.c.get("names", {k: k for k in self.DIMS})
        rankings = []
        for opt, data in res.items():
            scores = sorted(data["c"])
            fatal = [
                names[k]
                for k in self.DIMS
                if percentile(sorted(data["d"][k]), 0.05) < 30
            ]
            next_best = []
            for (action, label), kid in root.kids.items():
                if action == opt and kid.astat:
                    nxt = max(kid.astat, key=lambda x: kid.astat[x][1] / kid.astat[x][0])
                    next_best.append({"if_outcome": label, "next_best": nxt})
            rankings.append(
                {
                    "option": opt,
                    "expected": sum(scores) / len(scores),
                    "p5": percentile(scores, 0.05),
                    "fatal_dimensions": fatal,
                    "next_best": next_best,
                }
            )
        rankings.sort(key=lambda x: -x["expected"])
        return {"recommendation": rankings[0]["option"], "rankings": rankings}


def appeal(choice: str, traits: dict[str, float], tags: dict[str, dict[str, float]]) -> float:
    t = tags.get(choice, {})

    def g(name: str) -> float:
        return (float(t.get(name, 50)) - 50) / 50

    def tr(name: str) -> float:
        return (float(traits.get(name, 50)) - 50) / 50

    return (
        tr("自律") * g("费力")
        + tr("学习意愿") * g("学习")
        + tr("风险偏好") * g("风险")
        + tr("责任心") * g("责任")
        + tr("行动力") * g("主动")
        + tr("享乐倾向") * g("爽感")
    )


def personality_pick(
    choices: list[str], traits: dict[str, float], tags: dict[str, dict[str, float]], temp: float = 1.5
) -> str:
    exps = [math.exp(appeal(c, traits, tags) / temp) for c in choices]
    total = sum(exps) or 1.0
    r = random.random()
    acc = 0.0
    for choice, exp_value in zip(choices, exps):
        acc += exp_value / total
        if r <= acc:
            return choice
    return choices[-1]


def predict_life(
    case: dict[str, Any],
    traits: dict[str, float],
    tags: dict[str, dict[str, float]],
    months: int = 12,
    sims: int = 3000,
) -> dict[str, Any]:
    engine = LifeMCTS(case)
    engine.validate(strict=True)
    opts0 = engine.feasible()
    if not opts0:
        raise ValueError("No feasible start options. Check hard_limits/requires.")

    ends = {"c": [], "d": {k: [] for k in engine.DIMS}}
    first: dict[str, int] = {}
    path_by_first: dict[str, list[tuple[int, str, str, float]]] = {}
    stop_loss_hit = 0
    awakening = case.get("awakening", {})
    emotion = case.get("emotion_profile", {})

    for _ in range(sims):
        dims = dict(case["init"])
        track = case["start"]
        tr = dict(traits)
        params = dict(case["params"])
        path = []
        triggered = False
        streak_neg = 0
        streak_pos = 0
        for m in range(months):
            choices = opts0 if track == case["start"] else engine.actions_of(track, m)
            if not choices:
                break
            choice = personality_pick(choices, tr, tags)
            outcome = engine.sample(choice, params)
            dims, params, track, sl_fire = engine.transit(dims, outcome, params)
            if sl_fire and not triggered:
                stop_loss_hit += 1
                triggered = True
            score = engine.comp(dims)
            path.append((m + 1, choice, outcome["label"], round(score, 1)))

            if score < 45:
                streak_neg += 1
                streak_pos = 0
                decay = 1.0 + streak_neg * float(emotion.get("rejection_sensitivity", 0.3)) * 0.3
                tr["自律"] = max(0, float(tr.get("自律", 50)) - decay)
                tr["享乐倾向"] = min(100, float(tr.get("享乐倾向", 50)) + decay)
                tr["行动力"] = max(0, float(tr.get("行动力", 50)) - decay * 0.5)
            elif score > 70:
                streak_pos += 1
                streak_neg = 0
                boost = 1.0 + streak_pos * float(emotion.get("momentum_boost", 0.2)) * 0.2
                tr["行动力"] = min(100, float(tr.get("行动力", 50)) + boost)
                tr["自律"] = min(100, float(tr.get("自律", 50)) + boost * 0.3)
            else:
                streak_neg = max(0, streak_neg - 1)
                streak_pos = max(0, streak_pos - 1)

            if awakening.get("active") and m < int(awakening.get("max_months", 6)):
                rate = float(awakening.get("rate", 0.5))
                tr["自律"] = min(100, float(tr.get("自律", 50)) + rate)
                tr["学习意愿"] = min(100, float(tr.get("学习意愿", 50)) + rate * 0.8)
                tr["行动力"] = min(100, float(tr.get("行动力", 50)) + rate * 0.6)
                tr["享乐倾向"] = max(0, float(tr.get("享乐倾向", 50)) - rate * 0.4)
            if m == 0:
                first[choice] = first.get(choice, 0) + 1

        if path:
            path_by_first.setdefault(path[0][1], path)
        ends["c"].append(engine.comp(dims))
        for k in engine.DIMS:
            ends["d"][k].append(dims[k])

    scores = sorted(ends["c"])
    first_total = sum(first.values()) or 1
    first_distribution = [
        {"option": choice, "percent": count / first_total * 100}
        for choice, count in sorted(first.items(), key=lambda x: -x[1])
    ]
    most_likely = first_distribution[0]["option"] if first_distribution else None
    return {
        "most_likely_first_choice": most_likely,
        "first_choice_distribution": first_distribution,
        "typical_path": path_by_first.get(most_likely, []),
        "avg_final_score": sum(scores) / len(scores) if scores else 0.0,
        "p5_final_score": percentile(scores, 0.05),
        "stop_loss_rate": stop_loss_hit / sims * 100,
    }


EXAMPLE_CASE = {
    "dims": ["F", "C", "H", "M", "R", "G", "S"],
    "names": {"F": "财务", "C": "事业", "H": "身体", "M": "心理", "R": "关系", "G": "成长", "S": "稳定"},
    "weights": {"F": 0.18, "C": 0.18, "H": 0.12, "M": 0.16, "R": 0.14, "G": 0.12, "S": 0.10},
    "params": {"能力弹药库": 70, "赛道景气度": 68, "方向清晰度": 45, "安全垫厚度": 40},
    "hard_limits": {"现金": 80000},
    "requires": {"裸辞全职创业": {"现金": 150000}},
    "start": "现状",
    "init": {"F": 48, "C": 62, "H": 70, "M": 55, "R": 62, "G": 58, "S": 65},
    "horizon": 3,
    "noise": 5,
    "couplings": [{"if": "F", "below": 35, "then": {"M": -6, "R": -4}}, {"if": "R", "below": 35, "then": {"M": -6, "H": -3}}],
    "stop_loss_line": 30,
    "recovery_track": "止损撤退",
    "tracks": {
        "现状": {"choices": ["裸辞全职创业", "先兼职试水", "安心打工"]},
        "试过之后": {"choices": ["转全职创业", "继续兼职攒钱", "安心打工"]},
        "创业中": {"choices": ["稳扎稳打", "加速扩张"]},
        "打工中": {"choices": ["争取晋升", "躺平摸鱼"]},
        "止损撤退": {"choices": ["赶紧上岸", "边做边等"]},
    },
    "transitions": {
        "裸辞全职创业": [{"label": "硬撑", "p": 1.0, "to": "创业中", "impact": {"F": -20, "C": 10, "G": 15, "S": -20, "M": -10}}],
        "先兼职试水": [
            {"label": "试出来适合", "p": 0.5, "to": "试过之后", "impact": {"F": 3, "C": 6, "G": 10, "M": 4, "S": -2}, "on_outcome": {"方向清晰度": 22, "安全垫厚度": 6}},
            {"label": "试出来不合适", "p": None, "to": "试过之后", "impact": {"F": 1, "C": 2, "G": 6, "M": -3, "S": -1}, "on_outcome": {"方向清晰度": 15}},
        ],
        "安心打工": [{"label": "平稳", "p": 1.0, "to": "打工中", "impact": {"F": 5, "C": 4, "M": 3, "S": 8, "R": 3}}],
        "转全职创业": [
            {"label": "做成了", "p": {"base": 0.28, "param": "方向清晰度", "center": 50, "scale": 130}, "to": "创业中", "impact": {"F": 25, "C": 28, "G": 25, "S": -10, "M": 8, "H": -8}},
            {"label": "没做成", "p": None, "to": "止损撤退", "impact": {"F": -25, "C": 5, "G": 18, "S": -25, "M": -22, "R": -12}},
        ],
        "继续兼职攒钱": [{"label": "稳步积累", "p": 1.0, "to": "试过之后", "impact": {"F": 8, "C": 5, "G": 6, "S": 4, "M": 2}}],
        "稳扎稳打": [
            {"label": "小步前进", "p": 0.55, "to": "创业中", "impact": {"F": 10, "C": 12, "G": 10, "M": 5}},
            {"label": "遇冷", "p": None, "to": "止损撤退", "impact": {"F": -15, "M": -12, "S": -10}},
        ],
        "加速扩张": [
            {"label": "起飞", "p": {"base": 0.30, "param": "赛道景气度", "center": 60, "scale": 200}, "to": "创业中", "impact": {"F": 30, "C": 25, "G": 18, "H": -12, "M": 6}},
            {"label": "扩崩了", "p": None, "to": "止损撤退", "impact": {"F": -35, "M": -25, "S": -30, "R": -10}},
        ],
        "争取晋升": [
            {"label": "升了", "p": {"base": 0.35, "param": "能力弹药库", "center": 60, "scale": 200}, "to": "打工中", "impact": {"F": 18, "C": 20, "M": 12, "S": 5, "H": -6}},
            {"label": "没升", "p": None, "to": "打工中", "impact": {"F": 2, "M": -8, "C": 2}},
        ],
        "躺平摸鱼": [{"label": "舒服但荒废", "p": 1.0, "to": "打工中", "impact": {"F": -3, "C": -8, "G": -8, "M": 2, "S": 3}}],
        "赶紧上岸": [
            {"label": "找到稳定工作", "p": {"base": 0.60, "param": "能力弹药库", "center": 60, "scale": 200}, "to": "打工中", "impact": {"F": 12, "M": 15, "S": 18, "C": 3, "R": 6}},
            {"label": "暂时没着落", "p": None, "to": "止损撤退", "impact": {"F": -8, "M": -10, "S": -8}},
        ],
        "边做边等": [{"label": "骑驴找马", "p": 1.0, "to": "止损撤退", "impact": {"F": -2, "M": -3, "C": 2}}],
    },
}

EXAMPLE_TRAITS = {"自律": 25, "学习意愿": 30, "行动力": 35, "风险偏好": 40, "责任心": 45, "享乐倾向": 70, "抗压": 45}
EXAMPLE_TAGS = {
    "裸辞全职创业": {"费力": 90, "学习": 70, "风险": 90, "责任": 50, "主动": 90, "爽感": 15},
    "先兼职试水": {"费力": 50, "学习": 55, "风险": 30, "责任": 45, "主动": 65, "爽感": 45},
    "安心打工": {"费力": 35, "学习": 25, "风险": 15, "责任": 45, "主动": 40, "爽感": 65},
    "转全职创业": {"费力": 85, "学习": 65, "风险": 80, "责任": 50, "主动": 85, "爽感": 20},
    "继续兼职攒钱": {"费力": 50, "学习": 45, "风险": 30, "责任": 50, "主动": 60, "爽感": 45},
    "稳扎稳打": {"费力": 65, "学习": 50, "风险": 40, "责任": 55, "主动": 60, "爽感": 35},
    "加速扩张": {"费力": 90, "学习": 60, "风险": 85, "责任": 45, "主动": 90, "爽感": 20},
    "争取晋升": {"费力": 70, "学习": 55, "风险": 40, "责任": 55, "主动": 75, "爽感": 30},
    "躺平摸鱼": {"费力": 5, "学习": 5, "风险": 15, "责任": 15, "主动": 10, "爽感": 85},
    "赶紧上岸": {"费力": 50, "学习": 30, "风险": 25, "责任": 55, "主动": 65, "爽感": 40},
    "边做边等": {"费力": 30, "学习": 20, "风险": 35, "责任": 35, "主动": 40, "爽感": 45},
}


def load_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.example:
        return {"case": EXAMPLE_CASE, "traits": EXAMPLE_TRAITS, "choice_tags": EXAMPLE_TAGS}
    if args.file:
        return json.loads(open(args.file, encoding="utf-8").read())
    if args.json:
        if args.json == "-":
            return json.loads(sys.stdin.read())
        return json.loads(args.json)
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return json.loads(data)
    raise SystemExit("Provide --example, --json, --file, or JSON on stdin.")


def print_text(decision: dict[str, Any], personality: dict[str, Any]) -> None:
    print("=" * 60)
    print("[MCTS 该怎么选]")
    print("=" * 60)
    for row in decision["rankings"]:
        fatal = "、".join(row["fatal_dimensions"]) if row["fatal_dimensions"] else "无致命短板"
        print(f"{row['option']}: 期望 {row['expected']:.1f}/100, 最差5% {row['p5']:.1f}, {fatal}")
        for nxt in row["next_best"]:
            print(f"  若「{nxt['if_outcome']}」-> 下一步最优: {nxt['next_best']}")
    print(f"=> MCTS推荐: {decision['recommendation']}")

    print("\n" + "=" * 60)
    print("[性格推演 他会怎么选]")
    print("=" * 60)
    for row in personality["first_choice_distribution"]:
        print(f"{row['option']}: {row['percent']:.1f}%")
    print("一条典型轨迹:")
    for month, choice, outcome, score in personality["typical_path"][:8]:
        print(f"  第{month}月 选「{choice}」-> {outcome} -> 综合{score}")
    print(
        f"{'最终'}: 平均 {personality['avg_final_score']:.1f}/100, "
        f"最差5% {personality['p5_final_score']:.1f}, "
        f"止损触发率 {personality['stop_loss_rate']:.1f}%"
    )
    print(
        f"\n[最优 vs 现实] 该选:{decision['recommendation']} | "
        f"性格大概率选:{personality['most_likely_first_choice']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", action="store_true", help="Run bundled example")
    parser.add_argument("--json", help="Input JSON payload")
    parser.add_argument("--file", help="Path to input JSON payload")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--months", type=int, default=None)
    parser.add_argument("--sims", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument(
        "--skip-budget",
        metavar="原因",
        help="仅当决策与财务完全无关时使用；原因会写入凭证供用户监督",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="仅供人类开发者调试。AI 助手禁止使用此参数绕过闸门。",
    )
    args = parser.parse_args()

    gate_prov = "example模式(不代表真实案例)"
    if not args.example:
        problems, gate_prov = check_gates(args.skip_budget)
        if problems and not args.force:
            print("闸门未通过，推演拒绝启动：", file=sys.stderr)
            for i, p in enumerate(problems, 1):
                print(f"  {i}. {p}", file=sys.stderr)
            print(
                "\n流程：intake_score.py 达标 → budget_sim.py（或显式 --skip-budget 原因）→ 再来跑推演。"
                "这是强制顺序，不是建议。",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if problems and args.force:
            print("⚠️ WARNING: --force 绕过了未通过的闸门，本次结果不可用于正式建议。", file=sys.stderr)
            gate_prov = "FORCED(闸门被绕过,结果无效)"

    try:
        payload = load_payload(args)
        seed = args.seed if args.seed is not None else int(payload.get("seed", 42))
        random.seed(seed)
        case = payload["case"]
        traits = payload.get("traits", {})
        tags = payload.get("choice_tags", payload.get("tags", {}))
        iters = args.iters if args.iters is not None else int(payload.get("iters", 12000))
        months = args.months if args.months is not None else int(payload.get("months", 12))
        sims = args.sims if args.sims is not None else int(payload.get("sims", 3000))

        engine = LifeMCTS(case)
        decision = engine.decide(iters=iters, verbose=args.format == "text")
        personality = predict_life(case, traits, tags, months=months, sims=sims)
        result = {"mcts": decision, "personality": personality, "seed": seed}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    stamp = (
        f"[凭证] mcts ts={ts} iters={iters} sims={sims} months={months} "
        f"seed={seed} gate=({gate_prov})"
    )
    try:
        state = read_state()
        state["mcts"] = {"ts": ts, "iters": iters, "sims": sims, "months": months, "seed": seed, "gate": gate_prov}
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"WARN: 凭证状态写入失败: {exc}", file=sys.stderr)

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(stamp, file=sys.stderr)
    else:
        print_text(decision, personality)
        print(stamp)


if __name__ == "__main__":
    main()
