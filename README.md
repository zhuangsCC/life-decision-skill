# Life Decision Skill

一个面向职业、城市、创业、转行、备考、关系和重大消费选择的决策辅助 Skill。它强调：先理解现实约束，再做可复核的分析；不把分数或推演结果当作命运判决。

## 特点

- 先做安全风险与可逆性分诊。
- 用信息门槛防止“资料不够就硬给建议”。
- 涉及金钱、工作、城市、创业、转行或学习投入时，强制三档预算压力测试。
- 用脚本保存可核验的流程凭证；MCTS 推演不会跳过前置检查。
- 同时输出“理性最优”和“按真实行为模式更可能发生的路径”，并给出止损线与复盘点。

## 安装

将整个目录放入你的 Agent skills 目录，并按宿主工具的 Skill 加载方式启用。核心说明在 [SKILL.md](SKILL.md)。

运行时仅需要 Python 3 标准库：

```bash
python3 scripts/intake_score.py --help
python3 scripts/budget_sim.py --help
python3 scripts/mcts_sim.py --help
```

## 快速自检

```bash
python3 -m py_compile scripts/*.py
python3 scripts/mcts_sim.py --example --iters 200 --sims 50 --months 3 --format json
```

`--example` 是合成演示，不代表真实案例。真实决策请遵守 `SKILL.md` 的信息收集、预算与安全检查流程。

## 目录

- `SKILL.md`：流程、硬约束和 Agent 使用说明。
- `references/`：信息收集、问卷、预算、MCTS 与输出模板。
- `scripts/`：信息评分、预算模拟和推演引擎。

## 边界

本项目用于辅助思考，不替代医疗、法律、心理治疗、财务或职业资格意见。遇到自伤、自杀、暴力、胁迫或诈骗风险时，应优先寻求现实世界的紧急支持与专业帮助。

## License

[MIT](LICENSE)
