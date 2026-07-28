# 人生推演引擎与性格推演

本文件用于 MCTS 最优选择、性格推演、填空表、止损线、觉醒漂移、情绪螺旋、最优 vs 现实差距分析。只有在信息收集达标、预算/外部环境完成后才能读并执行。

## 执行入口

日常执行优先使用 `../scripts/mcts_sim.py`。本文件负责解释推演结构和填空表，不再作为主要执行脚本复制粘贴。若脚本报错，先补齐 `case / traits / choice_tags`，不要绕开脚本手工给结论。

最小运行命令：

```bash
python3 scripts/mcts_sim.py --example --iters 12000 --sims 3000 --months 12
python3 scripts/mcts_sim.py --file case.json --format json --iters 12000 --sims 3000 --months 12
```

### 第三阶段（续）：人生推演引擎 — 算两件事

填空表备好料后，进入**人生推演**。推演回答两个不同的问题，缺一不可：

| 推演 | 回答 | 谁在选 | 用途 |
|------|------|--------|------|
| **MCTS 决策树** | 他**应该**怎么选最好 | 绝对理性的最优大脑 | 给最优建议 |
| **性格推演** | 以他这性子**实际会**怎么选 | 真实的、有惰性有性格的本人 | 预测大概率命运 |

**两者的差距是最有杀伤力的输出**——数据上最优 vs 性格大概率走的路，差距就是要当面点破的"性格税"。

**架构（铁律三）**：下面 `class LifeMCTS` 是死的、通用的，不认识任何具体决定；具体东西全在 `CASE`/`TRAITS`/`CHOICE_TAGS` 填空表里。三个关键机制都做成了填空驱动：**可行性硬筛**（做不到的划掉）、**小步试错的信息增益**（试完看得更清、后面选得更准）、**止损线**（跌破红线自动撤退）。

**这一步必须走代码**。下面保留历史模板和概念实现，便于维护脚本和理解机制；实际执行优先跑 `scripts/mcts_sim.py`。

```python
# -*- coding: utf-8 -*-
# ============================================================
#  人生推演引擎 v3（最终版）
#  铁律三：本段 class LifeMCTS 是"死的"，通用，永不改。
#          所有跟具体人/具体决定有关的东西，全在下面的 CASE/TRAITS/TAGS 里填。
# ============================================================
import random, math
random.seed(42)
def clamp(p): return max(.02, min(.98, p))
def clip(x):  return max(0,  min(100, x))

# ##############################################################
# ##  第一层：通用引擎（死的，永不改）                        ##
# ##############################################################
class Node:
    __slots__=("year","track","N","kids","astat","actions")
    def __init__(self, year, track, eng):
        self.year, self.track, self.N = year, track, 0
        self.kids, self.astat = {}, {}
        self.actions = eng.actions_of(track, year)

class LifeMCTS:
    def __init__(self, case):
        self.c=case; self.DIMS=case["dims"]; self.W=case["weights"]
        self.H=case["horizon"]; self.P0=case["params"]
        self.tracks=case["tracks"]; self.trans=case["transitions"]
        self.noise=case.get("noise",5); self.couples=case.get("couplings",[])
        self.sl=case.get("stop_loss_line"); self.rec=case.get("recovery_track")
    def comp(self,d): return sum(d[k]*self.W[k] for k in self.DIMS)
    def actions_of(self,track,year):
        return [] if year>=self.H else self.tracks.get(track,{}).get("choices",[])
    # 机制A：可行性硬筛——把"根本做不到"的选项划掉（读 hard_limits / requires）
    def feasible(self, verbose=False):
        hard=self.c.get("hard_limits",{}); req=self.c.get("requires",{}); ok=[]
        for opt in self.tracks[self.c["start"]]["choices"]:
            need=req.get(opt,{})
            can=all((hard.get(k,0)>=v) if isinstance(v,(int,float)) else (hard.get(k)==v)
                    for k,v in need.items())
            if can: ok.append(opt)
            elif verbose: print(f"   [可行性筛掉] {opt}  缺条件:{need}（你只有 {hard}）")
        return ok
    def sample(self, action, P):                       # 概率读"当前这条人生的参数"
        outs=self.trans[action]; ps,rest=[],1.0
        for o in outs:
            p=o.get("p"); v=(p(P) if callable(p) else p) if p is not None else None
            ps.append(v); rest-=(v or 0)
        ps=[max(0,rest) if v is None else v for v in ps]
        tot=sum(ps) or 1; ps=[v/tot for v in ps]
        r,acc=random.random(),0
        for o,pp in zip(outs,ps):
            acc+=pp
            if r<=acc: return o
        return outs[-1]
    def step(self, dims, impact):
        d={k: dims[k]+random.gauss(impact.get(k,0),self.noise) for k in self.DIMS}
        for rule in self.couples:                      # 维度联动（有界、单向、防雪崩）
            if d[rule["if"]] < rule["below"]:
                for k,v in rule["then"].items(): d[k]+=random.gauss(v,2)
        return {k: clip(d[k]) for k in self.DIMS}
    def transit(self, dims, o, P):                     # 一步：算新维度 + 机制B信息增益 + 机制C止损
        nd=self.step(dims, o.get("impact",{}))
        nP=dict(P)
        for k,v in o.get("on_outcome",{}).items(): nP[k]=clip(nP.get(k,50)+v)   # 机制B
        to=o["to"]
        sl_fire=False
        if self.sl is not None and self.comp(nd)<self.sl and self.rec:          # 机制C
            to=self.rec; sl_fire=True
        return nd, nP, to, sl_fire
    def pick(self, node):
        for a in node.actions:
            if a not in node.astat: return a
        return max(node.actions, key=lambda a:(node.astat[a][1]/node.astat[a][0]
                   + 1.4*math.sqrt(math.log(node.N)/node.astat[a][0])))
    def run(self, node, dims, P, forced=None):
        if node.year>=self.H: return self.comp(dims)/100, dims
        node.N+=1
        a=forced or self.pick(node)
        o=self.sample(a, P)
        nd,nP,to,_=self.transit(dims, o, P)
        key=(a,o["label"])
        if key not in node.kids: node.kids[key]=Node(node.year+1, to, self)
        r,term=self.run(node.kids[key], nd, nP)
        n,w=node.astat.get(a,(0,0.0)); node.astat[a]=(n+1,w+r)
        return r,term
    def validate(self):
        bad=[a for tr in self.tracks.values() for a in tr.get("choices",[]) if a not in self.trans]
        if bad: print("WARN 缺命运分支:", bad)
        s=sum(self.W.values())
        if abs(s-1)>.01: print(f"WARN 权重和={s:.2f} 应≈1")
    def decide(self, iters=30000):
        self.validate()
        opts=self.feasible(verbose=True)
        root=Node(0,self.c["start"],self); root.actions=opts
        res={o:{"c":[],"d":{k:[] for k in self.DIMS}} for o in opts}
        for i in range(iters):
            a0=opts[i%len(opts)]
            r,term=self.run(root, dict(self.c["init"]), dict(self.P0), forced=a0)
            res[a0]["c"].append(r*100)
            for k in self.DIMS: res[a0]["d"][k].append(term[k])
        nm=self.c["names"]; order=sorted(res, key=lambda o:-sum(res[o]["c"])/len(res[o]["c"]))
        n=len(res[order[0]]["c"])
        print("="*60); print(f"[MCTS 该怎么选] 可行选项{len(opts)}个 · 每个跑{n}次 · 后续步步最优"); print("="*60)
        for o in order:
            c=sorted(res[o]["c"]); avg=sum(c)/n; p5=c[n//20]
            fatal=[nm[k] for k in self.DIMS if sorted(res[o]['d'][k])[n//20]<30]
            tag="致命维度:"+("、".join(fatal)) if fatal else "无致命短板"
            print(f"\n[{o}] 期望 {avg:5.1f}/100  最差5% {p5:5.1f}  {tag}")
            for (a,lb),kid in root.kids.items():
                if a==o and kid.astat:
                    nxt=max(kid.astat,key=lambda x:kid.astat[x][1]/kid.astat[x][0])
                    print(f"    若「{lb}」-> 下一步最优:{nxt}")
        print(f"\n=> MCTS推荐: {order[0]}")
        return order[0]

# ##############################################################
# ##  第二层（性格推演）：他实际会怎么选 + 逐月往下活 + 止损   ##
# ##############################################################
def appeal(c, tr, tg):
    t=tg.get(c,{}); g=lambda n:(t.get(n,50)-50)/50
    return ((tr["自律"]-50)/50*g("费力")+(tr["学习意愿"]-50)/50*g("学习")
           +(tr["风险偏好"]-50)/50*g("风险")+(tr["责任心"]-50)/50*g("责任")
           +(tr["行动力"]-50)/50*g("主动")+(tr["享乐倾向"]-50)/50*g("爽感"))
def personality_pick(choices, tr, tg, temp=1.5):
    ex=[math.exp(appeal(c,tr,tg)/temp) for c in choices]; z=sum(ex)
    r,acc=random.random(),0
    for c,e in zip(choices,ex):
        acc+=e/z
        if r<=acc: return c
    return choices[-1]
def predict_life(case, traits, tags, months=12, sims=3000):
    eng=LifeMCTS(case); opts0=eng.feasible()
    ends={"c":[],"d":{k:[] for k in eng.DIMS}}; first={}; path_by_first={}; sl_hit=0
    # 机制D：觉醒漂移（用户处于"开窍期"时性格持续向好）
    awakening = case.get("awakening", {})  # {"active": True, "rate": 0.5, "max_months": 6}
    # 机制E：情绪螺旋（连续负面→加速下滑，连续正面→信心飞轮）
    emotion = case.get("emotion_profile", {})  # {"rejection_sensitivity": 0.7, "momentum_boost": 0.3}
    for _ in range(sims):
        dims=dict(case["init"]); track=case["start"]; tr=dict(traits); P=dict(case["params"])
        path=[]; triggered=False
        streak_neg = 0; streak_pos = 0  # 情绪螺旋计数器
        for m in range(months):
            choices = opts0 if track==case["start"] else case["tracks"].get(track,{}).get("choices",[])
            if not choices: break
            ch=personality_pick(choices, tr, tags)
            o=eng.sample(ch, P)
            dims,P,track,sl_fire=eng.transit(dims, o, P)
            if sl_fire and not triggered: sl_hit+=1; triggered=True
            path.append((m+1, ch, o["label"], round(eng.comp(dims),1)))
            # 情绪螺旋（机制E）
            comp = eng.comp(dims)
            if comp < 45:
                streak_neg += 1; streak_pos = 0
                decay = 1.0 + streak_neg * emotion.get("rejection_sensitivity", 0.3) * 0.3
                tr["自律"] = max(0, tr["自律"] - decay)
                tr["享乐倾向"] = min(100, tr["享乐倾向"] + decay)
                tr["行动力"] = max(0, tr["行动力"] - decay * 0.5)
            elif comp > 70:
                streak_pos += 1; streak_neg = 0
                boost = 1.0 + streak_pos * emotion.get("momentum_boost", 0.2) * 0.2
                tr["行动力"] = min(100, tr["行动力"] + boost)
                tr["自律"] = min(100, tr["自律"] + boost * 0.3)
            else:
                streak_neg = max(0, streak_neg - 1); streak_pos = max(0, streak_pos - 1)
            # 觉醒漂移（机制D）
            if awakening.get("active") and m < awakening.get("max_months", 6):
                rate = awakening.get("rate", 0.5)
                tr["自律"] = min(100, tr["自律"] + rate)
                tr["学习意愿"] = min(100, tr["学习意愿"] + rate * 0.8)
                tr["行动力"] = min(100, tr["行动力"] + rate * 0.6)
                tr["享乐倾向"] = max(0, tr["享乐倾向"] - rate * 0.4)
            if m==0: first[ch]=first.get(ch,0)+1
        path_by_first.setdefault(path[0][1], path)
        ends["c"].append(eng.comp(dims))
        for k in eng.DIMS: ends["d"][k].append(dims[k])
    N=sum(first.values()); c=sorted(ends["c"]); avg=sum(c)/len(c)
    sample_path=path_by_first[max(first,key=first.get)]
    print("\n"+"="*60); print(f"[性格推演 他会怎么选] 推{months}月 · 跑{sims}遍人生"); print("="*60)
    print("\n第1步他最可能的选择:")
    for ch,k in sorted(first.items(),key=lambda x:-x[1]): print(f"   {ch:<14}{k/N*100:5.1f}%")
    print("\n一条典型轨迹(逐月):")
    for m,ch,o,s in sample_path[:8]: print(f"   第{m:>2}月 选「{ch}」-> {o} -> 综合{s}")
    print(f"\n{months}月后大概落在 综合{avg:.1f}/100 (最差5% {c[len(c)//20]:.1f})")
    print(f"止损线触发率 {sl_hit/sims*100:.1f}% (这些人靠及时撤退避免崩盘)")
    return max(first,key=first.get)

# ##############################################################
# ##  填空表（活的）：示例=王五·8万存款要不要辞职创业        ##
# ##############################################################
CASE = {
 "dims":["F","C","H","M","R","G","S"],
 "names":{"F":"财务","C":"事业","H":"身体","M":"心理","R":"关系","G":"成长","S":"稳定"},
 "weights":{"F":.18,"C":.18,"H":.12,"M":.16,"R":.14,"G":.12,"S":.10},
 "params":{"能力弹药库":70,"赛道景气度":68,"方向清晰度":45,"安全垫厚度":40},
 "hard_limits":{"现金":80000},
 "requires":{"裸辞全职创业":{"现金":150000}},
 "start":"现状",
 "init":{"F":48,"C":62,"H":70,"M":55,"R":62,"G":58,"S":65},
 "horizon":3, "noise":5,
 "couplings":[{"if":"F","below":35,"then":{"M":-6,"R":-4}},
              {"if":"R","below":35,"then":{"M":-6,"H":-3}}],
 "stop_loss_line":30, "recovery_track":"止损撤退",
 "tracks":{
   "现状":   {"choices":["裸辞全职创业","先兼职试水","安心打工"]},
   "试过之后":{"choices":["转全职创业","继续兼职攒钱","安心打工"]},
   "创业中": {"choices":["稳扎稳打","加速扩张"]},
   "打工中": {"choices":["争取晋升","躺平摸鱼"]},
   "止损撤退":{"choices":["赶紧上岸","边做边等"]},
 },
 "transitions":{
   "裸辞全职创业":[{"label":"硬撑","p":1.0,"to":"创业中","impact":{"F":-20,"C":10,"G":15,"S":-20,"M":-10}}],
   "先兼职试水":[
     {"label":"试出来适合","p":0.5,"to":"试过之后","impact":{"F":3,"C":6,"G":10,"M":4,"S":-2},
      "on_outcome":{"方向清晰度":22,"安全垫厚度":6}},
     {"label":"试出来不合适","p":None,"to":"试过之后","impact":{"F":1,"C":2,"G":6,"M":-3,"S":-1},
      "on_outcome":{"方向清晰度":15}}],
   "安心打工":[{"label":"平稳","p":1.0,"to":"打工中","impact":{"F":5,"C":4,"M":3,"S":8,"R":3}}],
   "转全职创业":[
     {"label":"做成了","p":lambda P:clamp(.28+(P["方向清晰度"]-50)/130),"to":"创业中",
      "impact":{"F":25,"C":28,"G":25,"S":-10,"M":8,"H":-8}},
     {"label":"没做成","p":None,"to":"止损撤退",
      "impact":{"F":-25,"C":5,"G":18,"S":-25,"M":-22,"R":-12}}],
   "继续兼职攒钱":[{"label":"稳步积累","p":1.0,"to":"试过之后","impact":{"F":8,"C":5,"G":6,"S":4,"M":2}}],
   "稳扎稳打":[{"label":"小步前进","p":lambda P:clamp(.55),"to":"创业中","impact":{"F":10,"C":12,"G":10,"M":5}},
              {"label":"遇冷","p":None,"to":"止损撤退","impact":{"F":-15,"M":-12,"S":-10}}],
   "加速扩张":[{"label":"起飞","p":lambda P:clamp(.30+(P["赛道景气度"]-60)/200),"to":"创业中",
               "impact":{"F":30,"C":25,"G":18,"H":-12,"M":6}},
              {"label":"扩崩了","p":None,"to":"止损撤退","impact":{"F":-35,"M":-25,"S":-30,"R":-10}}],
   "争取晋升":[{"label":"升了","p":lambda P:clamp(.35+(P["能力弹药库"]-60)/200),"to":"打工中",
               "impact":{"F":18,"C":20,"M":12,"S":5,"H":-6}},
              {"label":"没升","p":None,"to":"打工中","impact":{"F":2,"M":-8,"C":2}}],
   "躺平摸鱼":[{"label":"舒服但荒废","p":1.0,"to":"打工中","impact":{"F":-3,"C":-8,"G":-8,"M":2,"S":3}}],
   "赶紧上岸":[{"label":"找到稳定工作","p":lambda P:clamp(.6+(P["能力弹药库"]-60)/200),"to":"打工中",
               "impact":{"F":12,"M":15,"S":18,"C":3,"R":6}},
              {"label":"暂时没着落","p":None,"to":"止损撤退","impact":{"F":-8,"M":-10,"S":-8}}],
   "边做边等":[{"label":"骑驴找马","p":1.0,"to":"止损撤退","impact":{"F":-2,"M":-3,"C":2}}],
 },
}
TRAITS={"自律":25,"学习意愿":30,"行动力":35,"风险偏好":40,"责任心":45,"享乐倾向":70,"抗压":45}
CHOICE_TAGS={
  "裸辞全职创业":{"费力":90,"学习":70,"风险":90,"责任":50,"主动":90,"爽感":15},
  "先兼职试水":{"费力":50,"学习":55,"风险":30,"责任":45,"主动":65,"爽感":45},
  "安心打工":{"费力":35,"学习":25,"风险":15,"责任":45,"主动":40,"爽感":65},
  "转全职创业":{"费力":85,"学习":65,"风险":80,"责任":50,"主动":85,"爽感":20},
  "继续兼职攒钱":{"费力":50,"学习":45,"风险":30,"责任":50,"主动":60,"爽感":45},
  "稳扎稳打":{"费力":65,"学习":50,"风险":40,"责任":55,"主动":60,"爽感":35},
  "加速扩张":{"费力":90,"学习":60,"风险":85,"责任":45,"主动":90,"爽感":20},
  "争取晋升":{"费力":70,"学习":55,"风险":40,"责任":55,"主动":75,"爽感":30},
  "躺平摸鱼":{"费力":5,"学习":5,"风险":15,"责任":15,"主动":10,"爽感":85},
  "赶紧上岸":{"费力":50,"学习":30,"风险":25,"责任":55,"主动":65,"爽感":40},
  "边做边等":{"费力":30,"学习":20,"风险":35,"责任":35,"主动":40,"爽感":45},
}

if __name__ == "__main__":
    print("#"*60); print("# 王五：8万存款，要不要辞职创业？"); print("#"*60)
    best = LifeMCTS(CASE).decide(iters=12000)
    pred = predict_life(CASE, TRAITS, CHOICE_TAGS, months=12, sims=3000)
    print("\n"+"="*60)
    print(f"[最优 vs 现实]  该选:{best}   |   性格大概率选:{pred}")
    print("一致 → 放心建议" if best==pred else "有差距 → 要点破'性格税'")
    print("="*60)
```

**⚠ 填空表怎么填（决定准确率的关键；引擎死、全靠填得准）**

| 要填的空 | 怎么填 | 数据来源 |
|---------|--------|---------|
| `weights` 价值观权重 | 谁重要数字大，加起来=1 | 第二阶段「价值观」 |
| `params` 核心参数 | 能力、安全垫、方向清晰度等 | 第三阶段聚合 + 个人画像 |
| `init` 起跑线 | 现在 7 个方面各打几分 | 个人现状 |
| `hard_limits` / `requires` | 用户硬条件(现金/签证/学历) + 每个选项的门槛 | 个人现状（可行性筛用） |
| `tracks` 各轨道的选择 | "走到这一步还能干嘛"，几个选项都行，会一层层长出来 | 大模型推理 |
| `transitions` 概率 `p` | **必须挂钩参数/真实数据** | 第三阶段搜来的数据 |
| `transitions` 7维冲击 | 这事发生各维度涨跌多少 | 大模型按常识+数据估 |
| `transitions` `on_outcome` | 小步试错试完更新哪个参数（信息增益） | 大模型按常识填 |
| `stop_loss_line`/`recovery_track` | 综合分跌破多少触发止损、撤到哪条轨道 | 和用户商定 |
| `awakening` 觉醒漂移 | 用户是否处于"开窍期"：active/rate/max_months | 第一阶段D6「从中学到什么」+D8「改变的种子」 |
| `emotion_profile` 情绪螺旋 | 被拒敏感度(0-1)、正面动量加成(0-1) | 第一阶段D3性格+D5心理状态 |
| `couplings` | 维度联动（某维崩了连带拖累谁），要有界防雪崩 | 大模型按常识 |
| `TRAITS` 性格画像 | 自律/学习意愿/享乐倾向等打分 | 第一阶段 D3「性格特征」 |
| `CHOICE_TAGS` 选择标签 | 每个选项要多少费力/学习/风险，给多少爽感 | 大模型按常识填 |

**⚠ 大模型使用推演结果的硬规则：**

1. **必须引用具体数字**：不说"风险较大"，而说"心理维度最差 5% 只有 28 分"。
2. **必须把"最优"和"现实"并排给用户看**：MCTS 推荐什么 vs 性格推演他大概会选什么，差距（性格税）要点破。一致才放心建议；不一致要警告"按你的性格大概率会偏到 X，代价是 Y"。
3. **可行性先筛**：`feasible()` 把硬条件不够的选项划掉，别分析做不到的选项。
4. **小步试错永远摆上桌**：只要可能，就在 `tracks` 里加一个低成本"先试试/再等等"的选项（用 `on_outcome` 体现"试完更看得清"），并在建议里主动提它。
5. **止损线 + 复盘点**：用 `stop_loss_line`/`recovery_track` 让模拟自带"及时撤退"；给用户的建议也要带"出现什么信号就撤""几个月后回来复盘"。
6. **黑天鹅走压力测试、不塞概率**：疫情/战争/政策突变这类不可预测的，不给概率，单独问"它来了这个选项扛不扛得住"。
7. **概率标来源 + 敏感性分析**：每个 `p` 是搜来的还是估的；估的把它 ±10% 重跑，若结论翻转就告诉用户"这条结论很敏感，重点确认它"。
8. **必须识别"致命维度"**：任一维度最差 5% 低于 30 分，警告"有小概率让你在 X 方面崩盘"。
9. **结合价值观调权重**：用户说"家庭第一"就调高 R 权重重算。研究表明人普遍低估关系、高估稳定，可据此反问，但不替用户改价值观。
10. **那个分数不是真理**：综合分背后全是估的，给用户看时要说清"这是带误差的参考，不是命运判决"。
11. **决定权永远还给用户**。
