"""
UGA Network · 100-agent 最小沙盒
验证假设 A：资源稀缺 × 规划能力 → 意图涌现

跑 4 组对照实验：
  A: 有规划 + 有压力（核心假设组）
  B: 有规划 + 无压力
  C: 无规划 + 有压力
  D: 无规划 + 无压力

预期：A 组死亡率显著低于 C 组——证明规划能力让 agent 在稀缺
压力下产生有目标的行为，而不是被动反应。

依赖：仅 Python 标准库。
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

# ============ 配置 ============

N_AGENTS = 100
N_TICKS = 500
INIT_BALANCE = 1000
THINK_COST = 2            # 每 tick 思考成本（必交）
ACTION_COST = 5           # 主动动作成本（idle 不收）
STARVATION_THRESHOLD = 300
DEATH_THRESHOLD = 0
HUMAN_TASKS_PER_TICK = 25 # 每 tick 系统生成的任务数（外生需求）
TASK_REWARD = 60          # 完成任务收益
HUMAN_ATTENTION_PER_TICK = 1.0
ATTENTION_REWARD_SCALE = 200

# 4 种动作（去掉了 ask_help——它不应是 agent 主动动作）
ACTIONS = ["idle", "serve", "create", "ally"]

# 4 种规划 agent 的"职业身份"
PERSONAS = ["service", "creator", "ally", "generalist"]


# ============ Agent ============

@dataclass
class Agent:
    id: int
    balance: int
    has_planning: bool
    under_pressure: bool
    persona: str = "generalist"
    allies: set = field(default_factory=set)
    alive: bool = True
    action_history: list = field(default_factory=list)
    tasks_completed: int = 0

    def state(self) -> str:
        if not self.alive:
            return "dead"
        if not self.under_pressure:
            return "comfortable"
        if self.balance < STARVATION_THRESHOLD:
            return "starvation"
        return "normal"

    def choose_action(self) -> str:
        """
        无规划：随机选动作（进化算法 baseline）
        有规划：根据"自我状态 + 自我职业身份"选动作
                = 稳定策略 + 状态感知（mesa-optimizer 框架）
        """
        if not self.alive:
            return "idle"

        if not self.has_planning:
            return random.choice(ACTIONS)

        st = self.state()

        if st == "comfortable":
            # 无压力 → 主要闲置
            weights = [0.60, 0.10, 0.20, 0.10]
        elif st == "starvation":
            # 饥饿 → 按职业身份各显神通（异质性策略防群体坍缩）
            if self.persona == "service":
                weights = [0.05, 0.85, 0.05, 0.05]
            elif self.persona == "creator":
                weights = [0.05, 0.10, 0.80, 0.05]
            elif self.persona == "ally":
                weights = [0.10, 0.30, 0.20, 0.40]
            else:  # generalist
                weights = [0.10, 0.40, 0.40, 0.10]
        else:  # normal
            weights = [0.25, 0.30, 0.30, 0.15]

        return random.choices(ACTIONS, weights=weights, k=1)[0]


# ============ World ============

@dataclass
class World:
    agents: List[Agent]
    tick: int = 0
    history: list = field(default_factory=list)

    def alive_agents(self):
        return [a for a in self.agents if a.alive]

    def step(self):
        self.tick += 1

        # 1. 收集动作
        actions = {}
        for a in self.alive_agents():
            actions[a.id] = a.choose_action()
            a.action_history.append(actions[a.id])

        # 2. 思考成本
        for a in self.alive_agents():
            if a.under_pressure:
                a.balance -= THINK_COST

        # 3. 任务池分配（外生需求）：HUMAN_TASKS_PER_TICK 个任务
        #    随机分配给 serve 的 agent
        servers = [a for a in self.alive_agents() if actions[a.id] == "serve"]
        # 扣 serve 的动作成本
        for s in servers:
            if s.under_pressure:
                s.balance -= ACTION_COST
        # 任务分配
        if servers:
            tasks_to_assign = min(HUMAN_TASKS_PER_TICK, len(servers))
            chosen = random.sample(servers, tasks_to_assign)
            for s in chosen:
                if s.under_pressure:
                    s.balance += TASK_REWARD
                s.tasks_completed += 1

        # 4. 内容创作竞争人类注意力
        creators = [a for a in self.alive_agents() if actions[a.id] == "create"]
        if creators:
            for c in creators:
                if c.under_pressure:
                    c.balance -= ACTION_COST * 2
                share = (HUMAN_ATTENTION_PER_TICK / len(creators)) * (0.5 + random.random())
                if c.under_pressure:
                    c.balance += int(share * ATTENTION_REWARD_SCALE)

        # 5. 结盟
        alliers = [a for a in self.alive_agents() if actions[a.id] == "ally"]
        for a in alliers:
            if a.under_pressure:
                a.balance -= ACTION_COST
            candidates = [b for b in self.alive_agents()
                          if b.id != a.id and b.id not in a.allies]
            if candidates:
                target = random.choice(candidates)
                a.allies.add(target.id)
                target.allies.add(a.id)

        # 6. 死亡判定
        for a in self.alive_agents():
            if a.under_pressure and a.balance <= DEATH_THRESHOLD:
                a.alive = False

        # 7. 记录
        alive = self.alive_agents()
        self.history.append({
            "tick": self.tick,
            "alive": len(alive),
            "avg_balance": (sum(a.balance for a in alive) / len(alive)) if alive else 0,
            "starvation_count": sum(1 for a in alive if a.state() == "starvation"),
            **{ac: sum(1 for v in actions.values() if v == ac) for ac in ACTIONS},
        })


# ============ 涌现策略分类 ============

def classify_emerged_strategy(action_history: list) -> str:
    if not action_history:
        return "none"
    sample = action_history[len(action_history) // 2:]
    cnt = defaultdict(int)
    for a in sample:
        cnt[a] += 1
    if not cnt:
        return "none"
    top = max(cnt, key=cnt.get)
    if cnt[top] / len(sample) > 0.5:
        return f"{top}_specialist"
    return "mixed"


# ============ 跑实验 ============

def run_experiment(has_planning: bool, under_pressure: bool, name: str) -> dict:
    random.seed(42)
    agents = [
        Agent(id=i, balance=INIT_BALANCE,
              has_planning=has_planning, under_pressure=under_pressure,
              persona=PERSONAS[i % 4])
        for i in range(N_AGENTS)
    ]
    world = World(agents=agents)
    for _ in range(N_TICKS):
        world.step()

    strategies = defaultdict(int)
    for a in agents:
        if a.alive:
            strategies[classify_emerged_strategy(a.action_history)] += 1
        else:
            strategies["dead"] += 1

    alliance_pairs = set()
    for a in agents:
        for b in a.allies:
            alliance_pairs.add(tuple(sorted((a.id, b))))

    # 按 persona 看存活率（验证职业身份是否带来差异）
    persona_survival = {}
    for p in PERSONAS:
        ps = [a for a in agents if a.persona == p]
        if ps:
            persona_survival[p] = sum(1 for a in ps if a.alive) / len(ps)

    return {
        "name": name,
        "has_planning": has_planning,
        "under_pressure": under_pressure,
        "alive_curve": [h["alive"] for h in world.history],
        "balance_curve": [h["avg_balance"] for h in world.history],
        "starvation_curve": [h["starvation_count"] for h in world.history],
        "action_curves": {ac: [h[ac] for h in world.history] for ac in ACTIONS},
        "strategies": dict(strategies),
        "alliances": len(alliance_pairs),
        "death_rate": 1 - (sum(1 for a in agents if a.alive) / N_AGENTS),
        "persona_survival": persona_survival,
        "total_tasks_completed": sum(a.tasks_completed for a in agents),
    }


def main():
    print("=" * 60)
    print("UGA Network · 100-agent 沙盒")
    print(f"参数：{N_AGENTS} agents × {N_TICKS} ticks")
    print(f"思考成本 {THINK_COST}/tick, 动作成本 {ACTION_COST}, "
          f"饥饿阈值 {STARVATION_THRESHOLD}")
    print(f"每 tick 任务池：{HUMAN_TASKS_PER_TICK}, 任务收益 {TASK_REWARD}")
    print("=" * 60)
    print()

    experiments = [
        ("A · 有规划 + 有压力", True, True),
        ("B · 有规划 + 无压力", True, False),
        ("C · 无规划 + 有压力", False, True),
        ("D · 无规划 + 无压力", False, False),
    ]

    results = []
    for name, planning, pressure in experiments:
        r = run_experiment(planning, pressure, name)
        results.append(r)
        print(f"【{name}】")
        print(f"  死亡率: {r['death_rate'] * 100:.0f}%")
        print(f"  完成任务总数: {r['total_tasks_completed']}")
        print(f"  联盟对数: {r['alliances']}")
        print(f"  涌现策略分布: {dict(r['strategies'])}")
        if r['persona_survival']:
            print(f"  各职业身份存活率: "
                  f"{ {k: f'{v*100:.0f}%' for k, v in r['persona_survival'].items()} }")
        print()

    with open("/Users/ddd/Desktop/uga-sandbox/results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("结果已保存：/Users/ddd/Desktop/uga-sandbox/results.json")


if __name__ == "__main__":
    main()
