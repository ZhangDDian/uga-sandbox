"""
UGA Network · 演化沙盒 v3 — 生态竞争版

设计：
  一个世界里同时放 planning agent 和 evolving agent，争同一个任务池。
  planning 不是免费的——有代谢成本（PLAN_COST/tick）。
  plan_gene 是连续可遗传性状 [0,1]，> 0.5 表达为 planner 表型。
  两个环境对照：高压（任务稀缺）vs 低压（任务充裕）。

  核心检验：
    高压世界 → planner 频率上升（规划在压力下有竞争优势）
    低压世界 → planner 频率下降或持平（压力是规划价值的前提）

  如果两个世界的 planner 频率走势有显著分化，
  则支持假设 A："资源稀缺 × 规划能力 → 意图涌现"。

仅用 Python 标准库。
"""

import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import List

# ============ 超参数 ============

N_INIT_AGENTS = 100
POP_CAP = 200
N_TICKS = 2000
SHOCK_TICK = 1000

INIT_BALANCE = 500
THINK_COST = 2
ACTION_COST = 5
PLAN_COST = 4          # planner 每 tick 额外代谢成本
DEATH_THRESHOLD = 0
REPRO_THRESHOLD = 1200
REPRO_COST_FRAC = 0.5

TASK_REWARD = 60
ATTN_PER_TICK = 1.0
EPS_EXPLORE = 0.05
MUTATION_SIGMA = 0.12
PLAN_GENE_SIGMA = 0.08

ACTIONS = ["idle", "serve", "create", "ally"]
N_ACTIONS = len(ACTIONS)
STATES = ["normal", "starving"]
STARVING_THRESHOLD = 200

# ============ Agent ============

@dataclass
class Agent:
    id: int
    balance: float
    plan_gene: float            # [0,1]，>0.5 表达为 planner
    policy: list                # [normal_logits(4), starving_logits(4)] 展平为 8 floats
    alive: bool = True
    birth_tick: int = 0
    parent_id: int = -1
    tasks_completed: int = 0
    # 增量计数器代替全量 history
    normal_actions: list = field(default_factory=lambda: [0, 0, 0, 0])
    starving_actions: list = field(default_factory=lambda: [0, 0, 0, 0])

    @property
    def is_planner(self) -> bool:
        return self.plan_gene > 0.5

    def state(self) -> str:
        return "starving" if self.balance < STARVING_THRESHOLD else "normal"

    def choose_by_policy(self) -> str:
        st = self.state()
        offset = 0 if st == "normal" else N_ACTIONS
        logits = self.policy[offset:offset + N_ACTIONS]
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        s = sum(exps)
        probs = [e / s for e in exps]
        return random.choices(ACTIONS, weights=probs, k=1)[0]


def random_policy() -> list:
    return [random.gauss(0, 0.5) for _ in range(N_ACTIONS * len(STATES))]


def mutate_policy(policy: list, sigma: float) -> list:
    return [v + random.gauss(0, sigma) for v in policy]


def mutate_plan_gene(gene: float, sigma: float) -> float:
    return max(0.0, min(1.0, gene + random.gauss(0, sigma)))


# ============ 1-step lookahead ============

def estimate_reward(action: str, obs: dict) -> float:
    n_servers = max(1, obs["n_servers"])
    n_creators = max(1, obs["n_creators"])
    tasks = obs["tasks"]
    attn_scale = obs["attn_scale"]

    if action == "idle":
        return -THINK_COST
    if action == "serve":
        p_get_task = min(tasks / (n_servers + 1), 1.0)
        return p_get_task * TASK_REWARD - ACTION_COST - THINK_COST
    if action == "create":
        share = ATTN_PER_TICK / (n_creators + 1)
        return share * attn_scale - 2 * ACTION_COST - THINK_COST
    if action == "ally":
        return -ACTION_COST - THINK_COST
    return 0.0


def planner_choose(obs: dict) -> str:
    if random.random() < EPS_EXPLORE:
        return random.choice(ACTIONS)
    rewards = [estimate_reward(a, obs) for a in ACTIONS]
    best = max(range(N_ACTIONS), key=lambda i: rewards[i])
    return ACTIONS[best]


# ============ World ============

@dataclass
class World:
    agents: List[Agent]
    tick: int = 0
    next_id: int = 0
    obs: dict = field(default_factory=lambda: {
        "n_servers": 5, "n_creators": 5, "tasks": 12, "attn_scale": 200,
    })
    # 环境配置
    tasks_mean: float = 12
    tasks_std: float = 3
    tasks_mean_post: float = 6
    attn_scale_pre: float = 200
    attn_scale_post: float = 400
    # 时间序列
    pop_curve: list = field(default_factory=list)
    planner_frac_curve: list = field(default_factory=list)
    avg_balance_curve: list = field(default_factory=list)

    def step(self):
        self.tick += 1
        # 环境参数
        if self.tick < SHOCK_TICK:
            tasks = max(0, int(random.gauss(self.tasks_mean, self.tasks_std)))
            attn_scale = self.attn_scale_pre
        else:
            tasks = max(0, int(random.gauss(self.tasks_mean_post, self.tasks_std)))
            attn_scale = self.attn_scale_post

        alive = [a for a in self.agents if a.alive]
        if not alive:
            self._record(0, 0, 0)
            return

        n_alive = len(alive)

        # 1. 选动作 + 记录
        actions = {}
        for a in alive:
            if a.is_planner:
                act = planner_choose(self.obs)
                a.balance -= PLAN_COST  # 规划代谢
            else:
                act = a.choose_by_policy()
            actions[a.id] = act
            # 增量计数
            idx = ACTIONS.index(act)
            if a.state() == "normal":
                a.normal_actions[idx] += 1
            else:
                a.starving_actions[idx] += 1

        # 2. 思考成本
        for a in alive:
            a.balance -= THINK_COST

        # 3. serve 结算
        servers = [a for a in alive if actions[a.id] == "serve"]
        for s in servers:
            s.balance -= ACTION_COST
        if servers and tasks > 0:
            n_assigned = min(tasks, len(servers))
            chosen = random.sample(servers, n_assigned)
            for s in chosen:
                s.balance += TASK_REWARD
                s.tasks_completed += 1

        # 4. create 结算
        creators = [a for a in alive if actions[a.id] == "create"]
        for c in creators:
            c.balance -= 2 * ACTION_COST
        if creators:
            for c in creators:
                share = (ATTN_PER_TICK / len(creators)) * (0.5 + random.random())
                c.balance += share * attn_scale

        # 5. ally — 社交投资：无即时回报但有概率产出合作红利
        allyers = [a for a in alive if actions[a.id] == "ally"]
        for a in allyers:
            a.balance -= ACTION_COST
        # 两两配对的 allyers 有概率获得合作红利（模拟互利）
        if len(allyers) >= 2:
            random.shuffle(allyers)
            for i in range(0, len(allyers) - 1, 2):
                if random.random() < 0.3:
                    bonus = TASK_REWARD * 0.5
                    allyers[i].balance += bonus
                    allyers[i + 1].balance += bonus

        # 6. 死亡
        for a in alive:
            if a.balance <= DEATH_THRESHOLD:
                a.alive = False

        # 7. 繁殖（性能修复：只遍历一次）
        new_agents = []
        alive_count = sum(1 for a in self.agents if a.alive)
        for a in self.agents:
            if not a.alive:
                continue
            if alive_count >= POP_CAP:
                break
            if a.balance > REPRO_THRESHOLD:
                child_balance = a.balance * REPRO_COST_FRAC
                a.balance -= child_balance
                child = Agent(
                    id=self.next_id,
                    balance=child_balance,
                    plan_gene=mutate_plan_gene(a.plan_gene, PLAN_GENE_SIGMA),
                    policy=mutate_policy(a.policy, MUTATION_SIGMA),
                    birth_tick=self.tick,
                    parent_id=a.id,
                )
                self.next_id += 1
                new_agents.append(child)
                alive_count += 1
        self.agents.extend(new_agents)

        # 8. 更新观察 + 记录
        self.obs = {
            "n_servers": len(servers),
            "n_creators": len(creators),
            "tasks": tasks,
            "attn_scale": attn_scale,
        }
        alive_now = [a for a in self.agents if a.alive]
        n_now = len(alive_now)
        planners = sum(1 for a in alive_now if a.is_planner)
        self._record(n_now, planners / n_now if n_now else 0,
                     sum(a.balance for a in alive_now) / n_now if n_now else 0)

    def _record(self, pop, planner_frac, avg_bal):
        self.pop_curve.append(pop)
        self.planner_frac_curve.append(planner_frac)
        self.avg_balance_curve.append(avg_bal)


# ============ 度量 ============

def intent_kl(agent: Agent) -> float:
    """KL( starving 动作分布 ‖ normal 动作分布 )"""
    n_n = sum(agent.normal_actions)
    n_s = sum(agent.starving_actions)
    if n_n < 10 or n_s < 10:
        return None
    eps = 1e-9
    p = [x / n_n + eps for x in agent.normal_actions]
    q = [x / n_s + eps for x in agent.starving_actions]
    return sum(q[i] * math.log(q[i] / p[i]) for i in range(N_ACTIONS))


# ============ 实验 ============

def run_one(pressure: str, seed: int) -> dict:
    random.seed(seed)

    if pressure == "high":
        cfg = dict(tasks_mean=12, tasks_std=3, tasks_mean_post=6,
                   attn_scale_pre=200, attn_scale_post=400)
    else:
        cfg = dict(tasks_mean=50, tasks_std=5, tasks_mean_post=50,
                   attn_scale_pre=200, attn_scale_post=200)

    # 初始种群：50% planner（gene≈0.8），50% evolver（gene≈0.2）
    agents = []
    for i in range(N_INIT_AGENTS):
        if i < N_INIT_AGENTS // 2:
            gene = 0.8 + random.gauss(0, 0.05)
        else:
            gene = 0.2 + random.gauss(0, 0.05)
        gene = max(0.0, min(1.0, gene))
        agents.append(Agent(
            id=i, balance=INIT_BALANCE, plan_gene=gene, policy=random_policy()
        ))

    world = World(agents=agents, next_id=N_INIT_AGENTS, **cfg)
    for _ in range(N_TICKS):
        world.step()
        if not any(a.alive for a in world.agents):
            break

    alive = [a for a in world.agents if a.alive]
    total_ever = len(world.agents)
    n_alive = len(alive)

    planners_alive = [a for a in alive if a.is_planner]
    evolvers_alive = [a for a in alive if not a.is_planner]

    # planner 频率时间序列的关键点
    curve = world.planner_frac_curve
    init_frac = statistics.mean(curve[:50]) if len(curve) >= 50 else 0.5
    pre_shock = statistics.mean(curve[SHOCK_TICK-50:SHOCK_TICK]) if len(curve) >= SHOCK_TICK else init_frac
    final_frac = statistics.mean(curve[-50:]) if len(curve) >= 50 else 0.0

    # 意图 KL
    planner_kls = [k for a in planners_alive if (k := intent_kl(a)) is not None]
    evolver_kls = [k for a in evolvers_alive if (k := intent_kl(a)) is not None]

    return {
        "final_pop": n_alive,
        "total_ever": total_ever,
        "planner_frac_init": init_frac,
        "planner_frac_pre_shock": pre_shock,
        "planner_frac_final": final_frac,
        "planner_frac_delta": final_frac - init_frac,
        "planner_alive": len(planners_alive),
        "evolver_alive": len(evolvers_alive),
        "planner_intent_kl": statistics.mean(planner_kls) if planner_kls else 0.0,
        "evolver_intent_kl": statistics.mean(evolver_kls) if evolver_kls else 0.0,
        "planner_frac_curve": curve,
    }


def welch_t(a: list, b: list) -> tuple:
    if len(a) < 2 or len(b) < 2:
        return 0.0, 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    denom = va/na + vb/nb
    if denom == 0:
        return 0.0, 0.0
    se = math.sqrt(denom)
    t = (ma - mb) / se
    df = denom**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1)) if denom > 0 else 1
    return t, df


def sig_label(t):
    at = abs(t)
    if at > 3.5: return "***"
    if at > 2.5: return "**"
    if at > 1.7: return "*"
    return "ns"


def main():
    N_SEEDS = 20
    print("=" * 72)
    print("UGA Network · 演化沙盒 v3 — 生态竞争版")
    print(f"  pop≤{POP_CAP}  ticks={N_TICKS}  shock@{SHOCK_TICK}  seeds={N_SEEDS}")
    print(f"  初始种群: 50 planners (gene≈0.8) + 50 evolvers (gene≈0.2)")
    print(f"  规划代谢成本: {PLAN_COST}/tick")
    print(f"  高压: tasks~N(12,3) → shock后 N(6,3), attn 200→400")
    print(f"  低压: tasks~N(50,5) 全程不变")
    print("=" * 72)

    results = {}
    for pressure in ["high", "low"]:
        print(f"\n{'─'*72}")
        print(f"  环境: {pressure} pressure")
        print(f"{'─'*72}")
        runs = []
        for seed in range(N_SEEDS):
            r = run_one(pressure, seed)
            runs.append(r)
            # 进度
            if (seed + 1) % 5 == 0:
                print(f"  ... {seed+1}/{N_SEEDS} seeds done")

        results[pressure] = runs

        # 汇总
        def ms(key):
            vs = [r[key] for r in runs]
            return statistics.mean(vs), statistics.stdev(vs) if len(vs) > 1 else 0.0

        m, s = ms("planner_frac_final")
        print(f"\n  Planner 最终占比:   {m:.3f} ± {s:.3f}")
        m, s = ms("planner_frac_delta")
        print(f"  Planner 占比变化:   {m:+.3f} ± {s:.3f}  (正=planner扩张)")
        m, s = ms("final_pop")
        print(f"  最终存活人口:       {m:.1f} ± {s:.1f}")
        m, s = ms("planner_alive")
        print(f"    其中 planners:    {m:.1f} ± {s:.1f}")
        m, s = ms("evolver_alive")
        print(f"    其中 evolvers:    {m:.1f} ± {s:.1f}")
        m, s = ms("planner_intent_kl")
        print(f"  Planner 意图KL:     {m:.4f} ± {s:.4f}")
        m, s = ms("evolver_intent_kl")
        print(f"  Evolver 意图KL:     {m:.4f} ± {s:.4f}")

    # ============ 跨环境对比 ============
    print(f"\n{'='*72}")
    print("假设检验")
    print(f"{'='*72}")

    # 核心检验：高压下 planner 扩张 vs 低压下 planner 萎缩
    high_deltas = [r["planner_frac_delta"] for r in results["high"]]
    low_deltas = [r["planner_frac_delta"] for r in results["low"]]
    t, df = welch_t(high_deltas, low_deltas)
    mh, ml = statistics.mean(high_deltas), statistics.mean(low_deltas)
    print(f"\n  1) Planner 占比变化: 高压 vs 低压")
    print(f"     高压 Δ={mh:+.3f}   低压 Δ={ml:+.3f}")
    print(f"     差异={mh-ml:+.3f}  t={t:+.2f}  df={df:.1f}  [{sig_label(t)}]")
    if mh > ml and abs(t) > 2:
        print(f"     → 支持假设A: 高压下规划能力有选择优势")
    elif abs(t) <= 2:
        print(f"     → 无显著差异，假设A 未获支持")
    else:
        print(f"     → 方向相反，假设A 被否定")

    # 意图 KL：planner vs evolver（高压环境内）
    hp_kl = [r["planner_intent_kl"] for r in results["high"]]
    he_kl = [r["evolver_intent_kl"] for r in results["high"]]
    t2, df2 = welch_t(hp_kl, he_kl)
    print(f"\n  2) 高压环境内: Planner 意图KL vs Evolver 意图KL")
    print(f"     Planner={statistics.mean(hp_kl):.4f}  Evolver={statistics.mean(he_kl):.4f}")
    print(f"     t={t2:+.2f}  df={df2:.1f}  [{sig_label(t2)}]")

    # 补充：高压 planner 最终占比是否 > 0.5
    high_final = [r["planner_frac_final"] for r in results["high"]]
    mean_hf = statistics.mean(high_final)
    # one-sample t vs 0.5
    se_hf = statistics.stdev(high_final) / math.sqrt(len(high_final))
    t3 = (mean_hf - 0.5) / se_hf if se_hf > 0 else 0
    print(f"\n  3) 高压环境: Planner 最终占比 vs 50%（初始基线）")
    print(f"     mean={mean_hf:.3f}  t={t3:+.2f}  [{sig_label(t3)}]")
    if mean_hf > 0.5 and t3 > 2:
        print(f"     → Planner 在高压下成为优势表型")

    # 保存
    out_path = "/Users/ddd/Desktop/uga-sandbox/results_v3.json"
    # 去掉 curve 省空间（太大）
    save = {}
    for p in ["high", "low"]:
        save[p] = []
        for r in results[p]:
            r2 = {k: v for k, v in r.items() if k != "planner_frac_curve"}
            save[p].append(r2)
    # 存一个代表性 seed 的 curve
    save["sample_curves"] = {
        "high": results["high"][0]["planner_frac_curve"],
        "low": results["low"][0]["planner_frac_curve"],
    }
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
