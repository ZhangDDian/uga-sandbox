"""
UGA Network · 演化沙盒 v4 — 草地模型

世界物理：
  - 40×40 网格，每格有能量（逻辑斯蒂增长）
  - 季节周期：growth_rate 每 1000 tick 高低切换
  - agent 有限视野（3格），移动花钱，吃草获能
  - agent 间推拉（social_force）双方都有代价

Agent：
  - 前馈网络（FF）vs 递归网络（RNN）混合种群
  - 基因 = 网络权重，遗传 + 变异
  - 死亡（余额≤0）+ 繁殖（余额>阈值）

验证目标：
  1. 记忆（RNN）在压力 + 季节变化下是否有选择优势
  2. 偏好分化（探索者 vs 定居者）是否涌现
  3. 干扰实验：烧地后 RNN vs FF 的恢复时间差异

纯 Python 标准库。
"""

import json
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Tuple

# ============ 超参数 ============

GRID_SIZE = 40
MAX_ENERGY = 100.0
GROWTH_RATE_HIGH = 0.08
GROWTH_RATE_LOW = 0.02
SEASON_PERIOD = 1000  # tick

N_INIT_AGENTS = 200
POP_CAP = 400
N_TICKS = 10000
INIT_BALANCE = 300.0
DEATH_THRESHOLD = 0.0
REPRO_THRESHOLD = 800.0
REPRO_COST_FRAC = 0.5

VISION_RANGE = 3
MOVE_COST_PER_STEP = 3.0
HARVEST_EFFICIENCY = 1.0
SOCIAL_FORCE_COST = 0.3  # 双方各付 |force| × 此系数
SOCIAL_FORCE_TRANSFER = 0.5  # 被拉方损失 |force| × 此系数

MUTATION_SIGMA = 0.1
RNN_TO_FF_MUTATION_RATE = 0.02  # 记忆类型变异概率

# 神经网络维度
N_INPUTS = 12   # 见 build_inputs()
N_HIDDEN = 8
N_OUTPUTS = 4   # dx, dy, harvest_force, social_force
# FF: weights = N_INPUTS*N_HIDDEN + N_HIDDEN + N_HIDDEN*N_OUTPUTS + N_OUTPUTS
# RNN: 额外 + N_HIDDEN*N_HIDDEN（隐藏层自连接）
FF_GENOME_LEN = N_INPUTS * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_OUTPUTS + N_OUTPUTS
RNN_GENOME_LEN = FF_GENOME_LEN + N_HIDDEN * N_HIDDEN

# 干扰实验
DISTURB_TICK = 8000
DISTURB_RADIUS = 3
N_DISTURB_AGENTS = 30
RECOVERY_WINDOW = 200


# ============ 神经网络 ============

def tanh(x):
    if x > 20: return 1.0
    if x < -20: return -1.0
    return math.tanh(x)


def forward_ff(inputs: list, genome: list) -> list:
    """前馈网络：inputs → hidden(tanh) → outputs(tanh)"""
    idx = 0
    # input → hidden
    hidden = []
    for h in range(N_HIDDEN):
        s = genome[idx + N_INPUTS]  # bias
        for i in range(N_INPUTS):
            s += inputs[i] * genome[idx + i]
        hidden.append(tanh(s))
        idx += N_INPUTS + 1
    # hidden → output
    outputs = []
    for o in range(N_OUTPUTS):
        s = genome[idx + N_HIDDEN]  # bias
        for h in range(N_HIDDEN):
            s += hidden[h] * genome[idx + h]
        outputs.append(tanh(s))
        idx += N_HIDDEN + 1
    return outputs


def forward_rnn(inputs: list, genome: list, hidden_state: list) -> Tuple[list, list]:
    """递归网络：inputs + prev_hidden → new_hidden(tanh) → outputs(tanh)"""
    idx = 0
    # input → hidden (同 FF)
    new_hidden = []
    for h in range(N_HIDDEN):
        s = genome[idx + N_INPUTS]  # bias
        for i in range(N_INPUTS):
            s += inputs[i] * genome[idx + i]
        idx += N_INPUTS + 1
        # 加上隐藏层自连接
        recur_offset = FF_GENOME_LEN + h * N_HIDDEN
        for hh in range(N_HIDDEN):
            s += hidden_state[hh] * genome[recur_offset + hh]
        new_hidden.append(tanh(s))
    # hidden → output
    outputs = []
    for o in range(N_OUTPUTS):
        s = genome[idx + N_HIDDEN]  # bias
        for h in range(N_HIDDEN):
            s += new_hidden[h] * genome[idx + h]
        outputs.append(tanh(s))
        idx += N_HIDDEN + 1
    return outputs, new_hidden


# ============ Agent ============

@dataclass
class Agent:
    id: int
    x: int
    y: int
    balance: float
    is_rnn: bool
    genome: list
    hidden_state: list = field(default_factory=lambda: [0.0] * N_HIDDEN)
    alive: bool = True
    birth_tick: int = 0
    parent_id: int = -1
    # 统计
    total_harvest: float = 0.0
    total_moved: float = 0.0
    ticks_alive: int = 0
    positions_visited: set = field(default_factory=set)
    # 干扰实验
    disturbed: bool = False
    pre_disturb_harvest_rate: float = 0.0
    post_disturb_recovery_tick: int = -1


def random_genome(is_rnn: bool) -> list:
    length = RNN_GENOME_LEN if is_rnn else FF_GENOME_LEN
    return [random.gauss(0, 0.5) for _ in range(length)]


def mutate_genome(genome: list, sigma: float) -> list:
    return [g + random.gauss(0, sigma) for g in genome]


# ============ World ============

class World:
    def __init__(self, growth_mode="seasonal"):
        self.grid = [[MAX_ENERGY * 0.5 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.agents: List[Agent] = []
        self.tick = 0
        self.next_id = 0
        self.growth_mode = growth_mode
        # 空间索引
        self.cell_agents = {}  # (x,y) → [agent_id, ...]
        # 时间序列
        self.pop_curve = []
        self.rnn_frac_curve = []
        self.avg_balance_curve = []
        self.avg_harvest_curve = []
        # 干扰实验结果
        self.disturb_results = []

    def current_growth_rate(self) -> float:
        if self.growth_mode == "seasonal":
            cycle = (self.tick // SEASON_PERIOD) % 2
            return GROWTH_RATE_HIGH if cycle == 0 else GROWTH_RATE_LOW
        elif self.growth_mode == "high":
            return GROWTH_RATE_HIGH
        else:
            return GROWTH_RATE_LOW

    def spawn_initial(self):
        for i in range(N_INIT_AGENTS):
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            is_rnn = i < N_INIT_AGENTS // 2
            agent = Agent(
                id=self.next_id, x=x, y=y,
                balance=INIT_BALANCE, is_rnn=is_rnn,
                genome=random_genome(is_rnn),
            )
            agent.positions_visited.add((x, y))
            self.agents.append(agent)
            self.next_id += 1
        self._rebuild_index()

    def _rebuild_index(self):
        self.cell_agents = {}
        for a in self.agents:
            if a.alive:
                key = (a.x, a.y)
                if key not in self.cell_agents:
                    self.cell_agents[key] = []
                self.cell_agents[key].append(a.id)

    def _get_alive(self) -> List[Agent]:
        return [a for a in self.agents if a.alive]

    def _agent_by_id(self, aid: int):
        return self.agents[aid] if aid < len(self.agents) else None

    def build_inputs(self, agent: Agent) -> list:
        """构建感知输入向量（12维）"""
        x, y = agent.x, agent.y
        # 1. 自身余额（归一化）
        bal_norm = agent.balance / REPRO_THRESHOLD

        # 2-3. 视野内能量：正前方/周围的平均能量，最高能量方向
        energies = []
        best_e = -1
        best_dx, best_dy = 0, 0
        for dx in range(-VISION_RANGE, VISION_RANGE + 1):
            for dy in range(-VISION_RANGE, VISION_RANGE + 1):
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % GRID_SIZE
                ny = (y + dy) % GRID_SIZE
                e = self.grid[nx][ny]
                energies.append(e)
                if e > best_e:
                    best_e = e
                    best_dx, best_dy = dx, dy
        avg_energy = sum(energies) / len(energies) / MAX_ENERGY if energies else 0
        # 最佳食物方向（归一化到 [-1,1]）
        food_dir_x = best_dx / VISION_RANGE
        food_dir_y = best_dy / VISION_RANGE

        # 4. 脚下能量
        here_energy = self.grid[x][y] / MAX_ENERGY

        # 5-6. 视野内 agent 数量和最近 agent 方向
        n_nearby = 0
        nearest_dist = 999
        near_dx, near_dy = 0, 0
        for dx in range(-VISION_RANGE, VISION_RANGE + 1):
            for dy in range(-VISION_RANGE, VISION_RANGE + 1):
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % GRID_SIZE
                ny = (y + dy) % GRID_SIZE
                key = (nx, ny)
                if key in self.cell_agents:
                    cnt = len(self.cell_agents[key])
                    n_nearby += cnt
                    dist = abs(dx) + abs(dy)
                    if dist < nearest_dist and cnt > 0:
                        nearest_dist = dist
                        near_dx, near_dy = dx, dy
        crowd = min(n_nearby / 10.0, 1.0)
        agent_dir_x = near_dx / VISION_RANGE if nearest_dist < 999 else 0
        agent_dir_y = near_dy / VISION_RANGE if nearest_dist < 999 else 0

        # 7. 季节信号（当前生长率归一化）
        season = self.current_growth_rate() / GROWTH_RATE_HIGH

        # 8. 上一 tick 收获量（简单记忆，用 total_harvest/ticks_alive 近似）
        recent_rate = agent.total_harvest / max(1, agent.ticks_alive) / 50.0

        return [
            bal_norm,       # 0: 余额
            avg_energy,     # 1: 周围平均能量
            food_dir_x,     # 2: 最佳食物方向 x
            food_dir_y,     # 3: 最佳食物方向 y
            here_energy,    # 4: 脚下能量
            crowd,          # 5: 周围拥挤度
            agent_dir_x,    # 6: 最近 agent 方向 x
            agent_dir_y,    # 7: 最近 agent 方向 y
            season,         # 8: 季节信号
            recent_rate,    # 9: 最近获能速率
            random.gauss(0, 0.1),  # 10: 噪声1
            random.gauss(0, 0.1),  # 11: 噪声2
        ]

    def step(self):
        self.tick += 1
        gr = self.current_growth_rate()

        # 1. 草生长
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                e = self.grid[x][y]
                self.grid[x][y] = e + gr * (1 - e / MAX_ENERGY) * MAX_ENERGY

        alive = self._get_alive()
        if not alive:
            self._record(0, 0, 0, 0)
            return

        # 2. Agent 行动
        moves = {}  # id → (new_x, new_y, harvest, social_target)
        for a in alive:
            a.ticks_alive += 1
            inputs = self.build_inputs(a)

            if a.is_rnn:
                outputs, a.hidden_state = forward_rnn(inputs, a.genome, a.hidden_state)
            else:
                outputs = forward_ff(inputs, a.genome)

            # 解码输出 [-1, 1]
            raw_dx = outputs[0]
            raw_dy = outputs[1]
            harvest_force = (outputs[2] + 1) / 2  # [0, 1]
            social_force = outputs[3]  # [-1, 1]

            # 移动：输出 > 0.33 移动一格，> 0.66 移动两格
            dx = 0
            if raw_dx > 0.33: dx = 1
            elif raw_dx > 0.66: dx = 2
            elif raw_dx < -0.33: dx = -1
            elif raw_dx < -0.66: dx = -2
            dy = 0
            if raw_dy > 0.33: dy = 1
            elif raw_dy > 0.66: dy = 2
            elif raw_dy < -0.33: dy = -1
            elif raw_dy < -0.66: dy = -2

            new_x = (a.x + dx) % GRID_SIZE
            new_y = (a.y + dy) % GRID_SIZE
            move_dist = abs(dx) + abs(dy)

            moves[a.id] = {
                "new_x": new_x, "new_y": new_y,
                "move_dist": move_dist,
                "harvest_force": harvest_force,
                "social_force": social_force,
            }

        # 3. 执行移动 + 扣移动成本
        for a in alive:
            m = moves[a.id]
            a.x = m["new_x"]
            a.y = m["new_y"]
            a.balance -= m["move_dist"] * MOVE_COST_PER_STEP
            a.total_moved += m["move_dist"]
            a.positions_visited.add((a.x, a.y))

        # 重建空间索引
        self._rebuild_index()

        # 4. 吃草（同格多人按 harvest_force 比例分）
        cell_harvesters = {}  # (x,y) → [(agent, force)]
        for a in alive:
            f = moves[a.id]["harvest_force"]
            if f > 0.01:
                key = (a.x, a.y)
                if key not in cell_harvesters:
                    cell_harvesters[key] = []
                cell_harvesters[key].append((a, f))

        total_harvest_tick = 0.0
        for (cx, cy), harvesters in cell_harvesters.items():
            available = self.grid[cx][cy]
            if available <= 0:
                continue
            total_force = sum(f for _, f in harvesters)
            for a, f in harvesters:
                share = (f / total_force) * min(available, total_force * HARVEST_EFFICIENCY * 10)
                actual = min(share, available)
                a.balance += actual
                a.total_harvest += actual
                available -= actual
                total_harvest_tick += actual
            self.grid[cx][cy] = max(0, available)

        # 5. 社交力（推拉）——只对同格或相邻格的 agent 生效
        for a in alive:
            sf = moves[a.id]["social_force"]
            if abs(sf) < 0.1:
                continue
            # 找最近的 agent
            target = None
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx = (a.x + dx) % GRID_SIZE
                    ny = (a.y + dy) % GRID_SIZE
                    key = (nx, ny)
                    if key in self.cell_agents:
                        for tid in self.cell_agents[key]:
                            t = self._agent_by_id(tid)
                            if t and t.alive and t.id != a.id:
                                target = t
                                break
                    if target:
                        break
                if target:
                    break
            if not target:
                continue
            # 执行推拉
            force_mag = abs(sf) * 30  # 缩放到有意义的量
            cost = force_mag * SOCIAL_FORCE_COST
            a.balance -= cost  # 发起方付代价
            if sf < 0:
                # 拉（抢）：从 target 拿
                transfer = min(force_mag * SOCIAL_FORCE_TRANSFER, target.balance * 0.5)
                target.balance -= transfer
                a.balance += transfer
                target.balance -= cost * 0.5  # 被抢方也有损耗
            else:
                # 推（给）：给 target
                give = min(force_mag * SOCIAL_FORCE_TRANSFER, a.balance * 0.3)
                a.balance -= give
                target.balance += give

        # 6. 死亡
        for a in alive:
            if a.balance <= DEATH_THRESHOLD:
                a.alive = False

        # 7. 繁殖
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
                # 记忆类型变异
                child_is_rnn = a.is_rnn
                if random.random() < RNN_TO_FF_MUTATION_RATE:
                    child_is_rnn = not child_is_rnn
                # 如果类型变了，重新初始化对应长度的基因
                if child_is_rnn != a.is_rnn:
                    child_genome = random_genome(child_is_rnn)
                else:
                    child_genome = mutate_genome(a.genome, MUTATION_SIGMA)
                # 子代出生在父代附近
                cx = (a.x + random.randint(-2, 2)) % GRID_SIZE
                cy = (a.y + random.randint(-2, 2)) % GRID_SIZE
                child = Agent(
                    id=self.next_id, x=cx, y=cy,
                    balance=child_balance, is_rnn=child_is_rnn,
                    genome=child_genome, birth_tick=self.tick,
                    parent_id=a.id,
                )
                child.positions_visited.add((cx, cy))
                self.next_id += 1
                new_agents.append(child)
                alive_count += 1
        self.agents.extend(new_agents)

        # 8. 记录
        alive_now = self._get_alive()
        n_alive = len(alive_now)
        rnn_count = sum(1 for a in alive_now if a.is_rnn)
        avg_bal = sum(a.balance for a in alive_now) / n_alive if n_alive else 0
        self._record(n_alive, rnn_count / n_alive if n_alive else 0,
                     avg_bal, total_harvest_tick / n_alive if n_alive else 0)

    def _record(self, pop, rnn_frac, avg_bal, avg_harvest):
        self.pop_curve.append(pop)
        self.rnn_frac_curve.append(rnn_frac)
        self.avg_balance_curve.append(avg_bal)
        self.avg_harvest_curve.append(avg_harvest)

    # ============ 干扰实验 ============

    def run_disturbance(self):
        """在 DISTURB_TICK 时烧掉选定 agent 脚下的草"""
        alive = self._get_alive()
        if len(alive) < N_DISTURB_AGENTS * 2:
            return

        # 选一半 RNN 一半 FF
        rnns = [a for a in alive if a.is_rnn]
        ffs = [a for a in alive if not a.is_rnn]
        n_each = min(N_DISTURB_AGENTS // 2, len(rnns), len(ffs))
        if n_each < 5:
            return

        selected_rnn = random.sample(rnns, n_each)
        selected_ff = random.sample(ffs, n_each)

        for a in selected_rnn + selected_ff:
            # 记录干扰前的获能速率
            a.pre_disturb_harvest_rate = a.total_harvest / max(1, a.ticks_alive)
            a.disturbed = True
            # 烧地：清空周围 DISTURB_RADIUS 范围的草
            for dx in range(-DISTURB_RADIUS, DISTURB_RADIUS + 1):
                for dy in range(-DISTURB_RADIUS, DISTURB_RADIUS + 1):
                    nx = (a.x + dx) % GRID_SIZE
                    ny = (a.y + dy) % GRID_SIZE
                    self.grid[nx][ny] = 0.0

    def check_recovery(self):
        """检查被干扰的 agent 是否恢复"""
        for a in self.agents:
            if not a.disturbed or not a.alive:
                continue
            if a.post_disturb_recovery_tick > 0:
                continue
            # 检查最近 20 tick 的获能率是否恢复到干扰前的 70%
            if a.ticks_alive < 1:
                continue
            recent_rate = a.total_harvest / a.ticks_alive
            if recent_rate >= a.pre_disturb_harvest_rate * 0.7:
                a.post_disturb_recovery_tick = self.tick - DISTURB_TICK

    def collect_disturb_results(self):
        rnn_recovery = []
        ff_recovery = []
        rnn_survived = 0
        ff_survived = 0
        rnn_total = 0
        ff_total = 0
        for a in self.agents:
            if not a.disturbed:
                continue
            if a.is_rnn:
                rnn_total += 1
                if a.alive:
                    rnn_survived += 1
                    if a.post_disturb_recovery_tick > 0:
                        rnn_recovery.append(a.post_disturb_recovery_tick)
            else:
                ff_total += 1
                if a.alive:
                    ff_survived += 1
                    if a.post_disturb_recovery_tick > 0:
                        ff_recovery.append(a.post_disturb_recovery_tick)

        return {
            "rnn_survived": rnn_survived,
            "rnn_total": rnn_total,
            "ff_survived": ff_survived,
            "ff_total": ff_total,
            "rnn_recovery_ticks": rnn_recovery,
            "ff_recovery_ticks": ff_recovery,
        }


# ============ 偏好分析 ============

def analyze_preferences(agents: List[Agent]):
    """分析存活 agent 的探索偏好"""
    alive = [a for a in agents if a.alive and a.ticks_alive > 100]
    if not alive:
        return {}

    rnn_ranges = []
    ff_ranges = []
    for a in alive:
        # 探索范围 = 访问过的独立格子数
        exploration = len(a.positions_visited)
        if a.is_rnn:
            rnn_ranges.append(exploration)
        else:
            ff_ranges.append(exploration)

    return {
        "rnn_exploration_mean": statistics.mean(rnn_ranges) if rnn_ranges else 0,
        "rnn_exploration_std": statistics.stdev(rnn_ranges) if len(rnn_ranges) > 1 else 0,
        "ff_exploration_mean": statistics.mean(ff_ranges) if ff_ranges else 0,
        "ff_exploration_std": statistics.stdev(ff_ranges) if len(ff_ranges) > 1 else 0,
        "rnn_n": len(rnn_ranges),
        "ff_n": len(ff_ranges),
    }


# ============ 主实验 ============

def run_experiment(seed: int, growth_mode: str = "seasonal") -> dict:
    random.seed(seed)
    world = World(growth_mode=growth_mode)
    world.spawn_initial()

    for t in range(N_TICKS):
        world.step()

        # 干扰实验
        if world.tick == DISTURB_TICK:
            world.run_disturbance()
        if world.tick > DISTURB_TICK:
            world.check_recovery()

        # 种群灭绝则提前终止
        if not world._get_alive():
            break

    # 收集结果
    disturb = world.collect_disturb_results()
    prefs = analyze_preferences(world.agents)

    # 关键时间点的 RNN 占比
    curve = world.rnn_frac_curve
    n = len(curve)
    rnn_early = statistics.mean(curve[:500]) if n >= 500 else 0
    rnn_mid = statistics.mean(curve[n//2 - 250:n//2 + 250]) if n >= 500 else 0
    rnn_late = statistics.mean(curve[-500:]) if n >= 500 else 0

    return {
        "seed": seed,
        "growth_mode": growth_mode,
        "ticks_ran": world.tick,
        "final_pop": len(world._get_alive()),
        "rnn_frac_early": rnn_early,
        "rnn_frac_mid": rnn_mid,
        "rnn_frac_late": rnn_late,
        "rnn_frac_delta": rnn_late - rnn_early,
        "disturbance": disturb,
        "preferences": prefs,
    }


def welch_t(a: list, b: list) -> tuple:
    if len(a) < 2 or len(b) < 2:
        return 0.0, 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    denom = va/na + vb/nb
    if denom <= 0:
        return 0.0, 0.0
    se = math.sqrt(denom)
    t = (ma - mb) / se
    df = denom**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1)) if denom > 0 else 1
    return t, df


def main():
    N_SEEDS = 5  # 先跑 5 个 seed 看看（完整实验跑 20）
    print("=" * 72)
    print("UGA Network · 演化沙盒 v4 — 草地模型")
    print(f"  网格: {GRID_SIZE}×{GRID_SIZE}  种群上限: {POP_CAP}")
    print(f"  ticks: {N_TICKS}  季节周期: {SEASON_PERIOD}")
    print(f"  视野: {VISION_RANGE}  移动成本: {MOVE_COST_PER_STEP}/格")
    print(f"  seeds: {N_SEEDS}")
    print("=" * 72)

    all_results = {"seasonal": [], "high": []}

    for mode in ["seasonal", "high"]:
        print(f"\n{'─'*72}")
        print(f"  环境: {mode}")
        print(f"{'─'*72}")
        for seed in range(N_SEEDS):
            print(f"  seed {seed} 开始...", end="", flush=True)
            r = run_experiment(seed, mode)
            all_results[mode].append(r)
            print(f" 完成 (ticks={r['ticks_ran']}, pop={r['final_pop']}, "
                  f"RNN占比: {r['rnn_frac_early']:.2f}→{r['rnn_frac_late']:.2f})")

    # ============ 汇总 ============
    print(f"\n{'='*72}")
    print("汇总结果")
    print(f"{'='*72}")

    for mode in ["seasonal", "high"]:
        runs = all_results[mode]
        print(f"\n  [{mode}]")
        deltas = [r["rnn_frac_delta"] for r in runs]
        print(f"  RNN占比变化: {statistics.mean(deltas):+.3f} ± "
              f"{statistics.stdev(deltas):.3f}" if len(deltas) > 1 else "")
        late_fracs = [r["rnn_frac_late"] for r in runs]
        print(f"  RNN最终占比: {statistics.mean(late_fracs):.3f}")

        # 干扰实验
        all_rnn_rec = []
        all_ff_rec = []
        for r in runs:
            all_rnn_rec.extend(r["disturbance"]["rnn_recovery_ticks"])
            all_ff_rec.extend(r["disturbance"]["ff_recovery_ticks"])
        if all_rnn_rec and all_ff_rec:
            print(f"  干扰恢复时间: RNN={statistics.mean(all_rnn_rec):.1f} "
                  f"FF={statistics.mean(all_ff_rec):.1f}")
            t, df = welch_t(all_rnn_rec, all_ff_rec)
            sig = "***" if abs(t) > 3 else "**" if abs(t) > 2 else "*" if abs(t) > 1.5 else "ns"
            print(f"  Welch t={t:+.2f} df={df:.1f} [{sig}]")

        # 探索偏好
        rnn_exp = [r["preferences"]["rnn_exploration_mean"] for r in runs
                   if r["preferences"].get("rnn_exploration_mean", 0) > 0]
        ff_exp = [r["preferences"]["ff_exploration_mean"] for r in runs
                  if r["preferences"].get("ff_exploration_mean", 0) > 0]
        if rnn_exp and ff_exp:
            print(f"  探索范围: RNN={statistics.mean(rnn_exp):.1f} "
                  f"FF={statistics.mean(ff_exp):.1f} 格")

    # 核心假设检验
    print(f"\n{'='*72}")
    print("核心假设检验")
    print(f"{'='*72}")

    seasonal_deltas = [r["rnn_frac_delta"] for r in all_results["seasonal"]]
    high_deltas = [r["rnn_frac_delta"] for r in all_results["high"]]

    print(f"\n  H1: 季节环境下 RNN 有选择优势（占比上升）")
    m = statistics.mean(seasonal_deltas)
    if len(seasonal_deltas) > 1:
        se = statistics.stdev(seasonal_deltas) / math.sqrt(len(seasonal_deltas))
        t_one = m / se if se > 0 else 0
        sig = "***" if t_one > 3 else "**" if t_one > 2 else "*" if t_one > 1.5 else "ns"
        print(f"  季节环境 RNN Δ={m:+.3f}  t={t_one:.2f}  [{sig}]")
    else:
        print(f"  季节环境 RNN Δ={m:+.3f}  (样本不足)")

    print(f"\n  H2: 恒定环境下 RNN 无优势（对照）")
    m2 = statistics.mean(high_deltas)
    if len(high_deltas) > 1:
        se2 = statistics.stdev(high_deltas) / math.sqrt(len(high_deltas))
        t_two = m2 / se2 if se2 > 0 else 0
        sig2 = "***" if abs(t_two) > 3 else "**" if abs(t_two) > 2 else "*" if abs(t_two) > 1.5 else "ns"
        print(f"  恒定环境 RNN Δ={m2:+.3f}  t={t_two:.2f}  [{sig2}]")
    else:
        print(f"  恒定环境 RNN Δ={m2:+.3f}  (样本不足)")

    # 保存
    out_path = "/Users/ddd/Desktop/uga-sandbox/results_v4.json"
    # 清理不可序列化的数据
    save_results = {}
    for mode in all_results:
        save_results[mode] = []
        for r in all_results[mode]:
            r2 = {k: v for k, v in r.items()}
            save_results[mode].append(r2)
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
