"""
UGA Network · 演化沙盒 v5 — 严酷草地

v4 的问题：环境太温和，到处能看到草，记忆没用武之地。
v5 的改动：三种信息压力同时施加，不预设哪种更重要，让演化自己筛。

  1. 时间压力：冬天 growth=0（完全不长草），夏天正常。
     没记忆的 agent 每次冬天来都是"第一次"，有记忆的能提前囤。
  2. 空间压力：地图不均匀——有大片荒漠（永远不长草），肥沃区集中在几个绿洲。
     绿洲之间隔着看不见的荒漠，穿越需要凭记忆。
  3. 社交压力：被抢过的记录作为输入。有记忆的能学会躲避/反击。

视野缩到 2 格（只看紧邻的一圈）。
地图 50×50，但 40% 是荒漠（永远 0 能量）。

纯 Python 标准库。
"""

import json
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Tuple

# ============ 超参数 ============

GRID_SIZE = 50
MAX_ENERGY = 100.0
GROWTH_RATE_SUMMER = 0.12
GROWTH_RATE_WINTER = 0.0  # 冬天完全不长
SEASON_LENGTH = 800       # 夏天 800 tick
WINTER_LENGTH = 400       # 冬天 400 tick
CYCLE_LENGTH = SEASON_LENGTH + WINTER_LENGTH

DESERT_FRACTION = 0.40    # 40% 格子是永久荒漠
N_OASES = 6              # 绿洲数量
OASIS_RADIUS = 6         # 绿洲半径

N_INIT_AGENTS = 200
POP_CAP = 350
N_TICKS = 12000
INIT_BALANCE = 400.0
DEATH_THRESHOLD = 0.0
REPRO_THRESHOLD = 900.0
REPRO_COST_FRAC = 0.5

VISION_RANGE = 2          # 缩到 2 格
MOVE_COST_PER_STEP = 2.0
BASE_METABOLISM = 1.5     # 每 tick 基础代谢（活着就要花钱）
HARVEST_MAX = 15.0        # 单次最大收获
SOCIAL_COST = 0.3
SOCIAL_TRANSFER = 0.4

MUTATION_SIGMA = 0.08
TYPE_MUTATION_RATE = 0.01

# 神经网络
N_INPUTS = 14
N_HIDDEN = 8
N_OUTPUTS = 5  # dx, dy, harvest_force, social_force, "stay_intent"
FF_GENOME_LEN = N_INPUTS * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_OUTPUTS + N_OUTPUTS
RNN_GENOME_LEN = FF_GENOME_LEN + N_HIDDEN * N_HIDDEN

# 干扰实验
DISTURB_TICK = 10000
DISTURB_RADIUS = 4
N_DISTURB_AGENTS = 40
RECOVERY_WINDOW = 500


# ============ 地图生成 ============

def generate_terrain(seed: int) -> Tuple[list, list]:
    """生成地形：荒漠 + 绿洲。返回 (fertility_map, oasis_centers)"""
    rng = random.Random(seed * 1000 + 7)
    # 默认全是荒漠
    fertility = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]

    # 放置绿洲
    centers = []
    for _ in range(N_OASES):
        # 确保绿洲不重叠太多
        for _attempt in range(50):
            cx = rng.randint(OASIS_RADIUS, GRID_SIZE - OASIS_RADIUS - 1)
            cy = rng.randint(OASIS_RADIUS, GRID_SIZE - OASIS_RADIUS - 1)
            too_close = any(abs(cx - ox) + abs(cy - oy) < OASIS_RADIUS * 2.5
                           for ox, oy in centers)
            if not too_close:
                break
        centers.append((cx, cy))
        # 画绿洲（圆形渐变）
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < OASIS_RADIUS:
                    f = 1.0 - (dist / OASIS_RADIUS) ** 2
                    fertility[x][y] = max(fertility[x][y], f)

    # 在绿洲之间画一些窄走廊（半宽 1-2 格），让迁徙有路可走但很窄
    for i in range(len(centers) - 1):
        x0, y0 = centers[i]
        x1, y1 = centers[i + 1]
        steps = max(abs(x1 - x0), abs(y1 - y0))
        for t in range(steps + 1):
            frac = t / max(1, steps)
            px = int(x0 + (x1 - x0) * frac)
            py = int(y0 + (y1 - y0) * frac)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx = max(0, min(GRID_SIZE-1, px + dx))
                    ny = max(0, min(GRID_SIZE-1, py + dy))
                    fertility[nx][ny] = max(fertility[nx][ny], 0.3)

    return fertility, centers


# ============ 神经网络 ============

def tanh_clamp(x):
    if x > 15: return 1.0
    if x < -15: return -1.0
    return math.tanh(x)


def forward_ff(inputs: list, genome: list) -> list:
    idx = 0
    hidden = []
    for h in range(N_HIDDEN):
        s = genome[idx + N_INPUTS]
        for i in range(N_INPUTS):
            s += inputs[i] * genome[idx + i]
        hidden.append(tanh_clamp(s))
        idx += N_INPUTS + 1
    outputs = []
    for o in range(N_OUTPUTS):
        s = genome[idx + N_HIDDEN]
        for h in range(N_HIDDEN):
            s += hidden[h] * genome[idx + h]
        outputs.append(tanh_clamp(s))
        idx += N_HIDDEN + 1
    return outputs


def forward_rnn(inputs: list, genome: list, hidden_state: list) -> Tuple[list, list]:
    idx = 0
    new_hidden = []
    for h in range(N_HIDDEN):
        s = genome[idx + N_INPUTS]
        for i in range(N_INPUTS):
            s += inputs[i] * genome[idx + i]
        idx += N_INPUTS + 1
        recur_base = FF_GENOME_LEN + h * N_HIDDEN
        for hh in range(N_HIDDEN):
            s += hidden_state[hh] * genome[recur_base + hh]
        new_hidden.append(tanh_clamp(s))
    outputs = []
    for o in range(N_OUTPUTS):
        s = genome[idx + N_HIDDEN]
        for h in range(N_HIDDEN):
            s += new_hidden[h] * genome[idx + h]
        outputs.append(tanh_clamp(s))
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
    times_robbed: int = 0
    times_robbing: int = 0
    # 干扰
    disturbed: bool = False
    pre_disturb_rate: float = 0.0
    recovery_tick: int = -1
    # 轨迹（只记最近 200 tick 的位置用于分析）
    recent_positions: list = field(default_factory=list)


def random_genome(is_rnn: bool) -> list:
    length = RNN_GENOME_LEN if is_rnn else FF_GENOME_LEN
    return [random.gauss(0, 0.4) for _ in range(length)]


def mutate_genome(genome: list, sigma: float) -> list:
    return [g + random.gauss(0, sigma) for g in genome]


# ============ World ============

class World:
    def __init__(self, seed: int):
        self.seed = seed
        self.fertility, self.oasis_centers = generate_terrain(seed)
        self.grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]
        # 初始能量 = 肥力 × MAX
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.grid[x][y] = self.fertility[x][y] * MAX_ENERGY * 0.7
        self.agents: List[Agent] = []
        self.tick = 0
        self.next_id = 0
        self.cell_index = {}  # (x,y) → set of agent ids
        # 时间序列
        self.pop_curve = []
        self.rnn_frac_curve = []
        self.avg_balance_curve = []

    def is_winter(self) -> bool:
        phase = self.tick % CYCLE_LENGTH
        return phase >= SEASON_LENGTH

    def spawn_initial(self):
        # 在绿洲内随机放置
        for i in range(N_INIT_AGENTS):
            # 随机选一个绿洲
            cx, cy = random.choice(self.oasis_centers)
            x = (cx + random.randint(-OASIS_RADIUS+1, OASIS_RADIUS-1)) % GRID_SIZE
            y = (cy + random.randint(-OASIS_RADIUS+1, OASIS_RADIUS-1)) % GRID_SIZE
            is_rnn = i < N_INIT_AGENTS // 2
            agent = Agent(
                id=self.next_id, x=x, y=y,
                balance=INIT_BALANCE, is_rnn=is_rnn,
                genome=random_genome(is_rnn),
            )
            self.agents.append(agent)
            self.next_id += 1
        self._rebuild_index()

    def _rebuild_index(self):
        self.cell_index = {}
        for a in self.agents:
            if a.alive:
                key = (a.x, a.y)
                if key not in self.cell_index:
                    self.cell_index[key] = set()
                self.cell_index[key].add(a.id)

    def _get_alive(self) -> List[Agent]:
        return [a for a in self.agents if a.alive]

    def build_inputs(self, agent: Agent) -> list:
        x, y = agent.x, agent.y
        # 0: 余额归一化
        bal = agent.balance / REPRO_THRESHOLD
        # 1: 脚下能量
        here = self.grid[x][y] / MAX_ENERGY
        # 2-3: 视野内最佳食物方向
        best_e = -1
        best_dx, best_dy = 0, 0
        total_e = 0.0
        n_cells = 0
        for dx in range(-VISION_RANGE, VISION_RANGE + 1):
            for dy in range(-VISION_RANGE, VISION_RANGE + 1):
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % GRID_SIZE
                ny = (y + dy) % GRID_SIZE
                e = self.grid[nx][ny]
                total_e += e
                n_cells += 1
                if e > best_e:
                    best_e = e
                    best_dx, best_dy = dx, dy
        avg_e = total_e / n_cells / MAX_ENERGY if n_cells else 0
        food_dx = best_dx / VISION_RANGE
        food_dy = best_dy / VISION_RANGE
        # 4: 周围平均能量
        # 5-6: 最近 agent 方向
        n_nearby = 0
        near_dx, near_dy = 0, 0
        near_dist = 99
        for dx in range(-VISION_RANGE, VISION_RANGE + 1):
            for dy in range(-VISION_RANGE, VISION_RANGE + 1):
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % GRID_SIZE
                ny = (y + dy) % GRID_SIZE
                key = (nx, ny)
                if key in self.cell_index:
                    cnt = len(self.cell_index[key])
                    n_nearby += cnt
                    d = abs(dx) + abs(dy)
                    if d < near_dist and cnt > 0:
                        near_dist = d
                        near_dx, near_dy = dx, dy
        crowd = min(n_nearby / 8.0, 1.0)
        agent_dx = near_dx / VISION_RANGE if near_dist < 99 else 0
        agent_dy = near_dy / VISION_RANGE if near_dist < 99 else 0
        # 7: 是不是冬天（0 或 1）
        winter = 1.0 if self.is_winter() else 0.0
        # 8: 冬天还剩多少 tick（归一化）
        phase = self.tick % CYCLE_LENGTH
        if phase >= SEASON_LENGTH:
            winter_remaining = (CYCLE_LENGTH - phase) / WINTER_LENGTH
        else:
            winter_remaining = 0.0
        # 9: 被抢次数（归一化）
        robbed_signal = min(agent.times_robbed / 5.0, 1.0)
        # 10: 脚下肥力（永久属性，agent 可以学会识别好地）
        fertility_here = self.fertility[x][y]
        # 11: 移动量（归一化，反映自己是定居还是游牧）
        mobility = min(agent.total_moved / max(1, agent.ticks_alive) / 2.0, 1.0)
        # 12-13: 噪声
        noise1 = random.gauss(0, 0.1)
        noise2 = random.gauss(0, 0.1)

        return [bal, here, food_dx, food_dy, avg_e, crowd,
                agent_dx, agent_dy, winter, winter_remaining,
                robbed_signal, fertility_here, mobility, noise1]

    def step(self):
        self.tick += 1

        # 1. 草生长
        is_w = self.is_winter()
        gr = GROWTH_RATE_WINTER if is_w else GROWTH_RATE_SUMMER
        if gr > 0:
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    f = self.fertility[x][y]
                    if f > 0:
                        e = self.grid[x][y]
                        cap = f * MAX_ENERGY
                        self.grid[x][y] = min(cap, e + gr * (cap - e))

        alive = self._get_alive()
        if not alive:
            self._record(0, 0, 0)
            return

        # 2. 基础代谢
        for a in alive:
            a.balance -= BASE_METABOLISM
            a.ticks_alive += 1

        # 3. 感知 + 决策
        decisions = {}
        for a in alive:
            inputs = self.build_inputs(a)
            if a.is_rnn:
                outputs, a.hidden_state = forward_rnn(inputs, a.genome, a.hidden_state)
            else:
                outputs = forward_ff(inputs, a.genome)
            decisions[a.id] = outputs

        # 4. 移动
        for a in alive:
            out = decisions[a.id]
            dx = 0
            if out[0] > 0.3: dx = 1
            elif out[0] < -0.3: dx = -1
            dy = 0
            if out[1] > 0.3: dy = 1
            elif out[1] < -0.3: dy = -1
            # "stay_intent"（第 5 个输出）> 0.5 时不动，省移动费
            if out[4] > 0.5:
                dx, dy = 0, 0
            dist = abs(dx) + abs(dy)
            a.x = (a.x + dx) % GRID_SIZE
            a.y = (a.y + dy) % GRID_SIZE
            a.balance -= dist * MOVE_COST_PER_STEP
            a.total_moved += dist
            # 记录最近位置
            a.recent_positions.append((a.x, a.y))
            if len(a.recent_positions) > 200:
                a.recent_positions.pop(0)

        self._rebuild_index()

        # 5. 吃草
        cell_eaters = {}
        for a in alive:
            out = decisions[a.id]
            force = (out[2] + 1) / 2  # [0,1]
            if force > 0.05:
                key = (a.x, a.y)
                if key not in cell_eaters:
                    cell_eaters[key] = []
                cell_eaters[key].append((a, force))

        for (cx, cy), eaters in cell_eaters.items():
            available = self.grid[cx][cy]
            if available <= 0.1:
                continue
            total_force = sum(f for _, f in eaters)
            max_take = min(available, HARVEST_MAX * len(eaters))
            for a, f in eaters:
                share = (f / total_force) * max_take
                actual = min(share, available)
                a.balance += actual
                a.total_harvest += actual
                available -= actual
            self.grid[cx][cy] = max(0, available)

        # 6. 社交（推拉）
        for a in alive:
            out = decisions[a.id]
            sf = out[3]
            if abs(sf) < 0.2:
                continue
            # 找相邻格最近的 agent
            target = None
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx = (a.x + dx) % GRID_SIZE
                    ny = (a.y + dy) % GRID_SIZE
                    key = (nx, ny)
                    if key in self.cell_index:
                        for tid in self.cell_index[key]:
                            if tid != a.id and tid < len(self.agents):
                                t = self.agents[tid]
                                if t.alive:
                                    target = t
                                    break
                    if target:
                        break
                if target:
                    break
            if not target:
                continue
            force_mag = abs(sf) * 25
            cost = force_mag * SOCIAL_COST
            a.balance -= cost
            if sf < 0:  # 抢
                steal = min(force_mag * SOCIAL_TRANSFER, target.balance * 0.3)
                target.balance -= steal
                a.balance += steal
                target.balance -= cost * 0.3
                target.times_robbed += 1
                a.times_robbing += 1
            else:  # 给
                give = min(force_mag * SOCIAL_TRANSFER * 0.5, a.balance * 0.2)
                a.balance -= give
                target.balance += give

        # 7. 死亡
        for a in alive:
            if a.balance <= DEATH_THRESHOLD:
                a.alive = False

        # 8. 繁殖
        new_agents = []
        alive_count = sum(1 for a in self.agents if a.alive)
        for a in self.agents:
            if not a.alive:
                continue
            if alive_count >= POP_CAP:
                break
            if a.balance > REPRO_THRESHOLD:
                child_bal = a.balance * REPRO_COST_FRAC
                a.balance -= child_bal
                child_rnn = a.is_rnn
                if random.random() < TYPE_MUTATION_RATE:
                    child_rnn = not child_rnn
                if child_rnn != a.is_rnn:
                    child_genome = random_genome(child_rnn)
                else:
                    child_genome = mutate_genome(a.genome, MUTATION_SIGMA)
                cx = (a.x + random.randint(-1, 1)) % GRID_SIZE
                cy = (a.y + random.randint(-1, 1)) % GRID_SIZE
                child = Agent(
                    id=self.next_id, x=cx, y=cy,
                    balance=child_bal, is_rnn=child_rnn,
                    genome=child_genome, birth_tick=self.tick,
                    parent_id=a.id,
                )
                self.next_id += 1
                new_agents.append(child)
                alive_count += 1
        self.agents.extend(new_agents)

        # 9. 记录
        alive_now = self._get_alive()
        n = len(alive_now)
        rnn_n = sum(1 for a in alive_now if a.is_rnn)
        avg_b = sum(a.balance for a in alive_now) / n if n else 0
        self._record(n, rnn_n / n if n else 0, avg_b)

    def _record(self, pop, rnn_frac, avg_bal):
        self.pop_curve.append(pop)
        self.rnn_frac_curve.append(rnn_frac)
        self.avg_balance_curve.append(avg_bal)

    # ============ 干扰实验 ============

    def run_disturbance(self):
        alive = self._get_alive()
        rnns = [a for a in alive if a.is_rnn and a.ticks_alive > 500]
        ffs = [a for a in alive if not a.is_rnn and a.ticks_alive > 500]
        n_each = min(N_DISTURB_AGENTS // 2, len(rnns), len(ffs))
        if n_each < 3:
            return
        selected = random.sample(rnns, n_each) + random.sample(ffs, n_each)
        for a in selected:
            a.pre_disturb_rate = a.total_harvest / max(1, a.ticks_alive)
            a.disturbed = True
            # 烧掉脚下的草
            for dx in range(-DISTURB_RADIUS, DISTURB_RADIUS + 1):
                for dy in range(-DISTURB_RADIUS, DISTURB_RADIUS + 1):
                    nx = (a.x + dx) % GRID_SIZE
                    ny = (a.y + dy) % GRID_SIZE
                    self.grid[nx][ny] = 0.0

    def check_recovery(self):
        ticks_since = self.tick - DISTURB_TICK
        if ticks_since > RECOVERY_WINDOW:
            return
        for a in self.agents:
            if not a.disturbed or not a.alive or a.recovery_tick > 0:
                continue
            # 看最近 20 tick 有没有正常获能
            if a.ticks_alive < 20:
                continue
            recent_harvest = a.total_harvest / a.ticks_alive
            if recent_harvest >= a.pre_disturb_rate * 0.6:
                a.recovery_tick = ticks_since

    def collect_results(self) -> dict:
        alive = self._get_alive()
        # 干扰恢复
        rnn_rec, ff_rec = [], []
        rnn_surv, ff_surv = 0, 0
        rnn_tot, ff_tot = 0, 0
        for a in self.agents:
            if not a.disturbed:
                continue
            if a.is_rnn:
                rnn_tot += 1
                if a.alive:
                    rnn_surv += 1
                    if a.recovery_tick > 0:
                        rnn_rec.append(a.recovery_tick)
            else:
                ff_tot += 1
                if a.alive:
                    ff_surv += 1
                    if a.recovery_tick > 0:
                        ff_rec.append(a.recovery_tick)

        # 探索范围
        rnn_mobility, ff_mobility = [], []
        for a in alive:
            if a.ticks_alive < 200:
                continue
            unique_pos = len(set(a.recent_positions))
            mob = unique_pos / min(200, len(a.recent_positions)) if a.recent_positions else 0
            if a.is_rnn:
                rnn_mobility.append(mob)
            else:
                ff_mobility.append(mob)

        # RNN 频率曲线
        curve = self.rnn_frac_curve
        n = len(curve)

        return {
            "ticks_ran": self.tick,
            "final_pop": len(alive),
            "rnn_frac_early": statistics.mean(curve[:300]) if n >= 300 else 0.5,
            "rnn_frac_late": statistics.mean(curve[-300:]) if n >= 300 else 0.5,
            "rnn_frac_delta": (statistics.mean(curve[-300:]) - statistics.mean(curve[:300])) if n >= 300 else 0,
            "disturb": {
                "rnn_survived": rnn_surv, "rnn_total": rnn_tot,
                "ff_survived": ff_surv, "ff_total": ff_tot,
                "rnn_recovery": rnn_rec, "ff_recovery": ff_rec,
            },
            "mobility": {
                "rnn_mean": statistics.mean(rnn_mobility) if rnn_mobility else 0,
                "ff_mean": statistics.mean(ff_mobility) if ff_mobility else 0,
                "rnn_n": len(rnn_mobility), "ff_n": len(ff_mobility),
            },
        }


# ============ 统计 ============

def welch_t(a: list, b: list) -> Tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return 0.0, 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    d = va/na + vb/nb
    if d <= 0:
        return 0.0, 0.0
    t = (ma - mb) / math.sqrt(d)
    df = d**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1)) if d > 0 else 1
    return t, df


# ============ Main ============

def main():
    N_SEEDS = 5
    print("=" * 72)
    print("UGA Network · 演化沙盒 v5 — 严酷草地")
    print(f"  地图: {GRID_SIZE}×{GRID_SIZE}  绿洲×{N_OASES}(半径{OASIS_RADIUS})  荒漠{DESERT_FRACTION*100:.0f}%")
    print(f"  季节: 夏{SEASON_LENGTH}+冬{WINTER_LENGTH}={CYCLE_LENGTH} tick")
    print(f"  冬天 growth=0（完全不长草）")
    print(f"  视野: {VISION_RANGE}格  移动代价: {MOVE_COST_PER_STEP}/格  基础代谢: {BASE_METABOLISM}/tick")
    print(f"  种群: {N_INIT_AGENTS}→上限{POP_CAP}  ticks: {N_TICKS}  seeds: {N_SEEDS}")
    print(f"  初始: 50% RNN + 50% FF，混合竞争")
    print("=" * 72)

    results = []
    for seed in range(N_SEEDS):
        print(f"\n  seed {seed} ...", end="", flush=True)
        random.seed(seed)
        world = World(seed)
        world.spawn_initial()

        for t in range(N_TICKS):
            world.step()
            if world.tick == DISTURB_TICK:
                world.run_disturbance()
            if world.tick > DISTURB_TICK:
                world.check_recovery()
            if not world._get_alive():
                print(f" 灭绝 @tick {world.tick}", end="")
                break

        r = world.collect_results()
        r["seed"] = seed
        results.append(r)
        print(f" 完成 (pop={r['final_pop']}, "
              f"RNN: {r['rnn_frac_early']:.2f}→{r['rnn_frac_late']:.2f}, "
              f"Δ={r['rnn_frac_delta']:+.2f})")

    # ============ 汇总 ============
    print(f"\n{'='*72}")
    print("汇总")
    print(f"{'='*72}")

    deltas = [r["rnn_frac_delta"] for r in results]
    print(f"\n  RNN 占比变化: {statistics.mean(deltas):+.3f} ± "
          f"{statistics.stdev(deltas):.3f}" if len(deltas) > 1 else "")
    late = [r["rnn_frac_late"] for r in results]
    print(f"  RNN 最终占比: {statistics.mean(late):.3f}")

    # 方向判断
    m = statistics.mean(deltas)
    if len(deltas) > 1:
        se = statistics.stdev(deltas) / math.sqrt(len(deltas))
        t_val = m / se if se > 0 else 0
        sig = "***" if abs(t_val) > 3.5 else "**" if abs(t_val) > 2.5 else "*" if abs(t_val) > 1.7 else "ns"
        print(f"  单样本 t={t_val:+.2f} [{sig}]")
        if m > 0 and abs(t_val) > 2:
            print(f"  → 记忆在此环境下有选择优势")
        elif m < 0 and abs(t_val) > 2:
            print(f"  → 记忆在此环境下有选择劣势（基因搜索空间太大的代价）")
        else:
            print(f"  → 无显著差异")

    # 干扰实验
    all_rnn_rec, all_ff_rec = [], []
    for r in results:
        all_rnn_rec.extend(r["disturb"]["rnn_recovery"])
        all_ff_rec.extend(r["disturb"]["ff_recovery"])
    if all_rnn_rec and all_ff_rec:
        print(f"\n  干扰恢复: RNN={statistics.mean(all_rnn_rec):.1f}tick  "
              f"FF={statistics.mean(all_ff_rec):.1f}tick")
        t, df = welch_t(all_rnn_rec, all_ff_rec)
        sig = "***" if abs(t) > 3 else "**" if abs(t) > 2 else "*" if abs(t) > 1.5 else "ns"
        print(f"  Welch t={t:+.2f} df={df:.1f} [{sig}]")
        if t < -1.5:
            print(f"  → RNN 恢复更快（记忆有助于应对干扰）")
    elif not all_rnn_rec and not all_ff_rec:
        print(f"\n  干扰恢复: 无有效数据（可能都没恢复或样本太少）")

    # 探索偏好
    rnn_mob = [r["mobility"]["rnn_mean"] for r in results if r["mobility"]["rnn_n"] > 5]
    ff_mob = [r["mobility"]["ff_mean"] for r in results if r["mobility"]["ff_n"] > 5]
    if rnn_mob and ff_mob:
        print(f"\n  移动多样性: RNN={statistics.mean(rnn_mob):.3f}  FF={statistics.mean(ff_mob):.3f}")

    # 存活率
    rnn_surv_rate = sum(r["disturb"]["rnn_survived"] for r in results) / max(1, sum(r["disturb"]["rnn_total"] for r in results))
    ff_surv_rate = sum(r["disturb"]["ff_survived"] for r in results) / max(1, sum(r["disturb"]["ff_total"] for r in results))
    print(f"\n  干扰后存活率: RNN={rnn_surv_rate:.1%}  FF={ff_surv_rate:.1%}")

    # 保存
    out_path = "/Users/ddd/Desktop/uga-sandbox/results_v5.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
