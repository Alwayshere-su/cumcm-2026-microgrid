"""第一问 两层模型：LP 影子价格 + 连续状态 DP 斜率递推。

本模块只做 DP 部分的独立复现，不调用任何 LP/MILP 求解器：
  * 状态量是连续的内部储电量 e（kWh），不做 SOC 网格离散；
  * 价值函数 F_t(e) 由「凸分段线性函数」显式表示，用斜率合并 / 下确界卷积递推；
  * 命题 3.1 保证 F_t 凸且分段线性，故有限折线表示是精确的。

关键对象
--------
CPWL                 : 定义在 [lo, hi] 上、斜率自左向右单调非减的凸分段线性函数
build_phi / build_h  : 单时段费用函数 phi_t(x) 及其反射 h_t(y)=phi_t(-y)
two_layer_dp         : 反向递推价值函数，正向回代最优动作

约定（与参考论文一致）
----------------------
x_t = E_t - E_{t-1}            内部电量增量，正=充电、负=放电
psi(x) = x/eta (x>=0) / eta*x (x<0)   交流侧净用电
g_t = [n_t + psi(x_t)]^+       最优计划购电量，n_t = L_t - V_t
F_t(e) = min_{e'} {phi_t(e'-e) + F_{t+1}(e')},  F_{T+1}(e)=0 (e=6000) 否则 +inf
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# 凸分段线性函数
# --------------------------------------------------------------------------- #
class CPWL:
    """凸分段线性函数，定义在 [lo, lo + sum(widths)] 上。

    slopes 长度与 widths 相同，自左向右单调非减（凸性）。
    点函数用 widths=slopes=[] 表示（定义域退化为单点 lo==hi）。
    断点与断点函数值在构造时缓存，便于向量化求值。
    """

    __slots__ = ("lo", "f_lo", "widths", "slopes", "breaks", "cv")

    def __init__(self, lo: float, f_lo: float, widths, slopes) -> None:
        self.lo = float(lo)
        self.f_lo = float(f_lo)
        self.widths = np.asarray(widths, dtype=float)
        self.slopes = np.asarray(slopes, dtype=float)
        if self.widths.size != self.slopes.size:
            raise ValueError("widths/slopes 长度不一致")
        self.breaks = np.concatenate(([self.lo], self.lo + np.cumsum(self.widths)))
        self.cv = np.concatenate(([self.f_lo], self.f_lo + np.cumsum(self.widths * self.slopes)))

    # -- 基本属性 ---------------------------------------------------------- #
    @property
    def hi(self) -> float:
        return float(self.breaks[-1])

    @property
    def n_segments(self) -> int:
        return len(self.widths)

    # -- 向量化求值 --------------------------------------------------------- #
    def slopes_at(self, xs) -> np.ndarray:
        if self.n_segments == 0:
            raise ValueError("点函数没有斜率")
        xs = np.asarray(xs, dtype=float)
        i = np.searchsorted(self.breaks, xs, side="right") - 1
        i = np.clip(i, 0, self.n_segments - 1)
        return self.slopes[i]

    def slope_at(self, x: float) -> float:
        return float(self.slopes_at(x))

    def value_at(self, x):
        x = np.asarray(x, dtype=float)
        if self.n_segments == 0:
            return np.where(x == self.lo, self.f_lo, np.inf)
        i = np.searchsorted(self.breaks, x, side="right") - 1
        i = np.clip(i, 0, self.n_segments - 1)
        return self.cv[i] + self.slopes[i] * (x - self.breaks[i])

    def __call__(self, x):
        return self.value_at(x)

    # -- 斜率序列（用于检查单调性 / 报段数） -------------------------------- #
    def slope_span(self) -> tuple[float, float]:
        if self.n_segments == 0:
            return (0.0, 0.0)
        return (float(self.slopes[0]), float(self.slopes[-1]))


# --------------------------------------------------------------------------- #
# 基本运算
# --------------------------------------------------------------------------- #
def restrict(f: CPWL, lo: float, hi: float) -> CPWL | None:
    """把定义域截到 [lo, hi]，区间外取 +inf。不可行返回 None。"""
    lo = max(float(lo), f.lo)
    hi = min(float(hi), f.hi)
    if hi < lo - 1e-9:
        return None
    if abs(hi - lo) <= 1e-12:
        return CPWL(lo, float(np.asarray(f.value_at(lo))), [], [])
    b = f.breaks
    inner = b[(b > lo + 1e-9) & (b < hi - 1e-9)]
    bps = np.concatenate(([lo], inner, [hi]))
    widths = np.diff(bps)
    mids = 0.5 * (bps[:-1] + bps[1:])
    return CPWL(lo, float(np.asarray(f.value_at(lo))), widths, f.slopes_at(mids))


def shift(f: CPWL, s: float) -> CPWL:
    """返回 g(x) = f(x - s)。"""
    return CPWL(f.lo + s, f.f_lo, f.widths.copy(), f.slopes.copy())


def add(f: CPWL, g: CPWL) -> CPWL | None:
    """返回 f + g，定义域取交集。"""
    lo = max(f.lo, g.lo)
    hi = min(f.hi, g.hi)
    if hi < lo - 1e-9:
        return None
    if abs(hi - lo) <= 1e-12:
        v = float(np.asarray(f.value_at(lo))) + float(np.asarray(g.value_at(lo)))
        return CPWL(lo, v, [], [])
    bps_list = [lo, hi]
    for p in np.concatenate((f.breaks, g.breaks)):
        if lo + 1e-9 < p < hi - 1e-9:
            bps_list.append(float(p))
    bps = np.unique(np.asarray(bps_list, dtype=float))
    widths = np.diff(bps)
    mids = 0.5 * (bps[:-1] + bps[1:])
    f_lo = float(np.asarray(f.value_at(lo))) + float(np.asarray(g.value_at(lo)))
    return CPWL(lo, f_lo, widths, f.slopes_at(mids) + g.slopes_at(mids))


def infconv(f: CPWL, g: CPWL) -> CPWL:
    """下确界卷积 (f □ g)(z) = inf_u {f(z-u) + g(u)}。

    凸函数的下确界卷积仍是凸函数，且结果斜率集合 = 两者斜率集合的
    （按值排序的）合并，各斜率对应的定义域宽度相加。
    """
    s = np.concatenate((f.slopes, g.slopes))
    w = np.concatenate((f.widths, g.widths))
    if s.size == 0:
        return CPWL(f.lo + g.lo, f.f_lo + g.f_lo, [], [])
    order = np.argsort(s, kind="stable")
    s, w = s[order], w[order]
    keep = np.ones(s.size, dtype=bool)
    keep[1:] = np.abs(np.diff(s)) > 1e-12
    starts = np.nonzero(keep)[0]
    return CPWL(f.lo + g.lo, f.f_lo + g.f_lo, np.add.reduceat(w, starts), s[starts])


def argmin(f: CPWL, lo: float | None = None, hi: float | None = None, tol: float = 1e-12) -> float | None:
    """在 [lo, hi]（默认整个定义域）上取最左最优解。"""
    g = f if lo is None else restrict(f, lo, hi)  # type: ignore[arg-type]
    if g is None:
        return None
    if g.n_segments == 0:
        return g.lo
    idx = np.nonzero(g.slopes >= -tol)[0]
    if len(idx) == 0:
        return float(g.hi)
    return float(g.breaks[idx[0]])


# --------------------------------------------------------------------------- #
# 单时段费用函数
# --------------------------------------------------------------------------- #
def _psi(x: float, eta: float) -> float:
    return x / eta if x >= 0 else eta * x


def build_phi(c: float, n: float, eta: float, S: float) -> CPWL:
    """phi_t(x) = c * [n + psi(x)]^+，x ∈ [-S/eta, eta*S]。"""
    x_lo, x_hi = -S / eta, eta * S

    def phival(x: float) -> float:
        return c * max(n + _psi(x, eta), 0.0)

    if n > 0:
        x0 = -n / eta                       # < 0
        bps_all = [-np.inf, x0, 0.0, np.inf]
        sl = [0.0, c * eta, c / eta]
    else:
        x0 = -n * eta                       # >= 0
        bps_all = [-np.inf, x0, np.inf]
        sl = [0.0, c / eta]
    return _clip_construct(phival, bps_all, sl, x_lo, x_hi)


def build_h(c: float, n: float, eta: float, S: float) -> CPWL:
    """h_t(y) = phi_t(-y)，y ∈ [-eta*S, S/eta]（反射 + 功率限制）。"""
    y_lo, y_hi = -eta * S, S / eta

    def hval(y: float) -> float:
        return c * max(n + _psi(-y, eta), 0.0)

    if n > 0:
        y0 = n / eta                        # > 0
        bps_all = [-np.inf, 0.0, y0, np.inf]
        sl = [-c / eta, -c * eta, 0.0]
    else:
        y0 = n * eta                        # <= 0
        bps_all = [-np.inf, y0, np.inf]
        sl = [-c / eta, 0.0]
    return _clip_construct(hval, bps_all, sl, y_lo, y_hi)


def _clip_construct(val_fn, bps_all, sl, lo, hi) -> CPWL:
    """把 (-inf, +inf) 上的分段折线按断点/斜率裁剪到 [lo, hi]。"""
    finite = [b for b in bps_all if np.isfinite(b) and lo + 1e-9 < b < hi - 1e-9]
    bps = np.array(sorted({lo, hi, *finite}))
    widths = np.diff(bps)
    mids = 0.5 * (bps[:-1] + bps[1:])
    slopes = []
    for m in mids:
        k = len(sl) - 1
        for i in range(len(sl)):
            if m < bps_all[i + 1] or not np.isfinite(bps_all[i + 1]):
                k = i
                break
        slopes.append(sl[k])
    return CPWL(lo, val_fn(lo), widths, slopes)


# --------------------------------------------------------------------------- #
# 连续状态 DP
# --------------------------------------------------------------------------- #
@dataclass
class TwoLayerResult:
    cost: float
    buy_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    energy_kwh: np.ndarray          # 长度 T+1，energy[0]=E0
    surplus_kwh: np.ndarray
    F: list[CPWL]                   # F[0..T]，F[t] 为进入时段 t 的价值函数
    max_segments: int
    total_segments: int


def two_layer_dp(price, load_kwh, pv_kwh, prm: dict) -> TwoLayerResult:
    price = np.asarray(price, dtype=float)
    net = np.asarray(load_kwh, dtype=float) - np.asarray(pv_kwh, dtype=float)
    T = len(price)
    eta = float(prm["charge_efficiency"])
    S = float(prm["max_charge_discharge_power_kw"]) * float(prm["interval_hours"])
    e_min = float(prm["operating_min_kwh"])
    e_max = float(prm["operating_max_kwh"])
    e0 = float(prm["initial_energy_kwh"])
    eT = float(prm["terminal_energy_kwh"])

    # ---- 反向递推：F[T] 为终端点函数，F[t] = iota + (F[t+1] □ h_t) ------- #
    F: list[CPWL | None] = [None] * (T + 1)
    F[T] = CPWL(eT, 0.0, [], [])
    max_seg = 0
    total_seg = 0
    for t in range(T - 1, -1, -1):
        h = build_h(price[t], net[t], eta, S)
        g = infconv(F[t + 1], h)                        # type: ignore[arg-type]
        ft = restrict(g, e_min, e_max)
        if ft is None:
            raise RuntimeError(f"时段 {t} 的定义域为空，模型不可行")
        F[t] = ft
        max_seg = max(max_seg, ft.n_segments)
        total_seg += ft.n_segments

    cost = float(np.asarray(F[0].value_at(e0)))          # type: ignore[union-attr]

    # ---- 正向回代：逐时段取 argmin(phi_t(x) + F_{t+1}(e+x)) -------------- #
    buy = np.zeros(T)
    charge = np.zeros(T)
    discharge = np.zeros(T)
    surplus = np.zeros(T)
    energy = np.zeros(T + 1)
    energy[0] = e0
    e = e0
    for t in range(T):
        x_lo = max(-S / eta, e_min - e)
        x_hi = min(eta * S, e_max - e)
        phi = build_phi(price[t], net[t], eta, S)
        f_next = shift(F[t + 1], -e)                     # type: ignore[arg-type]
        p = add(phi, f_next)
        if p is None:
            raise RuntimeError(f"时段 {t} 无可行动作")
        x = argmin(p, x_lo, x_hi)
        if x is None:
            raise RuntimeError(f"时段 {t} argmin 为空")
        e_next = e + x
        if x >= 0:
            charge[t] = x / eta
        else:
            discharge[t] = -eta * x
        val = net[t] + _psi(x, eta)
        buy[t] = max(val, 0.0)
        surplus[t] = max(-val, 0.0)
        energy[t + 1] = e_next
        e = e_next

    return TwoLayerResult(
        cost=cost,
        buy_kwh=buy,
        charge_kwh=charge,
        discharge_kwh=discharge,
        energy_kwh=energy,
        surplus_kwh=surplus,
        F=[f for f in F],                                # type: ignore[misc]
        max_segments=max_seg,
        total_segments=total_seg,
    )


# --------------------------------------------------------------------------- #
# 边际储能价值（F_t 的负斜率）与内部价格
# --------------------------------------------------------------------------- #
def marginal_value(F: CPWL, e: float, tol: float = 1e-9) -> tuple[float, float]:
    """返回 e 处次微分取负后的区间 (Lambda_left, Lambda_right)。

    凸分段线性函数在段内部可微，左右斜率相同；在断点处左斜率 <= 右斜率，
    取负后满足 Lambda_left >= Lambda_right（命题 3.1 的单调非增）。
    """
    if F.n_segments == 0:
        raise ValueError("点函数没有边际价值")
    b = F.breaks
    j = int(np.searchsorted(b, e, side="right") - 1)
    j = max(0, min(j, F.n_segments - 1))
    lam_right = -float(F.slopes[j])
    if abs(e - b[j]) <= tol and j > 0:
        return -float(F.slopes[j - 1]), lam_right
    return lam_right, lam_right
