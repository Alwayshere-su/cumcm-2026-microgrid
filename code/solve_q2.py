from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
DATA = ROOT / "data_processed" / "microgrid_10min.csv"
PARAMS = ROOT / "data_processed" / "q1_storage_parameters.json"
TEMPLATE = ROOT / "附件" / "附件5" / "result2.xlsx"
OUTPUT = ROOT / "result2.xlsx"
RESULTS = ROOT / "results"


def seasonal_naive(history: pd.DataFrame, day: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """Only use observations strictly before day; return 144-point load/PV energy forecasts."""
    h = history[history["date"] < day]
    if h.empty:
        raise ValueError(f"no history before {day.date()}")
    by_day = {d: x.sort_values("slot") for d, x in h.groupby("date")}
    candidates, weights = [], []
    for lag, weight in ((1, .50), (7, .30)):
        d = day - pd.Timedelta(days=lag)
        if d in by_day:
            candidates.append(by_day[d][["load_kwh", "pv_kwh"]].to_numpy(float)); weights.append(weight)
    same_type = h[(h["date"].dt.weekday >= 5) == (day.weekday() >= 5)]
    recent_dates = sorted(same_type["date"].unique())[-4:]
    if len(recent_dates):
        curve = (same_type[same_type["date"].isin(recent_dates)]
                 .groupby("slot")[["load_kwh", "pv_kwh"]].mean().to_numpy())
        candidates.append(curve); weights.append(.20)
    weights = np.asarray(weights, float); weights /= weights.sum()
    pred = sum(w * x for w, x in zip(weights, candidates))
    # Physical clipping based exclusively on historical observations.
    load_cap = h.groupby("slot")["load_kwh"].quantile(.995).to_numpy()
    pv_cap = h.groupby("slot")["pv_kwh"].quantile(.995).to_numpy()
    pred[:, 0] = np.clip(pred[:, 0], 0, load_cap)
    pred[:, 1] = np.clip(pred[:, 1], 0, pv_cap)
    return pred[:, 0], pred[:, 1]


def plan_day(load_hat: np.ndarray, pv_hat: np.ndarray, price: np.ndarray,
             energy0: float, prm: dict) -> dict[str, np.ndarray]:
    T = 144; block = prm["max_charge_discharge_power_kw"] * prm["interval_hours"]
    # buy, charge, discharge, surplus, E[0..T]
    offsets = {"buy": 0, "charge": T, "discharge": 2*T, "surplus": 3*T, "energy": 4*T}
    n = 5*T + 1
    def ix(name, t): return offsets[name] + t
    c = np.zeros(n); c[:T] = price
    c[T:4*T] += 1e-8  # deterministic tie-break against needless cycling/surplus
    A = lil_matrix((2*T, n)); b = np.zeros(2*T)
    for t in range(T):
        A[t, ix("buy", t)] = 1; A[t, ix("discharge", t)] = 1
        A[t, ix("charge", t)] = -1; A[t, ix("surplus", t)] = -1
        b[t] = load_hat[t] - pv_hat[t]
        A[T+t, ix("energy", t+1)] = 1; A[T+t, ix("energy", t)] = -1
        A[T+t, ix("charge", t)] = -prm["charge_efficiency"]
        A[T+t, ix("discharge", t)] = 1/prm["discharge_efficiency"]
    lo = np.zeros(n); hi = np.full(n, np.inf)
    hi[T:3*T] = block
    lo[4*T:] = prm["operating_min_kwh"]; hi[4*T:] = prm["operating_max_kwh"]
    lo[ix("energy", 0)] = hi[ix("energy", 0)] = energy0
    lo[ix("energy", T)] = hi[ix("energy", T)] = energy0
    sol = milp(c, bounds=Bounds(lo, hi), constraints=LinearConstraint(A.tocsr(), b, b),
        options={"presolve": True, "time_limit": 30})
    if not sol.success:
        raise RuntimeError(f"daily LP failed: {sol.message}")
    x = sol.x
    return {k: x[o:o+(T+1 if k == "energy" else T)] for k, o in offsets.items()}


def execute_day(plan: dict[str, np.ndarray], load: np.ndarray, pv: np.ndarray,
                price: np.ndarray, energy0: float, prm: dict) -> pd.DataFrame:
    T = 144; block = prm["max_charge_discharge_power_kw"] * prm["interval_hours"]
    eta_c, eta_d = prm["charge_efficiency"], prm["discharge_efficiency"]
    emin, emax = prm["operating_min_kwh"], prm["operating_max_kwh"]
    rows, e = [], energy0
    for t in range(T):
        start = e; balance = plan["buy"][t] + pv[t] - load[t]
        if balance >= 0:
            charge = min(balance, block, (emax-e)/eta_c); discharge = emergency = 0.0
            surplus = balance-charge; e += eta_c*charge
        else:
            discharge = min(-balance, block, (e-emin)*eta_d); charge = surplus = 0.0
            emergency = -balance-discharge; e -= discharge/eta_d
        rows.append((t+1, plan["buy"][t], charge, discharge, surplus, emergency,
                     start, e, price[t]*plan["buy"][t], 5*price[t]*emergency))
    cols = ["slot","plan_buy_kwh","charge_kwh","discharge_kwh","surplus_kwh",
            "emergency_kwh","energy_start_kwh","energy_end_kwh","plan_cost_yuan","emergency_cost_yuan"]
    return pd.DataFrame(rows, columns=cols)


def solve_q2(forecast_model: str = "naive") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = pd.read_csv(DATA, parse_dates=["date"])
    forecast_path = ROOT / "data_processed" / "q2_forecasts.csv"
    forecasts = pd.read_csv(forecast_path, parse_dates=["date"]) if forecast_model == "lgb" else None
    prm = json.loads(PARAMS.read_text(encoding="utf-8"))
    dates = pd.date_range("2025-02-01", "2025-12-31", freq="D")
    energy = prm["initial_energy_kwh"]; dispatch, metrics = [], []
    for day in dates:
        actual = data[data.date == day].sort_values("slot")
        if forecasts is None:
            lh, ph = seasonal_naive(data, day)
        else:
            f = forecasts[forecasts.date == day].sort_values("slot")
            if len(f) != 144: raise ValueError(f"missing LGB forecast for {day.date()}")
            lh, ph = f.lgb_load_kwh.to_numpy(), f.lgb_pv_kwh.to_numpy()
        price = actual["price_fixed"].to_numpy(float)
        plan = plan_day(lh, ph, price, energy, prm)
        run = execute_day(plan, actual.load_kwh.to_numpy(), actual.pv_kwh.to_numpy(), price, energy, prm)
        run.insert(0, "date", day); run["load_kwh"] = actual.load_kwh.to_numpy(); run["pv_kwh"] = actual.pv_kwh.to_numpy()
        run["load_forecast_kwh"] = lh; run["pv_forecast_kwh"] = ph; dispatch.append(run)
        energy = float(run.energy_end_kwh.iloc[-1])
        metrics.append({"date": day, "load_mae_kwh": float(np.mean(np.abs(lh-actual.load_kwh))),
                        "pv_mae_kwh": float(np.mean(np.abs(ph-actual.pv_kwh)))})
    out = pd.concat(dispatch, ignore_index=True); pred = pd.DataFrame(metrics)
    bal = out.plan_buy_kwh + out.pv_kwh + out.discharge_kwh + out.emergency_kwh - out.load_kwh - out.charge_kwh - out.surplus_kwh
    soc = out.energy_end_kwh-out.energy_start_kwh-prm["charge_efficiency"]*out.charge_kwh+out.discharge_kwh/prm["discharge_efficiency"]
    summary = {"model": f"{forecast_model} P50 + deterministic LP + causal execution",
               "days": len(dates), "total_plan_purchase_kwh": float(out.plan_buy_kwh.sum()),
               "total_emergency_kwh": float(out.emergency_kwh.sum()),
               "plan_cost_yuan": float(out.plan_cost_yuan.sum()),
               "emergency_cost_yuan": float(out.emergency_cost_yuan.sum()),
               "total_cost_yuan": float(out.plan_cost_yuan.sum()+out.emergency_cost_yuan.sum()),
               "emergency_days": int((out.groupby("date").emergency_kwh.sum()>1e-6).sum()),
               "emergency_intervals": int((out.emergency_kwh>1e-6).sum()),
               "load_mae_kw": float(pred.load_mae_kwh.mean()*6), "pv_mae_kw": float(pred.pv_mae_kwh.mean()*6),
               "min_soc_kwh": float(out[["energy_start_kwh","energy_end_kwh"]].min().min()),
               "max_soc_kwh": float(out[["energy_start_kwh","energy_end_kwh"]].max().max()),
               "initial_soc_kwh": float(out.energy_start_kwh.iloc[0]), "terminal_soc_kwh": float(out.energy_end_kwh.iloc[-1]),
               "max_balance_residual_kwh": float(np.abs(bal).max()), "max_soc_residual_kwh": float(np.abs(soc).max()),
               "max_simultaneous_charge_discharge_kwh": float(np.minimum(out.charge_kwh,out.discharge_kwh).max())}
    return out, pred, summary


def _ranges(day: pd.DataFrame) -> list[tuple[str, float]]:
    active = day.emergency_kwh.to_numpy() > 1e-6; vals = day.emergency_kwh.to_numpy(); ans=[]; i=0
    while i < 144:
        if not active[i]: i += 1; continue
        j=i
        while j+1 < 144 and active[j+1]: j += 1
        def hm(k):
            m=k*10; return f"{m//60}:{m%60:02d}" if m < 1440 else f"0:{m%60:02d}+1"
        ans.append((f"{hm(i)}-{hm(j+1)}", float(vals[i:j+1].sum()))); i=j+1
    return ans


def write_outputs(out: pd.DataFrame, pred: pd.DataFrame, summary: dict) -> None:
    """Legacy deterministic-baseline writer; never call for official Q2 outputs."""
    RESULTS.mkdir(exist_ok=True); out.to_csv(RESULTS/"q2_dispatch_10min.csv",index=False,encoding="utf-8-sig")
    pred.to_csv(RESULTS/"q2_prediction_metrics_daily.csv",index=False,encoding="utf-8-sig")
    (RESULTS/"q2_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")


def write_template(out: pd.DataFrame) -> None:
    """Write result2.xlsx (the problem-2 deliverable) from a dispatch table.

    Owned exclusively by the method-3 main model (solve_q2_method3.py); legacy
    deterministic and SAA experiments must not overwrite the deliverable.
    """
    shutil.copy2(TEMPLATE, OUTPUT); wb=load_workbook(OUTPUT); ws=wb["计划购电量"]
    days=sorted(out.date.unique()); source_slots=list(range(2,145))+[1]
    for row, day in enumerate(days,start=2):
        x=out[out.date==day].set_index("slot")
        for col,slot in enumerate(source_slots,start=2): ws.cell(row,col,round(float(x.loc[slot,"plan_buy_kwh"]),6))
        ws.cell(row,146,round(float(x.plan_buy_kwh.sum()),6)); ws.cell(row,147,round(float(x.plan_cost_yuan.sum()),6))
    # The storage sheet requests the four official representative dates, six 4-hour blocks each.
    sw=wb["充放电量"]
    # Template uses first-two-days/ellipsis/last-day as a structural example; expand to all 334 days.
    if sw.max_row > 1: sw.delete_rows(2, sw.max_row-1)
    row=2
    for d in days:
        x=out[out.date==d].copy(); x["bin"]=(x.slot-1)//24
        g=x.groupby("bin").agg(charge=("charge_kwh","sum"),discharge=("discharge_kwh","sum"))
        for b in range(6):
            sw.cell(row,1,d.to_pydatetime() if b==0 else None)
            sw.cell(row,2,f"{4*b}:00-{4*(b+1)}:00")
            sw.cell(row,3,round(float(g.loc[b,"charge"]),6)); sw.cell(row,4,round(float(g.loc[b,"discharge"]),6))
            if b==0: sw.cell(row,5,"0:00"); sw.cell(row,6,round(float(x.energy_start_kwh.iloc[0]),6))
            if b==1: sw.cell(row,5,"24:00"); sw.cell(row,6,round(float(x.energy_end_kwh.iloc[-1]),6))
            row+=1
    ew=wb["紧急购电量"]
    for cells in ew.iter_rows(min_row=2):
        for cell in cells: cell.value=None
    row=2
    for d in days:
        ranges=_ranges(out[out.date==d])
        for k,(interval,amount) in enumerate(ranges):
            ew.cell(row,1,pd.Timestamp(d).to_pydatetime() if k==0 else None)
            ew.cell(row,2,interval); ew.cell(row,3,round(amount,6)); row+=1
    wb.save(OUTPUT)


def main() -> None:
    from solve_q2_method3 import main as method3_main
    method3_main()


if __name__ == "__main__": main()
