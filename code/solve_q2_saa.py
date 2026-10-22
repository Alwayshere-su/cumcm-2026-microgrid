from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack
from solve_q2 import execute_day, plan_day

ROOT=Path(__file__).resolve().parents[1]


def historical_scenarios(forecasts: pd.DataFrame, day: pd.Timestamp, n_scenarios: int,
                         seed: int=2026, lookback_days: int=90,
                         load_col: str="lgb_load_kwh", pv_col: str="lgb_pv_kwh") -> tuple[np.ndarray,list[pd.Timestamp]]:
    f=forecasts.copy(); f["date"]=pd.to_datetime(f.date); day=pd.Timestamp(day)
    past=f[(f.date<day)&(f.date>=day-pd.Timedelta(days=lookback_days))].copy()
    if past.date.nunique()<2: raise ValueError("at least two historical residual days are required")
    past["residual"]=(past.actual_load_kwh-past.actual_pv_kwh)-(past[load_col]-past[pv_col])
    wide=past.pivot(index="date",columns="slot",values="residual").dropna()
    same=wide[[pd.Timestamp(d).weekday()==day.weekday() for d in wide.index]]
    pool=same if len(same)>=max(3,min(n_scenarios,7)) else wide
    rng=np.random.default_rng(seed); chosen=rng.choice(len(pool),size=n_scenarios,replace=len(pool)<n_scenarios)
    return pool.iloc[chosen].to_numpy(float),[pd.Timestamp(x) for x in pool.index[chosen]]


def solve_saa_plan(net_scenarios: np.ndarray, price: np.ndarray, energy0: float, prm: dict,
                   risk_weight: float=.25, alpha: float=.95) -> dict[str,np.ndarray|float]:
    net=np.asarray(net_scenarios,float); price=np.asarray(price,float)
    S,T=net.shape
    if price.shape!=(T,): raise ValueError("price length differs from scenario horizon")
    if not (0<=risk_weight<=1 and 0<alpha<1): raise ValueError("invalid CVaR parameters")
    # x[T], then per scenario c,d,w,g,E[T+1], followed by nu and u[S].
    x0=0; per=5*T+1; rec0=T; nu_ix=T+S*per; u0=nu_ix+1; n=u0+S
    def base(s): return rec0+s*per
    def rec(s,k,t): return base(s)+{"c":0,"d":T,"w":2*T,"g":3*T,"e":4*T}[k]+t
    obj=np.zeros(n); obj[x0:x0+T]=price
    for s in range(S):
        obj[[rec(s,"g",t) for t in range(T)]]=(1-risk_weight)*5*price/S
        obj[[rec(s,"c",t) for t in range(T)]]=1e-9/S
        obj[[rec(s,"d",t) for t in range(T)]]=1e-9/S
        obj[[rec(s,"w",t) for t in range(T)]]=1e-9/S
    obj[nu_ix]=risk_weight; obj[u0:u0+S]=risk_weight/((1-alpha)*S)
    Aeq=lil_matrix((2*S*T,n)); beq=np.zeros(2*S*T)
    eta_c,eta_d=prm["charge_efficiency"],prm["discharge_efficiency"]
    for s in range(S):
        for t in range(T):
            r=2*(s*T+t); Aeq[r,x0+t]=1; Aeq[r,rec(s,"d",t)]=1; Aeq[r,rec(s,"g",t)]=1
            Aeq[r,rec(s,"c",t)]=-1; Aeq[r,rec(s,"w",t)]=-1; beq[r]=net[s,t]
            r+=1; Aeq[r,rec(s,"e",t+1)]=1; Aeq[r,rec(s,"e",t)]=-1
            Aeq[r,rec(s,"c",t)]=-eta_c; Aeq[r,rec(s,"d",t)]=1/eta_d
    # scenario total cost - nu - u_s <= 0
    Aub=lil_matrix((S,n)); bub=np.zeros(S)
    for s in range(S):
        Aub[s,x0:x0+T]=price; Aub[s,[rec(s,"g",t) for t in range(T)]]=5*price
        Aub[s,nu_ix]=-1; Aub[s,u0+s]=-1
    lo=np.zeros(n); hi=np.full(n,np.inf); lo[nu_ix]=-np.inf
    block=prm["max_charge_discharge_power_kw"]*prm["interval_hours"]
    for s in range(S):
        hi[[rec(s,k,t) for k in ("c","d") for t in range(T)]]=block
        es=np.array([rec(s,"e",t) for t in range(T+1)])
        lo[es]=prm["operating_min_kwh"]; hi[es]=prm["operating_max_kwh"]
        lo[es[0]]=hi[es[0]]=energy0; lo[es[-1]]=hi[es[-1]]=energy0
    ans=milp(obj,bounds=Bounds(lo,hi),constraints=[LinearConstraint(Aeq.tocsr(),beq,beq),LinearConstraint(Aub.tocsr(),-np.inf,bub)],
             options={"presolve":True,"time_limit":60})
    if not ans.success: raise RuntimeError(f"SAA LP failed: {ans.message}")
    z=ans.x; out={"buy":z[:T],"nu":float(z[nu_ix]),"u":z[u0:u0+S],"objective":float(ans.fun)}
    for k in ("c","d","w","g"):
        out[{"c":"charge","d":"discharge","w":"surplus","g":"emergency"}[k]]=np.vstack([[z[rec(s,k,t)] for t in range(T)] for s in range(S)])
    out["energy"]=np.vstack([[z[rec(s,"e",t)] for t in range(T+1)] for s in range(S)])
    return out


def run_saa_day(data: pd.DataFrame, forecasts: pd.DataFrame, day: pd.Timestamp, energy0: float,
                n_scenarios: int=20, risk_weight: float=.25, alpha: float=.95,
                prm: dict|None=None, load_col: str="lgb_load_kwh",
                pv_col: str="lgb_pv_kwh") -> tuple[pd.DataFrame,dict]:
    if prm is None:
        import json
        prm=json.loads((ROOT/"data_processed"/"q1_storage_parameters.json").read_text(encoding="utf-8"))
    day=pd.Timestamp(day); actual=data[data.date==day].sort_values("slot")
    f=forecasts[forecasts.date==day].sort_values("slot")
    residuals,used=historical_scenarios(forecasts,day,n_scenarios,seed=2026+day.dayofyear,
                                        load_col=load_col,pv_col=pv_col)
    center=(f[load_col]-f[pv_col]).to_numpy(float)
    scenarios=np.maximum(center[None,:]+residuals,-np.maximum(f[pv_col].to_numpy(),0)[None,:])
    price=actual.price_fixed.to_numpy(float)
    saa=solve_saa_plan(scenarios,price,energy0,prm,risk_weight,alpha)
    plan={"buy":saa["buy"]}
    run=execute_day(plan,actual.load_kwh.to_numpy(),actual.pv_kwh.to_numpy(),price,energy0,prm)
    run.insert(0,"date",day); run["load_kwh"]=actual.load_kwh.to_numpy(); run["pv_kwh"]=actual.pv_kwh.to_numpy()
    run["load_forecast_kwh"]=f[load_col].to_numpy(); run["pv_forecast_kwh"]=f[pv_col].to_numpy()
    return run,{"scenario_count":n_scenarios,"risk_weight":risk_weight,"alpha":alpha,
                "scenario_dates":[str(x.date()) for x in used],"saa_objective":saa["objective"]}


def run_saa_period(data: pd.DataFrame, forecasts: pd.DataFrame, dates, energy0: float,
                   n_scenarios: int=10, risk_weight: float=.1, alpha: float=.95,
                   prm: dict|None=None, load_col: str="lgb_load_kwh",
                   pv_col: str="lgb_pv_kwh") -> tuple[pd.DataFrame,dict]:
    if prm is None:
        import json
        prm=json.loads((ROOT/"data_processed"/"q1_storage_parameters.json").read_text(encoding="utf-8"))
    rows=[]; fallback=0; e=float(energy0)
    for day in pd.to_datetime(list(dates)):
        try:
            run,_=run_saa_day(data,forecasts,day,e,n_scenarios,risk_weight,alpha,prm,load_col,pv_col)
        except ValueError as exc:
            if "historical residual" not in str(exc): raise
            actual=data[data.date==day].sort_values("slot"); f=forecasts[forecasts.date==day].sort_values("slot")
            price=actual.price_fixed.to_numpy(); p=plan_day(f[load_col].to_numpy(),f[pv_col].to_numpy(),price,e,prm)
            run=execute_day(p,actual.load_kwh.to_numpy(),actual.pv_kwh.to_numpy(),price,e,prm)
            run.insert(0,"date",day); run["load_kwh"]=actual.load_kwh.to_numpy(); run["pv_kwh"]=actual.pv_kwh.to_numpy()
            run["load_forecast_kwh"]=f[load_col].to_numpy(); run["pv_forecast_kwh"]=f[pv_col].to_numpy(); fallback+=1
        rows.append(run); e=float(run.energy_end_kwh.iloc[-1])
    out=pd.concat(rows,ignore_index=True)
    summary={"model":"joint residual SAA-CVaR + causal execution","forecast_columns":[load_col,pv_col],"scenario_count":n_scenarios,
             "risk_weight":risk_weight,"alpha":alpha,"days":len(pd.unique(out.date)),"fallback_days":fallback,
             "total_plan_purchase_kwh":float(out.plan_buy_kwh.sum()),"total_emergency_kwh":float(out.emergency_kwh.sum()),
             "plan_cost_yuan":float(out.plan_cost_yuan.sum()),"emergency_cost_yuan":float(out.emergency_cost_yuan.sum()),
             "total_cost_yuan":float(out.plan_cost_yuan.sum()+out.emergency_cost_yuan.sum()),
             "emergency_days":int((out.groupby("date").emergency_kwh.sum()>1e-6).sum()),
             "emergency_intervals":int((out.emergency_kwh>1e-6).sum()),
             "min_soc_kwh":float(out[["energy_start_kwh","energy_end_kwh"]].min().min()),
             "max_soc_kwh":float(out[["energy_start_kwh","energy_end_kwh"]].max().max()),
             "initial_soc_kwh":float(out.energy_start_kwh.iloc[0]),"terminal_soc_kwh":float(out.energy_end_kwh.iloc[-1])}
    return out,summary
