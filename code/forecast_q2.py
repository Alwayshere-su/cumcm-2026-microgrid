from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from solve_q2 import DATA, RESULTS, seasonal_naive


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
OUT = ROOT / "data_processed" / "q2_forecasts.csv"


def feature_panel(data: pd.DataFrame) -> pd.DataFrame:
    dates = pd.Index(sorted(data.date.unique()), name="date")
    frames=[]
    for target in ("load_kwh","pv_kwh"):
        wide=data.pivot(index="date",columns="slot",values=target).reindex(dates)
        base=wide.stack(future_stack=True).rename(target).reset_index()
        for lag in (1,2,7,14):
            base[f"lag{lag}"]=wide.shift(lag).stack(future_stack=True).to_numpy()
        for win in (3,7,14):
            base[f"mean{win}"]=wide.shift(1).rolling(win,min_periods=2).mean().stack(future_stack=True).to_numpy()
            base[f"std{win}"]=wide.shift(1).rolling(win,min_periods=2).std().stack(future_stack=True).to_numpy()
        base["target"]=target; frames.append(base)
    panel=pd.concat(frames,ignore_index=True)
    panel["date"]=pd.to_datetime(panel.date); panel["dow"]=panel.date.dt.weekday
    panel["month"]=panel.date.dt.month; panel["doy"]=panel.date.dt.dayofyear
    panel["tod_sin"]=np.sin(2*np.pi*(panel.slot-1)/144); panel["tod_cos"]=np.cos(2*np.pi*(panel.slot-1)/144)
    panel["doy_sin"]=np.sin(2*np.pi*panel.doy/365); panel["doy_cos"]=np.cos(2*np.pi*panel.doy/365)
    return panel


FEATURES=["slot","dow","month","tod_sin","tod_cos","doy_sin","doy_cos",
          "lag1","lag2","lag7","lag14","mean3","std3","mean7","std7","mean14","std14"]

def causal_conformal(df: pd.DataFrame, actual_col: str, lo_col: str, hi_col: str,
                     window_days: int=28, min_days: int=7) -> pd.DataFrame:
    out=df.copy(); out["p10_cal"]=out[lo_col]; out["p90_cal"]=out[hi_col]
    dates=sorted(pd.to_datetime(out.date).unique())
    for day in dates:
        hist=out[(out.date<pd.Timestamp(day))&(out.date>=pd.Timestamp(day)-pd.Timedelta(days=window_days))]
        if hist.date.nunique()<min_days: continue
        score=np.maximum(hist[lo_col]-hist[actual_col],hist[actual_col]-hist[hi_col]).clip(lower=0)
        correction=float(np.quantile(score,.8,method="higher"))
        mask=out.date==pd.Timestamp(day); out.loc[mask,"p10_cal"]=np.maximum(0,out.loc[mask,lo_col]-correction)
        out.loc[mask,"p90_cal"]=out.loc[mask,hi_col]+correction
    return out


def walk_forward() -> tuple[pd.DataFrame,dict]:
    data=pd.read_csv(DATA,parse_dates=["date"]); panel=feature_panel(data)
    test_dates=pd.date_range("2025-02-01","2025-12-31")
    records=[]; models={}
    for i,day in enumerate(test_dates):
        # Weekly refit; every fit only sees labels strictly before the forecast day.
        if i%7==0:
            for target in ("load_kwh","pv_kwh"):
                tr=panel[(panel.target==target)&(panel.date<day)].dropna(subset=FEATURES+[target])
                for alpha in (.1,.5,.9):
                    model=lgb.LGBMRegressor(objective="quantile",alpha=alpha,n_estimators=220,
                        learning_rate=.04,num_leaves=25,min_child_samples=40,subsample=.85,
                        colsample_bytree=.85,reg_lambda=.2,random_state=2026,verbosity=-1,n_jobs=-1)
                    model.fit(tr[FEATURES],tr[target]); models[target,alpha]=model
        daily=panel[panel.date==day].sort_values(["target","slot"])
        actual=data[data.date==day].sort_values("slot")
        naive_l,naive_p=seasonal_naive(data,day)
        row={"date":day,"slot":np.arange(1,145),"naive_load_kwh":naive_l,"naive_pv_kwh":naive_p}
        result=pd.DataFrame(row)
        for target,prefix in (("load_kwh","load"),("pv_kwh","pv")):
            x=daily[daily.target==target].sort_values("slot")
            preds=[]
            for alpha in (.1,.5,.9): preds.append(np.maximum(models[target,alpha].predict(x[FEATURES]),0))
            preds=np.vstack(preds)
            # Preserve the independently trained median; only expand outer bounds to remove crossing.
            preds=np.vstack([np.minimum(preds[0],preds[1]),preds[1],np.maximum(preds[2],preds[1])])
            if target=="pv_kwh":
                hist=data[data.date<day]; cap=hist.groupby("slot").pv_kwh.quantile(.995).to_numpy(); preds=np.minimum(preds,cap)
            result[f"lgb_{prefix}_p10_kwh"],result[f"lgb_{prefix}_kwh"],result[f"lgb_{prefix}_p90_kwh"]=preds
        result["actual_load_kwh"]=actual.load_kwh.to_numpy(); result["actual_pv_kwh"]=actual.pv_kwh.to_numpy()
        records.append(result)
    out=pd.concat(records,ignore_index=True)
    for prefix in ("load","pv"):
        c=causal_conformal(out,"actual_"+prefix+"_kwh",f"lgb_{prefix}_p10_kwh",f"lgb_{prefix}_p90_kwh")
        out[f"lgb_{prefix}_p10_cal_kwh"]=c.p10_cal; out[f"lgb_{prefix}_p90_cal_kwh"]=c.p90_cal
    metrics={}
    for name in ("naive","lgb"):
        metrics[name]={"load_mae_kw":float(np.mean(np.abs(out[f"{name}_load_kwh"]-out.actual_load_kwh))*6),
                       "pv_mae_kw":float(np.mean(np.abs(out[f"{name}_pv_kwh"]-out.actual_pv_kwh))*6),
                       "net_mae_kw":float(np.mean(np.abs((out[f"{name}_load_kwh"]-out[f"{name}_pv_kwh"])-(out.actual_load_kwh-out.actual_pv_kwh)))*6)}
    for prefix in ("load","pv"):
        inside=(out[f"actual_{prefix}_kwh"]>=out[f"lgb_{prefix}_p10_cal_kwh"])&(out[f"actual_{prefix}_kwh"]<=out[f"lgb_{prefix}_p90_cal_kwh"])
        width=out[f"lgb_{prefix}_p90_cal_kwh"]-out[f"lgb_{prefix}_p10_cal_kwh"]
        metrics["lgb"][f"{prefix}_picp80"]=float(inside.mean())
        metrics["lgb"][f"{prefix}_pinaw80"]=float(width.mean()/max(out[f"actual_{prefix}_kwh"].max()-out[f"actual_{prefix}_kwh"].min(),1e-9))
    return out,metrics


def main():
    out,metrics=walk_forward(); out.to_csv(OUT,index=False,encoding="utf-8-sig")
    (RESULTS/"q2_forecast_comparison.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(metrics,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
