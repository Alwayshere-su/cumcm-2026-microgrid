"""问题三最终模型候选图：raw/process/result 各三张。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_script_path, add_style_path  # noqa: E402
ROOT = find_project_root(__file__)
FIG, RES = ROOT / "figures", ROOT / "results"
add_style_path()
add_script_path()
from plot_style import apply_publication_style, export_figure
from ablation_q3_forecast import build_library
from solve_q3_final import load_inputs

REPS = pd.to_datetime(["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"])


def main():
    FIG.mkdir(exist_ok=True)
    style = apply_publication_style(journal="nature", lang="zh")
    c = style["colors"]
    out = pd.read_csv(RES / "q3_final_dispatch.csv", parse_dates=["date"])
    voi = pd.read_csv(RES / "q3_voi_daily.csv", parse_dates=["date"])
    policy = pd.read_csv(RES / "q3_policy_summary.csv")
    data, official, stat, ridge, _ = load_inputs()
    lib = build_library("ridge", "fused", data, official, stat, ridge, verbose=False)

    fig, axs = plt.subplots(4, 1, figsize=(7.1, 7.0), sharex=True)
    for ax, day in zip(axs, REPS):
        x = out[out.date == day]; h = (x.slot - 1) / 6
        ax.plot(h, x.load_kwh*6, color=c[0], lw=1, label="实际负荷")
        ax.plot(h, x.pv_kwh*6, color=c[1], lw=1, ls="--", label="实际光伏")
        ax.text(.01, .80, day.strftime("%m-%d"), transform=ax.transAxes); ax.set_ylabel("功率/kW")
    axs[0].legend(frameon=False, ncol=2); axs[-1].set_xlabel("时刻/h")
    export_figure(fig, FIG / "raw_q3_representative_source_load", dpi=300)

    fig, axs = plt.subplots(4, 1, figsize=(7.1, 7.0), sharex=True)
    for ax, day in zip(axs, REPS):
        x = lib["by_day"][day]; h = (x.slot.to_numpy()-1)/6
        actual = (x.load_kwh.to_numpy()-x.pv_kwh.to_numpy())*6
        rp = ridge[ridge.date == day].sort_values("slot").lgb_pv_kwh.to_numpy()
        reg = (lib["load_hat"][day]-rp)*6
        fused = (lib["load_hat"][day]-lib["pv_hat"][(day,0)])*6
        ax.plot(h, actual, color="black", lw=.9, label="实际净负荷")
        ax.plot(h, reg, color=c[2], lw=.8, ls=":", label="Ridge")
        ax.plot(h, fused, color=c[0], lw=.9, ls="--", label="校正融合")
        ax.text(.01, .80, day.strftime("%m-%d"), transform=ax.transAxes); ax.set_ylabel("功率/kW")
    axs[0].legend(frameon=False, ncol=3); axs[-1].set_xlabel("时刻/h")
    export_figure(fig, FIG / "raw_q3_ridge_fusion_forecast", dpi=300)

    names = ["附件3裸值", "Ridge光伏", "校正融合", "融合负荷", "融合净负荷"]
    vals = [386.52, 309.70, 262.69, 131.80, 218.70]
    fig, ax = plt.subplots(figsize=(6.6, 3.5)); ax.bar(names, vals, color=[c[3],c[2],c[0],c[4],c[1]])
    ax.set_ylabel("样本外MAE/kW"); ax.tick_params(axis="x", rotation=18)
    export_figure(fig, FIG / "raw_q3_forecast_accuracy", dpi=300)

    mat = out.assign(adj=lambda z:(z.buy_kwh-z.original_plan_kwh)*6).pivot(index="date",columns="slot",values="adj").to_numpy()
    lim = np.quantile(np.abs(mat), .98)
    fig, ax = plt.subplots(figsize=(7.1,4.1)); im=ax.imshow(mat,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim,extent=[0,24,12,1])
    ax.set(xlabel="时刻/h",ylabel="月份",yticks=range(1,13)); cb=fig.colorbar(im,ax=ax,pad=.02); cb.set_label("调整功率/kW")
    export_figure(fig, FIG / "process_q3_adjustment_heatmap", dpi=300)

    x=out[out.date==pd.Timestamp("2025-03-20")]; h=(x.slot-1)/6
    fig,ax=plt.subplots(figsize=(7.1,3.6)); ax.step(h,x.original_plan_kwh*6,where="post",color=c[2],ls="--",label="0:00计划"); ax.step(h,x.buy_kwh*6,where="post",color=c[0],label="执行前最后确认量")
    for k in (6,12,18): ax.axvline(k,color="gray",ls=":",lw=.8)
    ax.set(xlabel="时刻/h",ylabel="购电功率/kW"); ax.legend(frameon=False)
    export_figure(fig, FIG / "process_q3_four_issue_adjustment", dpi=300)

    parts=[(out.price*np.minimum(out.original_plan_kwh,out.buy_kwh)).sum(),(out.price*.5*np.maximum(out.original_plan_kwh-out.buy_kwh,0)).sum(),(out.price*1.5*np.maximum(out.buy_kwh-out.original_plan_kwh,0)).sum(),(out.price*5*out.emergency_kwh).sum()]
    fig,ax=plt.subplots(figsize=(6.4,3.5)); ax.bar(["计划内","少购违约","超购","紧急购电"],np.array(parts)/1e4,color=[c[0],c[2],c[3],c[1]]); ax.set_ylabel("全年费用/万元")
    export_figure(fig, FIG / "process_q3_settlement_components", dpi=300)

    fig,ax=plt.subplots(figsize=(6.3,3.6)); ax.bar(policy.policy,policy.total_cost_yuan/1e4,color=[c[3],c[2],c[1],c[0]]); ax.set_ylabel("全年总费用/万元")
    export_figure(fig, FIG / "result_q3_policy_cost", dpi=300)

    fig,ax=plt.subplots(figsize=(6.3,3.6)); bp=ax.boxplot([voi.voi_6_yuan,voi.voi_12_yuan,voi.voi_18_yuan],tick_labels=["6:00","12:00","18:00"],showfliers=False,patch_artist=True)
    for box,color in zip(bp["boxes"],[c[0],c[1],c[2]]): box.set_facecolor(color)
    ax.axhline(0,color="black",lw=.7); ax.set_ylabel("单日边际信息价值/元")
    export_figure(fig, FIG / "result_q3_voi_distribution", dpi=300)

    # 四个代表日采用2×2布局；每个面板以双纵轴同时展示购电功率与SOC，
    # 避免原8行纵排图过长，便于在A4正文中横向比较四个日期。
    # 注意：apply_publication_style 会开启 constrained_layout，且 set_layout_engine("none")
    # 只会换成 PlaceHolderLayoutEngine，subplots_adjust 仍会被忽略；必须在 subplots 时
    # 直接传 layout="none"，否则 twinx 的中缝坐标轴标签会互相重叠。
    fig,axs=plt.subplots(2,2,figsize=(10.5,5.8),sharex=True,layout="none")
    handles = None
    for ax,day in zip(axs.flat,REPS):
        x=out[out.date==day]; h=(x.slot-1)/6
        p1=ax.step(h,x.buy_kwh*6,where="post",color=c[0],lw=.9,label="调整购电")
        p2=ax.fill_between(h,0,x.emergency_kwh*6,color=c[1],alpha=.55,label="紧急购电")
        ae=ax.twinx()
        p3,=ae.plot(h,x.energy_end_kwh,color=c[3],lw=.9,ls="--",label="SOC")
        ae.axhline(1200,color="gray",ls=":",lw=.5)
        ae.axhline(10800,color="gray",ls=":",lw=.5)
        ax.text(.03,.88,day.strftime("%m-%d"),transform=ax.transAxes)
        ax.set_xlim(0,24); ax.set_xticks([0,6,12,18,24])
        ax.set_ylabel("购电功率/kW"); ae.set_ylabel("SOC/kWh")
        ae.set_ylim(700,11300)
        if handles is None: handles=([p1[0],p2,p3],["调整购电","紧急购电","SOC"])
    for ax in axs[-1,:]: ax.set_xlabel("时刻/h")
    fig.legend(*handles,loc="upper right",ncol=3,frameon=False,
               bbox_to_anchor=(.998,.995))
    fig.subplots_adjust(top=.89,bottom=.11,left=.075,right=.925,wspace=.42,hspace=.28)
    export_figure(fig, FIG / "result_q3_representative_dispatch", dpi=300)
    plt.close("all")


if __name__ == "__main__": main()
