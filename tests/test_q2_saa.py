import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_source_path  # noqa: E402
ROOT = find_project_root(__file__)
add_source_path()
from solve_q2_saa import historical_scenarios, solve_saa_plan, run_saa_day, run_saa_period
from forecast_q2 import causal_conformal

PRM={"max_charge_discharge_power_kw":6,"interval_hours":1/6,
     "charge_efficiency":.9,"discharge_efficiency":.9,
     "operating_min_kwh":0,"operating_max_kwh":10}

class SAATests(unittest.TestCase):
    def test_scenarios_accept_ridge_forecast_columns(self):
        dates=pd.date_range("2025-01-01",periods=4)
        f=pd.DataFrame({"date":np.repeat(dates,2),"slot":[1,2]*4,
            "reg_load_kwh":1.,"reg_pv_kwh":0.,
            "actual_load_kwh":np.repeat([1.,2.,3.,4.],2),"actual_pv_kwh":0.})
        scen,used=historical_scenarios(f,pd.Timestamp("2025-01-04"),2,
                                       load_col="reg_load_kwh",pv_col="reg_pv_kwh")
        self.assertEqual(scen.shape,(2,2))
        self.assertTrue(all(x<pd.Timestamp("2025-01-04") for x in used))

    def test_conformal_calibration_does_not_use_current_or_future_label(self):
        dates=pd.date_range("2025-01-01",periods=4)
        f=pd.DataFrame({"date":dates,"actual":[0.,0.,1000.,0.],"p10":[0.]*4,"p90":[0.]*4})
        a=causal_conformal(f,"actual","p10","p90",window_days=2,min_days=1)
        # The shock on day 3 can widen day 4, but cannot widen its own interval.
        self.assertEqual(a.loc[2,"p90_cal"],0.)
        self.assertGreater(a.loc[3,"p90_cal"],0.)

    def test_scenarios_use_only_past_dates(self):
        dates=pd.date_range("2025-01-01",periods=5)
        f=pd.DataFrame({"date":np.repeat(dates,2),"slot":list(range(1,3))*5,
            "lgb_load_kwh":1.,"lgb_pv_kwh":0.,
            "actual_load_kwh":np.repeat(np.arange(5.),2)+1,"actual_pv_kwh":0.})
        scen,used=historical_scenarios(f,pd.Timestamp("2025-01-05"),3,seed=1)
        self.assertTrue(all(pd.Timestamp(x)<pd.Timestamp("2025-01-05") for x in used))
        self.assertFalse(np.any(np.isclose(scen,4.0)))

    def test_saa_solution_satisfies_every_scenario(self):
        net=np.array([[1.,1.,1.,1.],[2.,2.,2.,2.],[4.,4.,4.,4.]])
        sol=solve_saa_plan(net,np.ones(4),5.,PRM,risk_weight=.4,alpha=.8)
        for s in range(3):
            residual=sol["buy"]+sol["discharge"][s]+sol["emergency"][s]-sol["charge"][s]-sol["surplus"][s]-net[s]
            self.assertLess(np.abs(residual).max(),1e-7)
            self.assertGreaterEqual(sol["energy"][s].min(),-1e-8)
            self.assertLessEqual(sol["energy"][s].max(),10+1e-8)
            cost=sol["buy"].sum()+5*sol["emergency"][s].sum()
            self.assertLessEqual(cost-sol["nu"]-sol["u"][s],1e-7)

    @unittest.skipUnless((ROOT / "data_processed" / "microgrid_10min.csv").exists(),
                         "原始附件未随仓库分发")
    def test_daily_runner_returns_executable_144_intervals(self):
        root = find_project_root(__file__)
        data=pd.read_csv(root/"data_processed"/"microgrid_10min.csv",parse_dates=["date"])
        forecasts=pd.read_csv(root/"data_processed"/"q2_forecasts.csv",parse_dates=["date"])
        run,meta=run_saa_day(data,forecasts,pd.Timestamp("2025-03-20"),6000.,n_scenarios=4,risk_weight=.25)
        self.assertEqual(len(run),144)
        self.assertTrue((run.plan_buy_kwh>=-1e-9).all())
        self.assertEqual(meta["scenario_count"],4)

    @unittest.skipUnless((ROOT / "data_processed" / "microgrid_10min.csv").exists(),
                         "原始附件未随仓库分发")
    def test_period_runner_preserves_cross_day_soc(self):
        root = find_project_root(__file__)
        data=pd.read_csv(root/"data_processed/microgrid_10min.csv",parse_dates=["date"])
        forecasts=pd.read_csv(root/"data_processed/q2_forecasts.csv",parse_dates=["date"])
        out,_=run_saa_period(data,forecasts,pd.date_range("2025-03-20","2025-03-21"),6000.,3,.1)
        self.assertEqual(len(out),288)
        first=out[out.date==pd.Timestamp("2025-03-20")].energy_end_kwh.iloc[-1]
        second=out[out.date==pd.Timestamp("2025-03-21")].energy_start_kwh.iloc[0]
        self.assertAlmostEqual(first,second,places=7)

if __name__=='__main__': unittest.main()
