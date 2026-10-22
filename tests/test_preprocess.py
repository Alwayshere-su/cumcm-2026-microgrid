import json
import unittest
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
DATA = ROOT / "data_processed" / "microgrid_10min.csv"


@unittest.skipUnless(DATA.exists(), "原始附件未随仓库分发；运行 preprocess.py 后再测")
class PreprocessOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.micro = pd.read_csv(ROOT / "data_processed" / "microgrid_10min.csv")
        cls.q1 = pd.read_csv(ROOT / "data_processed" / "q1_typical_day.csv")
        cls.forecast = pd.read_csv(ROOT / "data_processed" / "pv_forecast_hourly.csv")
        cls.mapping = pd.read_csv(ROOT / "data_processed" / "time_mapping.csv")
        cls.report = json.loads((ROOT / "results" / "data_quality_report.json").read_text(encoding="utf-8"))

    def test_dimensions(self):
        self.assertEqual(self.q1.shape, (144, 8))
        self.assertEqual(len(self.micro), 365 * 144)
        self.assertEqual(len(self.forecast), 365 * 4 * 24)
        self.assertEqual(len(self.mapping), 146)
        self.assertEqual(self.mapping.iloc[-2]["template_interval"], "全天购电量")
        self.assertEqual(self.mapping.iloc[-1]["template_interval"], "全天购电费")

    def test_energy_units(self):
        self.assertLess((self.q1["load_kw"] / 6 - self.q1["load_kwh"]).abs().max(), 1e-9)
        self.assertLess((self.q1["pv_forecast_kw"] / 6 - self.q1["pv_forecast_kwh"]).abs().max(), 1e-9)
        self.assertLess((self.micro["load_kw"] / 6 - self.micro["load_kwh"]).abs().max(), 1e-9)
        self.assertLess((self.micro["pv_actual_kw"] / 6 - self.micro["pv_kwh"]).abs().max(), 1e-9)

    def test_dates_and_slots(self):
        counts = self.micro.groupby("date")["slot"].nunique()
        self.assertEqual(len(counts), 365)
        self.assertTrue((counts == 144).all())
        last = self.micro[self.micro["slot"] == 144].copy()
        self.assertTrue(
            (
                pd.to_datetime(last["timestamp_end"]).dt.normalize()
                == pd.to_datetime(last["date"]) + pd.Timedelta(days=1)
            ).all()
        )

    def test_nonnegative_core(self):
        self.assertTrue((self.micro[["load_kw", "pv_actual_kw", "price_dynamic"]] >= 0).all().all())

    def test_quality_contract(self):
        self.assertTrue(self.report["passed"], self.report["issues"])


if __name__ == "__main__":
    unittest.main()
