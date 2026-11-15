"""Causality, cameo and exposure tests for the replacement recent feature."""
import unittest
import ast
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pandas as pd
from recent_defcon import RecentRate, attach_recent_rates, feature_columns


def frame(values, minutes=None, seasons=None):
    n=len(values)
    return pd.DataFrame({"Player":["A"]*n,"MatchID":range(n),"Season":seasons or [2025]*n,
        "RawStat":values,"Minutes":minutes or [90]*n,"AvgPer90":[8.]*n})


class RecentTest(unittest.TestCase):
    def test_production_prior_off_ablation_builds_unseeded_recent_feature(self):
        source=Path(__file__).with_name("predict.py").read_text(encoding="utf-8-sig")
        function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="_no_prior_val_rows")
        rows=frame([2.,4.,6.],seasons=[2025,2026,2026])
        builder=Mock(return_value=rows)
        scope={"pd":pd,"matchlogs":rows,"_prior_players":{"A"},"_SEASON_CURRENT":2026,"MAX_SEASON":2026,
               "DEF_FEATURE_COLS":["RecentEWPer90"],"build_dataset":builder,"attach_recent_rates":attach_recent_rates,
               "DEFCON_LEVEL_ADJUST":False,"OPP_FACTOR_ON":False}
        exec(compile(ast.Module(body=[function],type_ignores=[]),"<production validation helper>","exec"),scope)
        result=scope["_no_prior_val_rows"](["IsDefender"],"DefStats_Def","Per90_Def",10,{}, {}, {}, ["CB"])
        self.assertEqual(result.Season.tolist(),[2026,2026])
        self.assertEqual(result.RecentEWPer90.iloc[0],2.)
        self.assertFalse(builder.call_args.kwargs["use_priors"])

    def test_row_outcome_cannot_change_own_feature(self):
        x=frame([8.,10.,12.])
        old,_=attach_recent_rates(x,{},2025)
        x.loc[1,"RawStat"]=30.
        new,_=attach_recent_rates(x,{},2025)
        np.testing.assert_array_equal(old[:2],new[:2])
        self.assertGreater(new[2],old[2])

    def test_cameo_does_not_update_or_evict(self):
        _,a=attach_recent_rates(frame([8.,12.]),{},2025)
        _,b=attach_recent_rates(frame([8.,100.,12.],[90,1,90]),{},2025)
        self.assertEqual(a,b)

    def test_rate_uses_exposure_not_mean_of_ratios(self):
        state=RecentRate()
        state.observe(6.,60.)
        state.observe(12.,90.)
        d=2**(-1/4)
        self.assertAlmostEqual(state.value(),(d*6+12)/(d*60/90+1))

    def test_old_observation_decays_instead_of_expiring(self):
        _,a=attach_recent_rates(frame([20.,8.,8.,8.,8.,8.]),{},2025)
        _,b=attach_recent_rates(frame([8.,8.,8.,8.,8.,8.]),{},2025)
        self.assertGreater(a["A"],b["A"])

    def test_prior_activates_only_at_arrival(self):
        x=frame([2.,2.,8.],seasons=[2024,2024,2025])
        r,_=attach_recent_rates(x,{"A":[(2025,{"matches":10.,"per90":12.})]},2025)
        self.assertEqual(r[0],8.)
        self.assertEqual(r[1],2.)
        self.assertEqual(r[2],12.)

    def test_prior_applies_for_not_yet_debuted_arrival(self):
        _,latest=attach_recent_rates(frame([2.],seasons=[2024]),{"A":[(2025,{"matches":10.,"per90":12.})]},2025)
        self.assertEqual(latest["A"],12.)

    def test_full_constant_rate_stays_constant(self):
        r,latest=attach_recent_rates(frame([9.]*20),{},2025)
        np.testing.assert_allclose(r[1:],9.)
        self.assertAlmostEqual(latest["A"],9.)

    def test_inference_matches_next_training_feature(self):
        priors={"A":[(2025,{"matches":10.,"per90":8.})]}
        x=frame([8.,10.,7.,12.,9.])
        rates,_=attach_recent_rates(x,priors,2025)
        _,latest=attach_recent_rates(x.iloc[:-1],priors,2025)
        self.assertEqual(rates[-1],latest["A"])

    def test_prior_weight_halves_after_four_qualifying_appearances(self):
        state=RecentRate(numerator=20.,exposure=1.)
        for _ in range(4):state.observe(0.,90.)
        self.assertAlmostEqual(state.numerator,10.)

    def test_player_histories_do_not_mix(self):
        x=pd.concat([frame([4.,4.]),frame([20.,20.]).assign(Player="B")],ignore_index=True)
        r,latest=attach_recent_rates(x,{},2025)
        self.assertEqual(r.tolist(),[8.,4.,8.,20.])
        self.assertEqual(latest,{"A":4.,"B":20.})

    def test_lean_schema_removes_only_registered_features(self):
        cols=["AvgPer90","AvgRaw","StdPer90","Recent5AvgPer90","Recent5AvgRaw","Pos_CB"]
        self.assertEqual(feature_columns(cols,"lean"),["AvgPer90","Pos_CB","RecentEWPer90"])


if __name__=="__main__":unittest.main()
