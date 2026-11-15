"""Registered defender feature ablations, selecting on 2022–24, confirming 2025/26.

No production writes or tuning beyond the four pre-registered schemas. Reuses
the exact stored monotonic incumbent fits and identical training-row order.
"""
from fpl_model import paths
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd

from fpl_model.defcon.defender_model import monotonic_params, POSITIVE_EVIDENCE
from fpl_model.defcon.recent_defcon import attach_recent_rates, feature_columns
from fpl_model.defcon import evaluate_defender_history as fitting
from fpl_model.defcon.evaluate_defender_monotonic import OUT as OLD, DATA, freeze_control, load_scope, predict, metrics, bootstrap

OUT = paths.VALIDATION / "defcon_feature_experiment_20261020"
BASE_COLS = ["AvgPer90", "AvgRaw", "StdPer90", "MatchesPlayed", "Recent5AvgPer90", "Recent5AvgRaw", "PctAboveThreshold",
             "OpponentAvgStat", "TeamAvgStat", "VenueAvg", "IsHome", "Is_FPL_API", "Pos_CB", "Pos_FB", "Pos_WB", "Pos_LB", "Pos_RB"]
ARMS = ["no_std", "ew", "no_std_ew", "lean"]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def preserve():
    freeze_control()
    baseline=OUT/"baseline"
    if baseline.exists():
        m=json.loads((baseline/"manifest.json").read_text())
        assert sha(OUT/"preregistration.json")==m["preregistration_sha256"]
        for name,h in m["input_sha256"].items():assert sha(DATA/name)==h, name
        return
    baseline.mkdir()
    for name in ("predictions_defensive.csv","defcon_average_probabilities.csv","player_ratings.csv"):
        shutil.copy2(DATA/name,baseline/name)
    here=Path(__file__).resolve().parent
    for name in ("predict.py","defender_model.py"):
        shutil.copy2(here/name,baseline/name)
    (baseline/"models").mkdir()
    for path in (here.parents[1]/"cache"/"defcon").glob("*.pkl.gz"):
        shutil.copy2(path,baseline/"models"/path.name)
    original=json.loads((OLD/"baseline"/"manifest.json").read_text())
    m={"preregistration_sha256":sha(OUT/"preregistration.json"),"input_sha256":original["inputs"],
       "output_sha256":{p.name:sha(p) for p in baseline.glob("*.csv")},
       "source_sha256":{p.name:sha(p) for p in baseline.glob("*.py")}}
    (baseline/"manifest.json").write_text(json.dumps(m,indent=2))


def params(cols):
    p=monotonic_params(cols)
    p["monotonic_cst"]=[int(c in POSITIVE_EVIDENCE or c=="RecentEWPer90") for c in cols]
    return p


def baseline_model(year):
    path=OLD/(f"history_updated_{year}.pkl.gz" if year<2025 else f"candidate_{year}.pkl.gz")
    with gzip.open(path,"rb") as stream:return pickle.load(stream)


def audit(model,val,cols):
    x=val[cols].values
    p=predict(model,x); raw=predict(model,x,raw=True)
    violation=0
    for c in [c for c in cols if c in POSITIVE_EVIDENCE or c=="RecentEWPer90"]:
        shifted=x.copy();j=cols.index(c)
        shifted[:,j]+= .001 if c=="PctAboveThreshold" else .1
        if c=="PctAboveThreshold":shifted[:,j]=np.minimum(shifted[:,j],1)
        violation+=int((predict(model,shifted)<p-1e-12).sum())+int((predict(model,shifted,raw=True)<raw-1e-12).sum())
    if "StdPer90" in cols:
        shifted=x.copy();shifted[:,cols.index("StdPer90")]+=.03
        sensitivity=abs(predict(model,shifted)-p)
    else:sensitivity=np.zeros(len(x))
    ordered=val.sort_values(["Player","MatchID"],kind="stable").reset_index(drop=True)
    personal=[c for c in cols if c not in {"OpponentAvgStat","TeamAvgStat","VenueAvg","IsHome"}]
    following=ordered.groupby("Player")[personal].shift(-1)
    good=following.notna().all(axis=1)
    old=ordered.loc[good,cols].copy();new=old.copy()
    new.loc[:,personal]=following.loc[good].values
    changes=abs(predict(model,new.values)-predict(model,old.values))
    return {"monotonicity_violations":violation,"std_003_abs_p95":float(np.quantile(sensitivity,.95)),
        "std_003_abs_max":float(sensitivity.max()),"personal_update_p95":float(np.quantile(changes,.95)),
        "personal_update_max":float(changes.max())}


def evaluate(year,mode,frame):
    cols=feature_columns(BASE_COLS,mode)
    train,val=frame[frame.Season<year],frame[frame.Season.eq(year)]
    model=baseline_model(year) if mode=="baseline" else fitting.fit(train,cols,mode,params(cols),year)
    p=predict(model,val[cols].values)
    if mode=="baseline":
        reference=pd.read_csv(OLD/(f"history_validation_{year}.csv" if year<2025 else f"validation_{year}.csv"),encoding="utf-8-sig")
        np.testing.assert_allclose(p,reference["updated" if year<2025 else "candidate"].values,atol=1e-14,rtol=0)
    rows=val[["Player","Season","MatchID","Target","MatchesPlayed"]].copy()
    rows["probability"]=p
    rows.to_csv(OUT/f"validation_{mode}_{year}.csv",index=False,encoding="utf-8-sig")
    result={"metrics":metrics(val.Target.values,p),"audit":audit(model,val,cols),"features":cols}
    print(year,mode,json.dumps(result),flush=True)
    return result,rows


def pooled(rows,base_rows):
    frame=pd.concat(rows,ignore_index=True)
    base=pd.concat(base_rows,ignore_index=True)
    pd.testing.assert_frame_equal(frame.drop(columns="probability"),base.drop(columns="probability"))
    return {"metrics":metrics(frame.Target.values,frame.probability.values),
            "paired_player_bootstrap_vs_baseline":bootstrap(frame,base.probability.values,frame.probability.values)}


def main():
    preserve()
    fitting.OUT=OUT
    scope=load_scope()
    frame=pd.read_pickle(OLD/"defender_frame.pkl")
    frame["RecentEWPer90"],latest=attach_recent_rates(frame,scope["_PRIORS_BY_PLAYER"],2026)
    frame.to_pickle(OUT/"frame_with_ew.pkl")
    (OUT/"latest_ew.json").write_text(json.dumps(latest,indent=2),encoding="utf-8")
    results={"preregistration_sha256":sha(OUT/"preregistration.json"),"development":{},"confirmation":{},"production_changed":False}
    rows={mode:[] for mode in ["baseline",*ARMS]}
    for year in (2022,2023,2024):
        results["development"][str(year)]={}
        for mode in rows:
            cell,r=evaluate(year,mode,frame)
            results["development"][str(year)][mode]=cell
            rows[mode].append(r)
    results["pooled_development"]={mode:pooled(r,rows["baseline"]) for mode,r in rows.items()}
    b=results["pooled_development"]["baseline"]["metrics"]
    eligible=[];gates={}
    for mode in ARMS:
        m=results["pooled_development"][mode]["metrics"]
        per_year=[(r["baseline"],r[mode]) for r in results["development"].values()]
        ok={"pooled_LL":m["log_loss"]<=b["log_loss"]+.002,"pooled_Brier":m["brier"]<=b["brier"]+.002,
            "pooled_AUC":m["auc"]>=b["auc"]-.01,
            "year_guardrails":all(n["metrics"]["log_loss"]<=o["metrics"]["log_loss"]+.01 and n["metrics"]["brier"]<=o["metrics"]["brier"]+.005 for o,n in per_year),
            "monotonicity":all(n["audit"]["monotonicity_violations"]==0 for _,n in per_year),
            "spread_sensitivity":all(n["audit"]["std_003_abs_p95"]<=o["audit"]["std_003_abs_p95"]*.5+1e-12 for o,n in per_year)}
        ok={k:bool(v) for k,v in ok.items()}
        gates[mode]=ok
        if all(ok.values()):eligible.append(mode)
    selected=None
    if eligible:
        best=min(results["pooled_development"][m]["metrics"]["log_loss"] for m in eligible)
        near=[m for m in eligible if results["pooled_development"][m]["metrics"]["log_loss"]<=best+.001]
        selected=min(near,key=lambda m:(len(feature_columns(BASE_COLS,m)),m))
    selection={"selected":selected,"gates":gates,"reason":"Lowest development LL among eligible arms, with registered simplicity tie-break","preregistration_sha256":results["preregistration_sha256"]}
    # Persist choice before touching confirmation outcomes for any new candidate.
    selection_path=OUT/"selection.json"
    if selection_path.exists():assert json.loads(selection_path.read_text())==selection
    else:selection_path.write_text(json.dumps(selection,indent=2))
    print("SELECTION",json.dumps(selection,indent=2),flush=True)
    results["selection"]=selection
    ship=False
    if selected:
        gates=[]
        for year in (2025,2026):
            old,_=evaluate(year,"baseline",frame)
            new,_=evaluate(year,selected,frame)
            b,n=old["metrics"],new["metrics"]
            ok={"LL":n["log_loss"]<=b["log_loss"]+.002,"Brier":n["brier"]<=b["brier"]+.002,
                "AUC":n["auc"]>=b["auc"]-.01,"monotonicity":new["audit"]["monotonicity_violations"]==0}
            ok={k:bool(v) for k,v in ok.items()}
            gates.append(all(ok.values()))
            results["confirmation"][str(year)]={"baseline":old,"selected":new,"gates":ok}
        ship=all(gates)
        if ship:
            cols=feature_columns(BASE_COLS,selected)
            fitting.fit(frame,cols,selected,params(cols),"production")
    results["ship"],results["deployed_mode"]=ship,selected if ship else "baseline"
    (OUT/"results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print("FINAL",json.dumps({"ship":ship,"selected":selected,"mode":results["deployed_mode"]}),flush=True)


if __name__=="__main__":main()
