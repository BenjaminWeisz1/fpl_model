"""Run production with the exact §46 fits; verify input keys before reusing them.

Unchanged RF fits come from the preserved current-head control, matched on full
training X/y, class, parameters, versions and seeds. Defender fits come from the
selected schema experiment. Reuse writes ordinary production cache entries;
normal weekly runs subsequently use predict.py without this runner.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS","2")
import ast
import gzip
import hashlib
import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from fpl_model.defcon.recent_defcon import attach_recent_rates
from research.defcon.evaluate_defender_monotonic import predict
from research.defcon.evaluate_feature_changes import OLD, OUT, preserve

HERE=Path(__file__).resolve().parent


def main():
    preserve()
    results=json.loads((OUT/"results.json").read_text())
    assert results["ship"]
    mode=results["deployed_mode"]
    assert os.environ.get("DEFCON_DEFENDER_FEATURES",mode)==mode
    assert os.environ.get("DEFCON_MONOTONIC_DEF","1")=="1"
    assert os.environ.get("DEFCON_TUNE","0")=="0"
    assert os.environ.get("DEFCON_MODEL_CACHE","1")=="1"
    baseline_hash=hashlib.md5((OUT/"baseline"/"predict.py").read_bytes()).hexdigest()
    source=(HERE/"predict.py").read_text(encoding="utf-8-sig")
    # Cache only this integration run's expensive feature tables, independently
    # of later validation/reporting code. The complete ingestion/builder/schema
    # prefix, helper sources, inputs and relevant switches determine the key.
    inputs=json.loads((OUT/"baseline"/"manifest.json").read_text())["input_sha256"]
    switches={k:v for k,v in os.environ.items() if k.startswith("DEFCON_")}
    frame_signature=hashlib.sha256(source[:source.index("def stratified_reliability")].encode()
        +(HERE/"recent_defcon.py").read_bytes()+(HERE/"defender_model.py").read_bytes()
        +json.dumps(inputs,sort_keys=True).encode()+json.dumps(switches,sort_keys=True).encode()).hexdigest()
    frame_cache=OUT/"production_frames.pkl"
    cached_frames={}
    if frame_cache.exists():
        with frame_cache.open("rb") as stream:cached=pickle.load(stream)
        if cached["signature"]==frame_signature:cached_frames=cached["frames"]
    scope={"__file__":str(HERE/"predict.py"),"__name__":"__main__"}
    checked=[]
    def install_frame_reuse():
        builder=scope["build_dataset"]
        def build(df,pos_cols,*args,**kwargs):
            name={"IsDefender":"df_def","IsMidfielderOrForward":"df_mid","IsForward":"df_fwd"}.get(pos_cols[0])
            if df is scope["matchlogs"] and name in cached_frames and kwargs.get("use_priors",True):
                print(f"Verified unchanged feature-table inputs; reusing {name}",flush=True)
                return cached_frames[name].copy()
            return builder(df,pos_cols,*args,**kwargs)
        scope["build_dataset"]=build
    def save_frames():
        if not cached_frames:
            temporary=frame_cache.with_suffix(".pkl.tmp")
            with temporary.open("wb") as stream:
                pickle.dump({"signature":frame_signature,"frames":{n:scope[n] for n in ("df_def","df_mid","df_fwd")}},stream)
            temporary.replace(frame_cache)
    def install():
        original=scope["tune_and_calibrate"]
        signature=inspect.signature(original)
        def reuse(*args,**kwargs):
            bound=signature.bind(*args,**kwargs);bound.apply_defaults();a=bound.arguments
            assert not a["do_tune"]
            key_args=[a[k] for k in ("X","y","model_cls","label","do_tune","default_params","random_state","cv","n_iter")]
            key=scope["_model_cache_key"](*key_args)
            path=scope["_cache_path_for"](a["label"],key)
            cls=a["model_cls"]
            if cls is scope["RandomForestClassifier"]:
                now=scope["_PREDICT_SOURCE_MD5"]
                try:
                    scope["_PREDICT_SOURCE_MD5"]=baseline_hash
                    old_key=scope["_model_cache_key"](*key_args)
                finally:scope["_PREDICT_SOURCE_MD5"]=now
                found=list((OUT/"baseline"/"models").glob(f"*__{old_key}.pkl.gz"))
                assert len(found)==1, f"RF inputs changed: {a['label']}"
                with gzip.open(found[0],"rb") as stream:payload=pickle.load(stream)
                kind="unchanged_RF"
            else:
                assert cls is scope["HistGradientBoostingClassifier"]
                assert scope["DEFENDER_FEATURE_MODE"]==mode
                assert (a["random_state"],a["cv"],a["n_iter"])==(42,5,30)
                tag="2025" if "2025" in a["label"] else "2026" if "2026" in a["label"] else "production"
                with gzip.open(OUT/f"history_{mode}_{tag}.pkl.gz","rb") as stream:payload=pickle.load(stream)
                expected=hashlib.sha256(a["X"].tobytes()+a["y"].tobytes()+json.dumps(a["default_params"],sort_keys=True).encode()
                    +f"{cls.__name__}|sklearn={sklearn.__version__}|numpy={np.__version__}|cv=5|rs=42".encode()).hexdigest()
                assert payload["history_key"]==expected, f"Defender inputs changed: {a['label']}"
                kind="validated_defender"
            assert type(payload["base_model"]) is cls
            checked.append({"label":a["label"],"kind":kind,"cache_key":key})
            if not path.exists():
                scope["_initialise_model_cache"]()
                scope["_store_cached_model"](path,a["label"],payload["base_model"],payload["calibrator"],payload["scaler"],payload["best_params"])
            print(f"Verified exact fit for {a['label']}",flush=True)
            return original(*args,**kwargs)
        scope["tune_and_calibrate"]=reuse
    scope["_install_verified_feature_reuse"]=install
    scope["_install_verified_frame_reuse"]=install_frame_reuse
    scope["_save_verified_frames"]=save_frames
    tree=ast.parse(source)
    body=[]
    for node in tree.body:
        body.append(node)
        if isinstance(node,ast.FunctionDef) and node.name=="tune_and_calibrate":
            body.extend(ast.parse("_install_verified_feature_reuse()").body)
        if isinstance(node,ast.FunctionDef) and node.name=="build_dataset":
            body.extend(ast.parse("_install_verified_frame_reuse()").body)
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="FWD_FEATURE_COLS" for t in node.targets):
            body.extend(ast.parse("_save_verified_frames()").body)
    tree.body=body
    exec(compile(ast.fix_missing_locations(tree),str(HERE/"predict.py"),"exec"),scope)
    assert len(checked)==9
    cols=scope["DEF_FEATURE_COLS"]
    assert "StdPer90" not in cols and "Recent5AvgRaw" not in cols and "RecentEWPer90" in cols
    with gzip.open(OUT/f"history_{mode}_production.pkl.gz","rb") as stream:experiment_model=pickle.load(stream)
    avg=scope["avg_df"]
    d=avg[avg.Pos.eq("D")]
    np.testing.assert_allclose(d.Probability.values,predict(experiment_model,d[cols].values),rtol=0,atol=1e-14)
    for name in ("defcon_average_probabilities.csv","predictions_defensive.csv"):
        old=pd.read_csv(OUT/"baseline"/name,encoding="utf-8-sig")
        new=pd.read_csv(scope["_DATA"]/name,encoding="utf-8-sig")
        pd.testing.assert_frame_equal(old[old.Pos.isin(["M","F"])].reset_index(drop=True),
            new[new.Pos.isin(["M","F"])][old.columns].reset_index(drop=True))
    # Reproduce the Ajer mechanism with this fitted model and context fixed.
    frame=pd.read_pickle(OLD/"defender_frame.pkl")
    fixtures=scope["fixtures"]  # prediction horizon, not GW7; use full schedule for truncation
    schedule=pd.read_csv(scope["_DATA"]/"fixtures.csv",encoding="utf-8-sig")
    gw=schedule[schedule.Season.eq(2026)&schedule.Round.eq(7)].copy()
    for c in ("Home","Away"):gw[c]=gw[c].str.strip()
    legs=pd.concat([gw.rename(columns={"Home":"Team","Away":"Opponent"}).assign(Venue="Home"),
                    gw.rename(columns={"Away":"Team","Home":"Opponent"}).assign(Venue="Away")])
    marked=frame.merge(legs[["Season","Team","Opponent","Venue"]].assign(_remove=True),
                       on=["Season","Team","Opponent","Venue"],how="left",validate="many_to_one")
    before=marked[marked._remove.isna()].drop(columns="_remove")
    _,before_ew=attach_recent_rates(before,scope["_PRIORS_BY_PLAYER"],2026)
    details=json.loads((OLD/"gw7_rating_change_details.json").read_text(encoding="utf-8"))
    cases={}
    for player in ("Kristoffer Ajer","James Tarkowski","John Egan"):
        now=d[d.Player.eq(player)]
        x_now=now[cols].copy()
        old_feat={**details[player]["before"],"RecentEWPer90":before_ew[player]}
        x_old=x_now.copy()
        for c in cols:
            if c in old_feat:x_old[c]=old_feat[c]
        p_old=float(predict(experiment_model,x_old.values).mean())
        p_now=float(predict(experiment_model,x_now.values).mean())
        # Std is absent, so changing only its recorded value cannot affect design.
        cases[player]={"fixed_current_model_before_personal":p_old,"fixed_current_model_after_personal":p_now,
            "old_EW90":before_ew[player],"new_EW90":scope["def_recent_lookup"][player],"isolated_std_effect":0.0}
    report={"mode":mode,"features":cols,"checked_fits":checked,"experiment_probabilities_match":True,
            "midfield_forward_outputs_unchanged":True,"cases":cases,
            "production_source_sha256":hashlib.sha256((HERE/"predict.py").read_bytes()).hexdigest()}
    (OUT/"integration.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("INTEGRATION",json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
