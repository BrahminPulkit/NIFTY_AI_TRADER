"""Complete deterministic audit of index and option history."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml

from .alignment import align_option_to_index

LAB = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parent


def _hash(path: Path) -> str:
    h=sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


@dataclass
class Loaded:
    name: str
    path: Path
    frame: pd.DataFrame
    schema_issues: list[str]
    digest: str


class InstitutionalDataValidator:
    def __init__(self, config_path: Path = ROOT/"config.yaml"):
        self.config_path=config_path.resolve()
        self.config=yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.critical: list[str]=[]

    def _path(self, raw: str) -> Path:
        path=(ROOT/raw).resolve()
        workspace=LAB.parents[1].resolve()
        if workspace not in path.parents:
            raise ValueError(f"Input outside approved workspace: {path}")
        return path

    def load(self, name: str) -> Loaded:
        spec=self.config["datasets"][name]; path=self._path(spec["path"])
        before=path.stat(); digest=_hash(path)
        df=pd.read_csv(path,low_memory=False)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise RuntimeError(f"{name} changed during read")
        missing=sorted(set(spec["required"])-set(df.columns))
        issues=[f"missing required column: {x}" for x in missing]
        timestamp=spec["timestamp"]
        if timestamp in df:
            if spec["timestamp_format"]=="unix_seconds":
                ts=pd.to_datetime(df[timestamp],unit="s",utc=True,errors="coerce").dt.tz_convert(self.config["timezone"])
            else:
                ts=pd.to_datetime(df[timestamp],utc=True,errors="coerce").dt.tz_convert(self.config["timezone"])
            df=df.assign(timestamp=ts).set_index("timestamp",drop=True)
            df.index.name="timestamp"
        return Loaded(name,path,df,issues,digest)

    def audit_dataset(self, loaded: Loaded) -> tuple[dict,list[pd.DataFrame]]:
        df=loaded.frame; has_time=isinstance(df.index,pd.DatetimeIndex)
        required=[x for x in self.config["datasets"][loaded.name]["required"] if x!="timestamp"]
        numeric=[x for x in required if x in df]
        nan_required=int(df[numeric].isna().sum().sum()) if numeric else 0
        total_required=max(len(df)*len(numeric),1)
        duplicates=int(df.index.duplicated().sum()) if has_time else 0
        monotonic=bool(df.index.is_monotonic_increasing) if has_time else False
        tz=str(df.index.tz) if has_time else None
        zero_prices=negative_prices=corrupt=0
        ohlc={"open","high","low","close"}
        if ohlc.issubset(df):
            prices=df[list(ohlc)].apply(pd.to_numeric,errors="coerce")
            zero_prices=int(prices.eq(0).sum().sum()); negative_prices=int(prices.lt(0).sum().sum())
            corrupt=int(((prices["high"]<prices[["open","close","low"]].max(axis=1))|
                         (prices["low"]>prices[["open","close","high"]].min(axis=1))).sum())
        zero_volume=int(pd.to_numeric(df["volume"],errors="coerce").eq(0).sum()) if "volume" in df else None
        negative_oi=int(pd.to_numeric(df["oi"],errors="coerce").lt(0).sum()) if "oi" in df else None
        iv_anomalies=None
        if "iv" in df:
            iv=pd.to_numeric(df["iv"],errors="coerce")
            iv_anomalies=int(
                ((iv<=self.config["quality"]["iv_min_exclusive"])|
                 (iv>self.config["quality"]["iv_max_inclusive"])).fillna(False).sum()
            )
        strike_anomalies=None
        if "strike" in df:
            strike=pd.to_numeric(df["strike"],errors="coerce")
            strike_anomalies=int(((strike<=0)|((strike%50)!=0)).fillna(False).sum())
        daily=pd.DataFrame(); gaps=pd.DataFrame(); missing_candles=0; half_sessions=0
        holiday_candidates=[]
        if has_time and len(df):
            counts=df.groupby(df.index.normalize()).size()
            expected=int(self.config["session"]["expected_minutes"])
            daily=pd.DataFrame({"date":counts.index,"candles":counts.values})
            daily["completeness_pct"]=daily.candles/expected*100
            daily["half_session"]=daily.candles.between(1,expected*.8-1)
            half_sessions=int(daily.half_session.sum())
            missing_candles=int((expected-daily.candles.clip(upper=expected)).clip(lower=0).sum())
            full=pd.date_range(counts.index.min(),counts.index.max(),freq="B")
            absent=full.difference(counts.index)
            holiday_candidates=[str(x.date()) for x in absent]
            series=df.index.to_series()
            diffs=series.diff().dt.total_seconds()
            gaps=pd.DataFrame({"timestamp":df.index,"gap_seconds":diffs.values})
            same_session_date=series.dt.normalize().eq(series.shift().dt.normalize()).to_numpy()
            gaps=gaps[(gaps.gap_seconds>60)&same_session_date]
        summary={
            "dataset":loaded.name,"source":str(loaded.path),"sha256":loaded.digest,
            "start_date":str(df.index.min()) if has_time and len(df) else None,
            "end_date":str(df.index.max()) if has_time and len(df) else None,
            "total_trading_days":int(df.index.normalize().nunique()) if has_time else 0,
            "total_candles":len(df),"missing_trading_day_candidates":len(holiday_candidates),
            "market_holiday_or_missing_day_candidates":holiday_candidates,
            "missing_candles":missing_candles,"duplicate_timestamps":duplicates,
            "timezone":tz,"timezone_consistent":tz==self.config["timezone"],
            "timestamp_monotonic":monotonic,"half_sessions":half_sessions,
            "abnormal_gaps":len(gaps),"corrupted_rows":corrupt,
            "nan_required_cells":nan_required,"nan_required_pct":nan_required/total_required*100,
            "zero_prices":zero_prices,"negative_prices":negative_prices,
            "zero_volume":zero_volume,"negative_oi":negative_oi,
            "iv_anomalies":iv_anomalies,"strike_anomalies":strike_anomalies,
            "schema_issues":loaded.schema_issues,
        }
        tables=[daily,gaps]
        return summary,tables

    def run(self) -> dict:
        enabled={
            name:spec for name,spec in self.config["datasets"].items()
            if spec.get("enabled",True)
        }
        loaded={name:self.load(name) for name in enabled}
        summaries={}; tables={}
        for name,item in loaded.items():
            summaries[name],tables[name]=self.audit_dataset(item)
            for issue in summaries[name]["schema_issues"]:
                self.critical.append(f"{name}: {issue}")
            if not summaries[name]["timestamp_monotonic"]:
                self.critical.append(f"{name}: timestamp unavailable or non-monotonic")
            if summaries[name]["duplicate_timestamps"]:
                self.critical.append(f"{name}: duplicate timestamps={summaries[name]['duplicate_timestamps']}")
            if summaries[name]["corrupted_rows"]:
                self.critical.append(f"{name}: corrupted OHLC rows={summaries[name]['corrupted_rows']}")
            if summaries[name]["nan_required_cells"]:
                self.critical.append(
                    f"{name}: required-field NaN cells={summaries[name]['nan_required_cells']}"
                )
            if summaries[name]["negative_oi"]:
                self.critical.append(f"{name}: negative OI rows={summaries[name]['negative_oi']}")
            if summaries[name]["iv_anomalies"]:
                self.critical.append(f"{name}: IV anomalies={summaries[name]['iv_anomalies']}")
            if name=="option_full" and summaries[name]["strike_anomalies"]:
                self.critical.append(f"{name}: missing/invalid strike rows={summaries[name]['strike_anomalies']}")
            if summaries[name]["zero_prices"] or summaries[name]["negative_prices"]:
                self.critical.append(
                    f"{name}: zero/negative price cells="
                    f"{summaries[name]['zero_prices'] + summaries[name]['negative_prices']}"
                )
        alignment=None; correlation=None; spot_consistency=None
        a,c=loaded["index"],loaded["option_full"]
        if all(isinstance(x.frame.index,pd.DatetimeIndex) for x in (a,c)):
            start,end=self.config["session"]["start"],self.config["session"]["end"]
            ai=a.frame.between_time(start,end)
            ci=c.frame.between_time(start,end)
            common_start=max(ai.index.min(),ci.index.min())
            common_end=min(ai.index.max(),ci.index.max())
            ai=ai.loc[common_start:common_end]
            ci=ci.loc[common_start:common_end]
            aligned_c=align_option_to_index(ai,ci,self.config["alignment"]["backward_tolerance_seconds"])
            matched_c=int(aligned_c.option_timestamp.notna().sum())
            option_days=set(ci.index.normalize())
            shared_mask=aligned_c.index.normalize().isin(option_days)
            shared=aligned_c.loc[shared_mask]
            alignment={
                "shared_session_start":str(common_start),"shared_session_end":str(common_end),
                "session_index_candles":len(ai),"session_option_candles":len(ci),
                "matched_candles":matched_c,
                "index_option_full_pct":matched_c/len(ai)*100,
                "exact_c_pct":float(aligned_c.exact_timestamp_match.mean()*100),
                "shared_day_exact_pct":float(shared.exact_timestamp_match.mean()*100),
                "shared_day_missing_candles":int((~shared.exact_timestamp_match).sum()),
            }
            if {"close_index","spot"}.issubset(aligned_c):
                pair=aligned_c[["close_index","spot"]].apply(pd.to_numeric,errors="coerce").dropna()
                correlation=float(pair.corr().iloc[0,1]) if len(pair)>1 else None
                spot_consistency={"observations":len(pair),"mean_absolute_error":float((pair.close_index-pair.spot).abs().mean()) if len(pair) else None}
            tables["_alignment"]=aligned_c[[
                "option_timestamp","alignment_lag_seconds","exact_timestamp_match",
                *[x for x in ("close_index","spot","strike","close_option") if x in aligned_c],
            ]].copy()
        else:
            self.critical.append("Exact minute alignment impossible: an enabled dataset has no timestamp column")
        full=c.frame
        if {"strike","spot"}.issubset(full):
            strike=pd.to_numeric(full["strike"],errors="coerce")
            spot=pd.to_numeric(full["spot"],errors="coerce")
            distance=(strike-spot).abs()
            summaries["option_full"]["atm_distance_mean"]=float(distance.mean())
            summaries["option_full"]["atm_distance_max"]=float(distance.max())
            summaries["option_full"]["atm_distance_over_limit"]=int(
                (distance>self.config["quality"]["maximum_atm_distance_points"]).sum()
            )
            switch=strike.ne(strike.shift())&strike.notna()&strike.shift().notna()
            same_day=full.index.normalize()==full.index.to_series().shift().dt.normalize().to_numpy()
            intraday=switch&same_day
            summaries["option_full"]["strike_switches"]=int(switch.sum())
            summaries["option_full"]["intraday_strike_switches"]=int(intraday.sum())
            if "close" in full:
                jump=pd.to_numeric(full["close"],errors="coerce").pct_change().abs()
                summaries["option_full"]["strike_switch_jump_over_20pct"]=int((intraday&(jump>.20)).sum())
            if summaries["option_full"]["intraday_strike_switches"]:
                self.critical.append(
                    "option_full: rolling strike changes intraday "
                    f"{summaries['option_full']['intraday_strike_switches']} times; "
                    "the prior contract's subsequent premium is unavailable for held-trade exits"
                )
            if summaries["option_full"]["atm_distance_over_limit"]:
                self.critical.append(
                    "option_full: strike farther than configured ATM limit on "
                    f"{summaries['option_full']['atm_distance_over_limit']} rows"
                )
        if alignment:
            minimum=self.config["quality"]["minimum_alignment_pct"]
            if alignment["shared_day_exact_pct"]<minimum:
                self.critical.append(
                    f"index/option exact alignment below {minimum}%: "
                    f"{alignment['shared_day_exact_pct']:.6f}% on shared trading days"
                )
        if correlation is None or correlation<self.config["quality"]["spot_index_correlation_minimum"]:
            self.critical.append(
                "spot/index correlation unavailable or below "
                f"{self.config['quality']['spot_index_correlation_minimum']}"
            )
        score=self._score(summaries,alignment)
        status="FAIL" if self.critical else ("PASS" if score>=self.config["quality"]["pass_minimum"] else "WARNING")
        result={"status":status,"data_quality_score":score,"can_research_continue":status=="PASS",
                "critical_issues":sorted(set(self.critical)),"datasets":summaries,
                "alignment":alignment,"spot_index_correlation":correlation,
                "spot_index_consistency":spot_consistency}
        return result,loaded,tables

    def _score(self,summaries,alignment):
        score=100.0
        for s in summaries.values():
            score-=min(len(s["schema_issues"])*8,24)
            score-=min(s["duplicate_timestamps"]/max(s["total_candles"],1)*100,10)
            score-=min(s["nan_required_pct"]*.25,10)
            if not s["timezone_consistent"]: score-=5
            if not s["timestamp_monotonic"]: score-=10
            if s["corrupted_rows"]: score-=10
        if alignment is None: score-=25
        else: score-=max(0,100-alignment["shared_day_exact_pct"])*.25
        return round(max(0,score),2)
