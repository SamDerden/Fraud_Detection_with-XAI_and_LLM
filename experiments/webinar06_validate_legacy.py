"""Webinar 6: pandas audit of ORIGINAL flawed CSV; optional Great Expectations 1.x."""
import argparse,json
from pathlib import Path
import pandas as pd

def audit(path):
    d=pd.read_csv(path)
    checks={'amount_positive':int((d.amount<=0).sum()),'history_7d_ge_24h':int((d.tx_count_7d<d.tx_count_24h).sum()),'merchants_7d_le_tx_7d':int((d.distinct_merchants_7d>d.tx_count_7d).sum()),'timestamp_parseable':int(pd.to_datetime(d.timestamp,errors='coerce',utc=True).isna().sum()),'fraud_binary':int((~d.is_fraud.isin([0,1])).sum())}
    issues=(d.tx_count_7d<d.tx_count_24h)|(d.distinct_merchants_7d>d.tx_count_7d)
    return {'dataset':str(path),'rows':len(d),'columns':len(d.columns),'missing_cells':int(d.isna().sum().sum()),'duplicate_rows':int(d.duplicated().sum()),'duplicate_transaction_ids':int(d.transaction_id.duplicated().sum()),'rule_violation_counts':checks,'rows_with_history_inconsistency':int(issues.sum()),'ge_status':'not_run'}

def run_ge(path,report):
    try:import great_expectations as gx
    except ImportError:
        report['ge_status']='unavailable: install great-expectations>=1,<2';return
    try:
        d=pd.read_csv(path);context=gx.get_context(mode='ephemeral');source=context.data_sources.add_pandas('legacy_pandas');asset=source.add_dataframe_asset(name='legacy_transactions');batch_def=asset.add_batch_definition_whole_dataframe('whole');batch=batch_def.get_batch(batch_parameters={'dataframe':d})
        expectations=[gx.expectations.ExpectColumnValuesToNotBeNull(column='transaction_id'),gx.expectations.ExpectColumnValuesToBeBetween(column='amount',min_value=0.000001),gx.expectations.ExpectColumnValuesToBeInSet(column='is_fraud',value_set=[0,1])]
        outcomes=[batch.validate(exp) for exp in expectations]
        report['ge_status']='completed';report['ge_expectations']=[{'expectation':type(e).__name__,'success':bool(o.success)} for e,o in zip(expectations,outcomes)]
    except Exception as exc:report['ge_status']='error: '+str(exc)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',default='legacy_validation.json');p.add_argument('--ge',action='store_true');args=p.parse_args()
    report=audit(args.data)
    if args.ge:run_ge(args.data,report)
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
