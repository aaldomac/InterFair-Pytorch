import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from modules.utils.reporting_utils import aggregate, summary_to_long
from modules.reporting.summarize import main
from modules.utils.plots.plot_uncertainty_bars import summarize_metrics

class SummaryAggregationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'condition':['b','b','a'], 'seed':[1,2,1],
                                   'F_tot':[1.,3.,5.], 'hidden_normalized':[.1,.3,.5]})

    def test_plot_raw_and_saved_summary_match(self):
        metrics=['F_tot','hidden_normalized']
        long=summary_to_long(aggregate(self.frame,['condition'],metrics),['condition'],metrics)
        raw=summarize_metrics(self.frame,['F_hidden','F_tot'],conditions=['b','a'])
        with patch('modules.utils.reporting_utils.aggregate',side_effect=AssertionError('No reaggregation')):
            saved=summarize_metrics(long,['F_hidden','F_tot'],conditions=['b','a'],input_format='summary')
        pd.testing.assert_frame_equal(raw,saved)
        self.assertAlmostEqual(raw.loc[1,'std'],np.sqrt(2))
        self.assertTrue(np.isnan(raw.loc[2,'std']))

    def test_missing_policy_and_duplicate_checks(self):
        frame=self.frame.copy();frame.loc[0,'F_tot']=np.nan
        stats=aggregate(frame,['condition'],['F_tot']).set_index('condition')
        self.assertEqual(stats.loc['b','F_tot_n'],1)
        self.assertEqual(stats.loc['b','F_tot_mean'],3.)
        with self.assertRaises(ValueError):summarize_metrics(frame,['F_tot'])
        with self.assertRaises(ValueError):summarize_metrics(pd.concat([self.frame,self.frame]),['F_tot'])
        with self.assertRaises(ValueError):aggregate(frame,['condition'],['F_tot'],missing='invalid')

    def test_cli_preserves_old_csv_and_exports_long_once(self):
        frame=self.frame.rename(columns={'seed':'data_seed'})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i,row in frame.iterrows():
                run=root/'runs'/row['condition']/f'seed_{row.data_seed}'
                run.mkdir(parents=True);(run/'COMPLETE').touch()
                row.to_frame().T.to_csv(run/'summary.csv',index=False)
            with patch('sys.argv',['summarize',str(root)]), patch('modules.utils.reporting_utils.aggregate',wraps=aggregate) as call:
                main();self.assertEqual(call.call_count,1)
            old=pd.read_csv(root/'by_condition.csv',header=[0,1],index_col=0)
            expected=frame.groupby('condition')[['F_tot','hidden_normalized']].agg(['count','mean','std'])
            pd.testing.assert_frame_equal(old,expected)
            long=pd.read_csv(root/'by_condition_long.csv')
            saved=summarize_metrics(long,['F_tot'],input_format='summary')
            raw=summarize_metrics(self.frame,['F_tot'],conditions=['a','b'])
            pd.testing.assert_frame_equal(saved,raw)

    def test_report_groups_and_empty_results(self):
        frame=self.frame.assign(group=['x','x','y'])
        result=aggregate(frame,['condition','group'],['F_tot'])
        self.assertEqual(result['runs'].tolist(),[1,2])
        self.assertTrue(aggregate(frame.iloc[:0],['condition'],['F_tot']).empty)
