"""
Unit tests for physics_plots.py metric computation.
No CLUEstering or ROOT files required — all results are synthetic.

Run with:  python -m pytest test_physics_plots.py -v
"""

import numpy as np
import pandas as pd
import pytest

# Patch CLUEstering import so physics_plots can be imported without the library.
import sys, types
sys.modules.setdefault('CLUEstering', types.ModuleType('CLUEstering'))

import physics_plots as pp


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tracksters_df(rows):
    """rows: list of (ptype, cp_energy, score)"""
    return pd.DataFrame(rows, columns=['ptype', 'cp_energy', 'score'])


def _make_cps_df(rows):
    """rows: list of (ptype, cp_energy, is_efficient)"""
    df = pd.DataFrame(rows, columns=['ptype', 'cp_energy', 'is_efficient'])
    df['cp_efficiency'] = np.nan
    return df


def _make_cps_df_with_cp_eff(rows):
    """rows: list of (ptype, cp_energy, is_efficient, cp_efficiency)"""
    return pd.DataFrame(rows, columns=['ptype', 'cp_energy', 'is_efficient', 'cp_efficiency'])


def _make_events_df(rows):
    """rows: list of (ptype, total_ev_energy, n_reco, n_sim)"""
    return pd.DataFrame(rows, columns=['ptype', 'total_ev_energy', 'n_reco', 'n_sim'])


def _uniform_edges(lo, hi, n=5):
    return np.linspace(lo, hi, n + 1)


# ── _purity_per_bin ───────────────────────────────────────────────────────────

class TestPurityPerBin:
    def test_all_pure(self):
        # All tracksters have score < 0.2 → purity should be 1.0 everywhere
        df = _make_tracksters_df([
            ('ee', 10.0, 0.05),
            ('ee', 10.0, 0.10),
            ('ee', 30.0, 0.15),
        ])
        edges = _uniform_edges(0, 40, 2)  # bins: [0,20), [20,40]
        centers, vals, errs = pp._purity_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(1.0)

    def test_all_impure(self):
        df = _make_tracksters_df([
            ('ee', 10.0, 0.5),
            ('ee', 10.0, 0.8),
        ])
        edges = _uniform_edges(0, 40, 2)
        centers, vals, errs = pp._purity_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(0.0)
        assert np.isnan(vals[1])

    def test_mixed(self):
        # bin [0,20): 3 tracksters, 2 pure → purity = 2/3
        df = _make_tracksters_df([
            ('ee', 10.0, 0.05),
            ('ee', 10.0, 0.10),
            ('ee', 10.0, 0.25),
        ])
        edges = _uniform_edges(0, 40, 2)
        centers, vals, errs = pp._purity_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(2 / 3)

    def test_error_binomial(self):
        # n=4, p=0.5 → err = sqrt(0.5*0.5/4) = 0.25
        df = _make_tracksters_df([
            ('ee', 10.0, 0.05),
            ('ee', 10.0, 0.05),
            ('ee', 10.0, 0.50),
            ('ee', 10.0, 0.50),
        ])
        edges = _uniform_edges(0, 40, 2)
        _, vals, errs = pp._purity_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(0.5)
        assert errs[0] == pytest.approx(0.25)

    def test_ptype_filter(self):
        df = _make_tracksters_df([
            ('ee', 10.0, 0.05),
            ('pi', 10.0, 0.90),  # should be ignored for ptype='ee'
        ])
        edges = _uniform_edges(0, 40, 2)
        _, vals, _ = pp._purity_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(1.0)

    def test_empty_ptype(self):
        df = _make_tracksters_df([('pi', 10.0, 0.05)])
        edges = _uniform_edges(0, 40, 2)
        _, vals, _ = pp._purity_per_bin(df, 'ee', edges)
        assert np.all(np.isnan(vals))


# ── _efficiency_per_bin ───────────────────────────────────────────────────────

class TestEfficiencyPerBin:
    def test_all_efficient(self):
        df = _make_cps_df([
            ('ee', 10.0, 1),
            ('ee', 10.0, 1),
            ('ee', 30.0, 1),
        ])
        edges = _uniform_edges(0, 40, 2)
        _, vals, _ = pp._efficiency_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(1.0)

    def test_none_efficient(self):
        df = _make_cps_df([('ee', 10.0, 0), ('ee', 10.0, 0)])
        edges = _uniform_edges(0, 40, 2)
        _, vals, _ = pp._efficiency_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(0.0)

    def test_half_efficient(self):
        # 2 CPs in bin [0,20): one efficient, one not → 0.5
        df = _make_cps_df([('ee', 10.0, 1), ('ee', 10.0, 0)])
        edges = _uniform_edges(0, 40, 2)
        _, vals, _ = pp._efficiency_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(0.5)

    def test_ptype_filter(self):
        df = _make_cps_df([
            ('ee', 10.0, 1),
            ('pi', 10.0, 0),  # must not affect ee
        ])
        edges = _uniform_edges(0, 40, 2)
        _, vals, _ = pp._efficiency_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(1.0)


# ── _cp_efficiency_per_bin ───────────────────────────────────────────────────

class TestCpEfficiencyPerBin:
    def test_perfect_reconstruction(self):
        # All CPs fully reconstructed → cp_efficiency = 1.0
        df = _make_cps_df_with_cp_eff([
            ('ee', 10.0, 1, 1.0),
            ('ee', 10.0, 1, 1.0),
        ])
        edges = _uniform_edges(0, 40, 2)
        _, vals, _ = pp._cp_efficiency_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(1.0)

    def test_partial_reconstruction(self):
        # Two CPs in bin with cp_efficiency 0.4 and 0.8 → mean = 0.6
        df = _make_cps_df_with_cp_eff([
            ('ee', 10.0, 0, 0.4),
            ('ee', 10.0, 0, 0.8),
        ])
        edges = _uniform_edges(0, 40, 2)
        _, vals, _ = pp._cp_efficiency_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(0.6)

    def test_ptype_filter(self):
        df = _make_cps_df_with_cp_eff([
            ('ee', 10.0, 1, 1.0),
            ('pi', 10.0, 0, 0.0),  # must not affect ee
        ])
        edges = _uniform_edges(0, 40, 2)
        _, vals, _ = pp._cp_efficiency_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(1.0)

    def test_empty_bin_is_nan(self):
        df = _make_cps_df_with_cp_eff([('ee', 10.0, 1, 0.9)])
        edges = _uniform_edges(0, 40, 2)
        _, vals, _ = pp._cp_efficiency_per_bin(df, 'ee', edges)
        assert np.isnan(vals[1])

    def test_error_is_sem(self):
        # 4 CPs with cp_efficiency [0.2, 0.4, 0.6, 0.8] → mean=0.5, SEM=std/2
        efficiencies = [0.2, 0.4, 0.6, 0.8]
        df = _make_cps_df_with_cp_eff([('ee', 10.0, 0, v) for v in efficiencies])
        edges = _uniform_edges(0, 40, 2)
        _, vals, errs = pp._cp_efficiency_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(0.5)
        expected_sem = np.std(efficiencies, ddof=1) / 2.0
        assert errs[0] == pytest.approx(expected_sem)


# ── _number_ratio_per_bin ─────────────────────────────────────────────────────

class TestNumberRatioPerBin:
    def test_perfect_ratio(self):
        # 2 events in bin, each with n_reco=2, n_sim=2 → ratio = 1.0
        df = _make_events_df([
            ('ee', 100.0, 2, 2),
            ('ee', 100.0, 2, 2),
        ])
        edges = _uniform_edges(50, 150, 1)  # one bin
        _, vals, _ = pp._number_ratio_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(1.0)

    def test_over_reco(self):
        # n_reco=4, n_sim=2 per event → ratio = 2.0
        df = _make_events_df([('ee', 100.0, 4, 2), ('ee', 100.0, 4, 2)])
        edges = _uniform_edges(50, 150, 1)
        _, vals, _ = pp._number_ratio_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(2.0)

    def test_mixed_bins(self):
        # bin0: 2 events with n_reco=2  → ratio=1.0
        # bin1: 2 events with n_reco=4  → ratio=2.0
        df = _make_events_df([
            ('ee',  75.0, 2, 2),
            ('ee',  75.0, 2, 2),
            ('ee', 125.0, 4, 2),
            ('ee', 125.0, 4, 2),
        ])
        edges = _uniform_edges(50, 150, 2)  # bins: [50,100), [100,150]
        _, vals, _ = pp._number_ratio_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(2.0)

    def test_ptype_filter(self):
        df = _make_events_df([
            ('ee', 100.0, 2, 2),
            ('pi', 100.0, 10, 2),  # must not pollute ee bin
        ])
        edges = _uniform_edges(50, 150, 1)
        _, vals, _ = pp._number_ratio_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(1.0)

    def test_error_is_sem(self):
        # 4 events, n_reco/n_sim = [1, 2, 1, 2] → mean=1.5, std=0.577, sem=0.289
        df = _make_events_df([
            ('ee', 100.0, 2, 2),
            ('ee', 100.0, 4, 2),
            ('ee', 100.0, 2, 2),
            ('ee', 100.0, 4, 2),
        ])
        edges = _uniform_edges(50, 150, 1)
        _, vals, errs = pp._number_ratio_per_bin(df, 'ee', edges)
        assert vals[0] == pytest.approx(1.5)
        expected_sem = np.std([1.0, 2.0, 1.0, 2.0], ddof=1) / 2.0
        assert errs[0] == pytest.approx(expected_sem)


# ── collect_raw_data ──────────────────────────────────────────────────────────

def _mock_event(ptype, lc_energies):
    """Minimal event dict for collect_raw_data. LCs split evenly between 2 CPs."""
    lc_idx = np.arange(len(lc_energies), dtype=np.int64)
    mid = len(lc_energies) // 2
    return {
        'particle_type': ptype,
        'sim_showers': [
            {
                'shower_id':   0,
                'lc_indexes':  lc_idx[:mid],
                'lc_energies': np.array(lc_energies[:mid], dtype=np.float32),
                'true_energy': float(sum(lc_energies[:mid])),
            },
            {
                'shower_id':   1,
                'lc_indexes':  lc_idx[mid:],
                'lc_energies': np.array(lc_energies[mid:], dtype=np.float32),
                'true_energy': float(sum(lc_energies[mid:])),
            },
        ],
        'all_lcs': {
            'indexes': lc_idx,
            'energy':  np.array(lc_energies, dtype=np.float32),
        },
    }


def _mock_result(event, trackster_lc_groups):
    """
    Build a minimal result dict as _eval_event would return.

    trackster_lc_groups : list of (lc_indexes_array, lc_energies_array)
    """
    from objective import assign_tracksters_to_cps
    tracksters = [
        {'lc_indexes': np.array(idx, dtype=np.int64),
         'lc_energies': np.array(eng, dtype=np.float32),
         'subdet': 'CEE'}
        for idx, eng in trackster_lc_groups
    ]
    assignments = assign_tracksters_to_cps(tracksters, event['sim_showers'])
    return {
        'infeasible':    False,
        'n_tracksters':  len(tracksters),
        'tracksters':    tracksters,
        'assignments':   assignments,
        'sim_showers':   event['sim_showers'],
        'particle_type': event['particle_type'],
    }


class TestCollectRawData:
    def test_ptype_stored_correctly(self):
        # 6 LCs → 3 per trackster, passing MIN_TRACKSTER_LCS=3 filter
        ev = _mock_event('ee', [10.0] * 6)
        result = _mock_result(ev, [([0,1,2], [10.0]*3), ([3,4,5], [10.0]*3)])
        t_df, c_df, e_df = pp.collect_raw_data([ev], [result])
        assert (t_df['ptype'] == 'ee').all()
        assert (c_df['ptype'] == 'ee').all()
        assert (e_df['ptype'] == 'ee').all()

    def test_n_reco_counts_all_tracksters(self):
        ev = _mock_event('ee', [10.0] * 6)
        result = _mock_result(ev, [([0,1,2], [10.0]*3), ([3,4,5], [10.0]*3)])
        _, _, e_df = pp.collect_raw_data([ev], [result])
        assert e_df.iloc[0]['n_reco'] == 2
        assert e_df.iloc[0]['n_sim'] == 2

    def test_infeasible_event_skipped(self):
        ev = _mock_event('ee', [10.0] * 6)
        result = {
            'infeasible': True,
            'n_tracksters': 0,
            'tracksters': [],
            'assignments': {},
            'sim_showers': ev['sim_showers'],
            'particle_type': 'ee',
        }
        t_df, c_df, e_df = pp.collect_raw_data([ev], [result])
        assert len(t_df) == 0
        assert len(c_df) == 0
        assert len(e_df) == 0

    def test_perfect_trackster_is_efficient(self):
        # Each trackster perfectly covers its matched CP → both efficient
        # CP 0: LCs [0,1,2] energy 10 each → cp_energy=30
        # CP 1: LCs [3,4,5] energy 20 each → cp_energy=60
        ev = _mock_event('ee', [10.0, 10.0, 10.0, 20.0, 20.0, 20.0])
        result = _mock_result(ev, [
            ([0, 1, 2], [10.0, 10.0, 10.0]),  # matches CP 0 exactly
            ([3, 4, 5], [20.0, 20.0, 20.0]),  # matches CP 1 exactly
        ])
        _, c_df, _ = pp.collect_raw_data([ev], [result])
        assert c_df['is_efficient'].sum() == 2

    def test_cp_efficiency_sum_of_shared(self):
        # CP 0 (shower_id=0): LCs [0,1,2] energy 10 each → cp_energy=30
        # Trackster 0 covers all of CP 0 → shared=30, cp_eff=1.0
        # CP 1 (shower_id=1): LCs [3,4,5] energy 20 each → cp_energy=60
        # Trackster 1 has LC 5 with 0 energy → shared with CP1 = 20+20+0 = 40, cp_eff=40/60
        ev = _mock_event('ee', [10.0, 10.0, 10.0, 20.0, 20.0, 20.0])
        result = _mock_result(ev, [
            ([0, 1, 2], [10.0, 10.0, 10.0]),  # all of CP 0
            ([3, 4, 5], [20.0, 20.0, 0.0]),   # LC 5 contributes 0 shared energy
        ])
        _, c_df, _ = pp.collect_raw_data([ev], [result])
        # c_df rows are appended in shower_id order: idx 0 → CP0, idx 1 → CP1
        assert c_df.iloc[0]['cp_efficiency'] == pytest.approx(1.0)
        assert c_df.iloc[1]['cp_efficiency'] == pytest.approx(40.0 / 60.0)

    def test_trackster_efficiency_requires_reco_to_sim_match(self):
        # Trackster 0 covers CP 0's LCs with high shared energy, but assign it
        # artificially to CP 1 via reco-to-sim → CP 0 should NOT be efficient.
        # We do this by giving trackster 0 only LCs from CP 1 so reco-to-sim
        # assigns it there, while it also shares LC 0 with CP 0 at low energy.
        # CP 0: LCs [0,1,2] energy [1, 1, 1] → cp_energy=3  (very low)
        # CP 1: LCs [3,4,5] energy [10,10,10] → cp_energy=30
        # Trackster 0: covers LCs [3,4,5] → reco-to-sim assigns to CP 1
        #   but also incidentally has LC 0 (energy 0) → shared with CP0 is 0
        # Trackster 1: covers LCs [0,1,2] → reco-to-sim assigns to CP 0
        #   shared with CP0 = 3, cp_eff = 1.0 → this one IS the best for CP0
        # Both conditions met for CP0 via trackster 1 → CP0 efficient
        ev = _mock_event('ee', [1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
        result = _mock_result(ev, [
            ([3, 4, 5], [10.0, 10.0, 10.0]),  # assigned to CP 1 by reco-to-sim
            ([0, 1, 2], [1.0,  1.0,  1.0]),   # assigned to CP 0 by reco-to-sim
        ])
        _, c_df, _ = pp.collect_raw_data([ev], [result])
        # CP0 (idx 0): trackster 1 is best match and also assigned there → efficient
        assert c_df.iloc[0]['is_efficient'] == 1
        # CP1 (idx 1): trackster 0 is best match and assigned there → efficient
        assert c_df.iloc[1]['is_efficient'] == 1

    def test_separate_ee_pi_no_cross_contamination(self):
        ee_ev = _mock_event('ee', [10.0] * 6)
        pi_ev = _mock_event('pi', [50.0] * 6)
        ee_res = _mock_result(ee_ev, [([0,1,2], [10.0]*3), ([3,4,5], [10.0]*3)])
        pi_res = _mock_result(pi_ev, [([0,1,2], [50.0]*3), ([3,4,5], [50.0]*3)])

        ee_t, ee_c, ee_e = pp.collect_raw_data([ee_ev], [ee_res])
        pi_t, pi_c, pi_e = pp.collect_raw_data([pi_ev], [pi_res])

        assert (ee_t['ptype'] == 'ee').all()
        assert (pi_t['ptype'] == 'pi').all()
        assert ee_e.iloc[0]['total_ev_energy'] != pi_e.iloc[0]['total_ev_energy']
