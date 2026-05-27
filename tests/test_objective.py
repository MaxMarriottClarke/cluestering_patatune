"""
Tests for objective.py — scoring primitives and objective sub-functions.

Run with:
    cd cluestering_patatune
    pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from objective import (
    reco_to_sim_score,
    assign_tracksters_to_cps,
    _shared_energy,
    _objective_purity,
    _objective_efficiency,
    _objective_fragmentation,
    filter_lcs_by_subdet,
    make_objective,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _trackster(lc_indexes, lc_energies, subdet='CEE'):
    return {
        'lc_indexes':  np.array(lc_indexes,  dtype=np.int64),
        'lc_energies': np.array(lc_energies, dtype=np.float32),
        'subdet':      subdet,
    }


def _shower(shower_id, lc_indexes, lc_energies):
    lc_e = np.array(lc_energies, dtype=np.float32)
    return {
        'shower_id':   shower_id,
        'lc_indexes':  np.array(lc_indexes, dtype=np.int64),
        'lc_energy':   lc_e,
        'true_energy': float(lc_e.sum()),
    }


def _feasible_result(tracksters, assignments, sim_showers):
    return {
        'infeasible':   False,
        'n_tracksters': len(tracksters),
        'tracksters':   tracksters,
        'assignments':  assignments,
        'sim_showers':  sim_showers,
    }


# ── reco_to_sim_score ──────────────────────────────────────────────────────────

class TestRecoToSimScore:

    def test_perfect_purity(self):
        """All trackster LCs belong to the CP → score 0."""
        score = reco_to_sim_score(
            np.array([0, 1, 2]),
            np.array([1.0, 2.0, 3.0]),
            cp_lc_index_set={0, 1, 2, 99},
        )
        assert score == pytest.approx(0.0)

    def test_full_contamination(self):
        """No trackster LCs belong to the CP → score 1."""
        score = reco_to_sim_score(
            np.array([0, 1, 2]),
            np.array([1.0, 2.0, 3.0]),
            cp_lc_index_set={10, 11, 12},
        )
        assert score == pytest.approx(1.0)

    def test_half_contamination(self):
        """Two equal-energy LCs, one from CP → score 0.5."""
        score = reco_to_sim_score(
            np.array([0, 1]),
            np.array([1.0, 1.0]),
            cp_lc_index_set={0},
        )
        assert score == pytest.approx(0.5)

    def test_energy_weighted(self):
        """Contamination is energy-weighted, not LC-count-weighted."""
        # LC 0 (energy 9) is from CP, LC 1 (energy 1) is NOT from CP
        # impure fraction = 1/10 = 0.1
        score = reco_to_sim_score(
            np.array([0, 1]),
            np.array([9.0, 1.0]),
            cp_lc_index_set={0},
        )
        assert score == pytest.approx(0.1)

    def test_zero_total_energy(self):
        """Zero-energy trackster returns 0 (not a division error)."""
        score = reco_to_sim_score(
            np.array([0, 1]),
            np.array([0.0, 0.0]),
            cp_lc_index_set={99},
        )
        assert score == pytest.approx(0.0)


# ── assign_tracksters_to_cps ───────────────────────────────────────────────────

class TestAssignTrackstersToCps:

    def test_clean_assignment(self):
        """Two pure tracksters map to their respective CPs with score 0."""
        tracksters = [
            _trackster([0, 1], [1.0, 1.0]),   # belongs to CP 0
            _trackster([2, 3], [1.0, 1.0]),   # belongs to CP 1
        ]
        sim_showers = [
            _shower(0, [0, 1], [1.0, 1.0]),
            _shower(1, [2, 3], [1.0, 1.0]),
        ]
        assignments = assign_tracksters_to_cps(tracksters, sim_showers)
        assert assignments[0] == (0, pytest.approx(0.0))
        assert assignments[1] == (1, pytest.approx(0.0))

    def test_mixed_trackster_picks_best_cp(self):
        """A mixed trackster is assigned to the CP with the largest shared energy."""
        # Trackster has LCs [0,1,2]: LCs 0,1 from CP0, LC 2 from CP1
        # score vs CP0 = 1/3 (only LC2 is impure, energy=1 out of 3)
        # score vs CP1 = 2/3 (LCs 0,1 are impure, energy=2 out of 3)
        # → should assign to CP0
        tracksters = [_trackster([0, 1, 2], [1.0, 1.0, 1.0])]
        sim_showers = [
            _shower(0, [0, 1], [1.0, 1.0]),
            _shower(1, [2],    [1.0]),
        ]
        assignments = assign_tracksters_to_cps(tracksters, sim_showers)
        assert assignments[0][0] == 0          # assigned to CP 0
        assert assignments[0][1] == pytest.approx(1/3)

    def test_lc_indexes_span_both_subdets(self):
        """CP LC set spans CEE and CHE; trackster with CHE LCs matched correctly."""
        # CP 0 owns LCs 0-4 (CEE) and 10-14 (CHE)
        tracksters = [_trackster([10, 11], [2.0, 2.0], subdet='CHE')]
        sim_showers = [
            _shower(0, [0, 1, 2, 10, 11], [1.0, 1.0, 1.0, 2.0, 2.0]),
            _shower(1, [5, 6],             [1.0, 1.0]),
        ]
        assignments = assign_tracksters_to_cps(tracksters, sim_showers)
        assert assignments[0][0] == 0
        assert assignments[0][1] == pytest.approx(0.0)


# ── _shared_energy ─────────────────────────────────────────────────────────────

class TestSharedEnergy:

    def test_full_overlap(self):
        t  = _trackster([0, 1], [3.0, 4.0])
        cp = _shower(0, [0, 1], [3.0, 4.0])
        assert _shared_energy(t, cp) == pytest.approx(7.0)

    def test_no_overlap(self):
        t  = _trackster([0, 1], [3.0, 4.0])
        cp = _shower(0, [5, 6], [1.0, 1.0])
        assert _shared_energy(t, cp) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # LC 0 shared (energy 3), LC 1 not shared (energy 4)
        t  = _trackster([0, 1], [3.0, 4.0])
        cp = _shower(0, [0, 9], [3.0, 1.0])
        assert _shared_energy(t, cp) == pytest.approx(3.0)


# ── _objective_purity ──────────────────────────────────────────────────────────

class TestObjectivePurity:

    def test_perfect(self):
        """All tracksters perfectly pure → F1 = 0."""
        showers = [_shower(0, [0, 1], [1.0, 1.0]), _shower(1, [2, 3], [1.0, 1.0])]
        result  = _feasible_result(
            tracksters  = [_trackster([0, 1], [1.0, 1.0]),
                           _trackster([2, 3], [1.0, 1.0])],
            assignments = {0: (0, 0.0), 1: (1, 0.0)},
            sim_showers = showers,
        )
        assert _objective_purity([result]) == pytest.approx(0.0)

    def test_takes_max_not_mean(self):
        """F1 = max score per event, not mean."""
        showers = [_shower(0, [0], [1.0]), _shower(1, [1], [1.0])]
        result  = _feasible_result(
            tracksters  = [_trackster([0], [1.0]), _trackster([1], [1.0])],
            assignments = {0: (0, 0.0), 1: (1, 0.8)},   # worst = 0.8
            sim_showers = showers,
        )
        assert _objective_purity([result]) == pytest.approx(0.8)

    def test_averaged_over_events(self):
        """F1 is the mean of per-event worst scores."""
        showers = [_shower(0, [0], [1.0]), _shower(1, [1], [1.0])]
        r1 = _feasible_result([], {0: (0, 0.2)}, showers)
        r2 = _feasible_result([], {0: (0, 0.6)}, showers)
        assert _objective_purity([r1, r2]) == pytest.approx(0.4)

    def test_infeasible_returns_inf(self):
        result = {'infeasible': True}
        assert _objective_purity([result]) == np.inf


# ── _objective_efficiency ──────────────────────────────────────────────────────

class TestObjectiveEfficiency:

    def test_perfect(self):
        """Both CPs fully recovered → F2 = 0."""
        showers = [_shower(0, [0, 1], [1.0, 1.0]), _shower(1, [2, 3], [1.0, 1.0])]
        result  = _feasible_result(
            tracksters  = [_trackster([0, 1], [1.0, 1.0]),
                           _trackster([2, 3], [1.0, 1.0])],
            assignments = {0: (0, 0.0), 1: (1, 0.0)},
            sim_showers = showers,
        )
        assert _objective_efficiency([result]) == pytest.approx(0.0)

    def test_one_cp_missed(self):
        """One CP has 0 energy recovered → min efficiency = 0 → F2 = 1."""
        showers = [_shower(0, [0, 1], [1.0, 1.0]), _shower(1, [2, 3], [1.0, 1.0])]
        # Both tracksters assigned to CP 0; CP 1 gets nothing
        result  = _feasible_result(
            tracksters  = [_trackster([0, 1], [1.0, 1.0]),
                           _trackster([0, 1], [1.0, 1.0])],
            assignments = {0: (0, 0.0), 1: (0, 0.0)},
            sim_showers = showers,
        )
        assert _objective_efficiency([result]) == pytest.approx(1.0)

    def test_partial_recovery(self):
        """CP 0 fully recovered, CP 1 half recovered → min eff = 0.5 → F2 = 0.5."""
        showers = [_shower(0, [0, 1], [1.0, 1.0]), _shower(1, [2, 3], [1.0, 1.0])]
        result  = _feasible_result(
            tracksters  = [_trackster([0, 1], [1.0, 1.0]),
                           _trackster([2],    [1.0])],      # only half of CP 1
            assignments = {0: (0, 0.0), 1: (1, 0.0)},
            sim_showers = showers,
        )
        assert _objective_efficiency([result]) == pytest.approx(0.5)

    def test_cee_and_che_tracksters_summed(self):
        """
        Pion CP has LCs in both CEE (indexes 0,1) and CHE (indexes 10,11).
        Recovery is correctly summed across both tracksters.
        """
        showers = [
            _shower(0, [5, 6],     [2.0, 2.0]),          # electron CP
            _shower(1, [0, 1, 10, 11], [1.0, 1.0, 1.0, 1.0]),  # pion CP (CEE+CHE)
        ]
        result = _feasible_result(
            tracksters = [
                _trackster([5, 6],   [2.0, 2.0], subdet='CEE'),  # electron
                _trackster([0, 1],   [1.0, 1.0], subdet='CEE'),  # pion CEE portion
                _trackster([10, 11], [1.0, 1.0], subdet='CHE'),  # pion CHE portion
            ],
            assignments = {0: (0, 0.0), 1: (1, 0.0), 2: (1, 0.0)},
            sim_showers = showers,
        )
        # Both pion tracksters contribute → total shared = 4.0 / 4.0 = 1.0 efficiency
        # min(1.0, 1.0) = 1.0  → F2 = 0.0
        assert _objective_efficiency([result]) == pytest.approx(0.0)

    def test_infeasible_returns_inf(self):
        assert _objective_efficiency([{'infeasible': True}]) == np.inf


# ── _objective_fragmentation ───────────────────────────────────────────────────

class TestObjectiveFragmentation:

    def test_perfect_ee_event(self):
        """Two CEE tracksters, one per CP → F3 = 0."""
        result = _feasible_result(
            tracksters  = [_trackster([0], [1.0], 'CEE'),
                           _trackster([1], [1.0], 'CEE')],
            assignments = {0: (0, 0.0), 1: (1, 0.0)},
            sim_showers = [],
        )
        assert _objective_fragmentation([result]) == pytest.approx(0.0)

    def test_perfect_pion_event(self):
        """
        3 tracksters: electron (CEE), pion-CEE, pion-CHE.
        Each CP has 1 trackster per subdetector → F3 = 0.
        This is the key case that was wrong before the fix.
        """
        result = _feasible_result(
            tracksters  = [
                _trackster([0], [1.0], 'CEE'),   # electron
                _trackster([1], [1.0], 'CEE'),   # pion CEE
                _trackster([2], [1.0], 'CHE'),   # pion CHE
            ],
            assignments = {
                0: (0, 0.0),   # electron → CP 0
                1: (1, 0.0),   # pion CEE → CP 1
                2: (1, 0.0),   # pion CHE → CP 1
            },
            sim_showers = [],
        )
        # (CP0, CEE):1  (CP1, CEE):1  (CP1, CHE):1 → no excess
        assert _objective_fragmentation([result]) == pytest.approx(0.0)

    def test_cee_oversplit(self):
        """Pion's CEE portion split into 2 CEE tracksters → F3 = 1."""
        result = _feasible_result(
            tracksters  = [
                _trackster([0],    [1.0], 'CEE'),   # electron
                _trackster([1],    [1.0], 'CEE'),   # pion CEE fragment 1
                _trackster([2],    [1.0], 'CEE'),   # pion CEE fragment 2  ← bad
                _trackster([3],    [1.0], 'CHE'),   # pion CHE
            ],
            assignments = {
                0: (0, 0.0),
                1: (1, 0.0),
                2: (1, 0.0),   # second CEE trackster also assigned to pion
                3: (1, 0.0),
            },
            sim_showers = [],
        )
        # (CP0,CEE):1  (CP1,CEE):2 → excess=1  (CP1,CHE):1 → total excess=1
        assert _objective_fragmentation([result]) == pytest.approx(1.0)

    def test_averaged_over_events(self):
        """F3 is the mean of per-event fragmentation."""
        # Event 1: no fragmentation
        r1 = _feasible_result(
            [_trackster([0],[1.],'CEE'), _trackster([1],[1.],'CEE')],
            {0:(0,0.), 1:(1,0.)}, [],
        )
        # Event 2: 1 fragmentation
        r2 = _feasible_result(
            [_trackster([0],[1.],'CEE'), _trackster([1],[1.],'CEE'),
             _trackster([2],[1.],'CEE')],
            {0:(0,0.), 1:(1,0.), 2:(1,0.)}, [],
        )
        assert _objective_fragmentation([r1, r2]) == pytest.approx(0.5)

    def test_infeasible_returns_inf(self):
        assert _objective_fragmentation([{'infeasible': True}]) == np.inf


# ── filter_lcs_by_subdet ───────────────────────────────────────────────────────

class TestFilterLcsBySubdet:

    def _make_all_lcs(self):
        return {
            'indexes': np.array([0, 1, 2, 3, 4], dtype=np.int64),
            'x':       np.array([1., 2., 3., 4., 5.], dtype=np.float32),
            'y':       np.zeros(5, dtype=np.float32),
            'z':       np.array([300., 320., 355., 360., 370.], dtype=np.float32),
            'energy':  np.array([1., 1., 2., 2., 2.], dtype=np.float32),
            'subdet':  np.array(['CEE','CEE','CHE','CHE','CHE']),
        }

    def test_cee_filter(self):
        lcs = filter_lcs_by_subdet(self._make_all_lcs(), 'CEE')
        assert list(lcs['indexes']) == [0, 1]
        assert list(lcs['energy'])  == pytest.approx([1., 1.])

    def test_che_filter(self):
        lcs = filter_lcs_by_subdet(self._make_all_lcs(), 'CHE')
        assert list(lcs['indexes']) == [2, 3, 4]
        assert list(lcs['energy'])  == pytest.approx([2., 2., 2.])

    def test_empty_result(self):
        all_lcs = {
            'indexes': np.array([0, 1], dtype=np.int64),
            'x': np.zeros(2, dtype=np.float32),
            'y': np.zeros(2, dtype=np.float32),
            'z': np.zeros(2, dtype=np.float32),
            'energy': np.ones(2, dtype=np.float32),
            'subdet': np.array(['CEE', 'CEE']),
        }
        lcs = filter_lcs_by_subdet(all_lcs, 'CHE')
        assert len(lcs['indexes']) == 0


# ── make_objective (integration) ───────────────────────────────────────────────

class TestMakeObjective:
    """
    Integration tests that mock run_cluestering so CLUEstering doesn't need
    to be installed.  The mock returns pre-defined tracksters per subdetector.
    """

    def _make_event(self, particle_type='ee'):
        """Minimal valid event: 2 CPs, LCs in CEE only (electron-like)."""
        return {
            'particle_type': particle_type,
            'sim_showers': [
                {
                    'shower_id':   0,
                    'lc_indexes':  np.array([0, 1], dtype=np.int64),
                    'lc_energy':   np.array([1.0, 1.0], dtype=np.float32),
                    'true_energy': 2.0,
                },
                {
                    'shower_id':   1,
                    'lc_indexes':  np.array([2, 3], dtype=np.int64),
                    'lc_energy':   np.array([1.0, 1.0], dtype=np.float32),
                    'true_energy': 2.0,
                },
            ],
            'all_lcs': {
                'indexes': np.array([0, 1, 2, 3], dtype=np.int64),
                'x':       np.array([1., 2., 3., 4.], dtype=np.float32),
                'y':       np.zeros(4, dtype=np.float32),
                'z':       np.array([320., 325., 330., 335.], dtype=np.float32),
                'energy':  np.array([1., 1., 1., 1.], dtype=np.float32),
                'subdet':  np.array(['CEE', 'CEE', 'CEE', 'CEE']),
            },
        }

    def _perfect_cluestering(self, lcs, *args):
        """Mock: returns 2 perfectly clean tracksters from CEE, nothing from CHE."""
        if len(lcs['indexes']) == 0:
            return []
        # Split LCs [0,1] → trackster 0,  [2,3] → trackster 1
        mid = len(lcs['indexes']) // 2
        return [
            {'lc_indexes': lcs['indexes'][:mid],  'lc_energies': lcs['energy'][:mid]},
            {'lc_indexes': lcs['indexes'][mid:],  'lc_energies': lcs['energy'][mid:]},
        ]

    def _single_trackster(self, lcs, *args):
        """Mock: returns only 1 trackster → infeasible."""
        if len(lcs['indexes']) == 0:
            return []
        return [{'lc_indexes': lcs['indexes'], 'lc_energies': lcs['energy']}]

    def test_perfect_reconstruction_scores_zero(self):
        events = [self._make_event()]
        with patch('objective.run_cluestering', side_effect=self._perfect_cluestering):
            obj_fn = make_objective(events)
            f1, f2, f3 = obj_fn([1.0]*10)
        assert f1 == pytest.approx(0.0, abs=1e-6)
        assert f2 == pytest.approx(0.0, abs=1e-6)
        assert f3 == pytest.approx(0.0, abs=1e-6)

    def test_infeasible_returns_inf(self):
        events = [self._make_event()]
        with patch('objective.run_cluestering', side_effect=self._single_trackster):
            obj_fn = make_objective(events)
            result = obj_fn([1.0]*10)
        assert all(v == np.inf for v in result)

    def test_returns_three_values(self):
        events = [self._make_event()]
        with patch('objective.run_cluestering', side_effect=self._perfect_cluestering):
            obj_fn = make_objective(events)
            result = obj_fn([1.0]*10)
        assert len(result) == 3

    def test_params_split_correctly(self):
        """
        Verify CEE uses params[:5] and CHE uses params[5:] by checking
        that the mock receives different param sets for each subdetector.
        """
        received = []

        def recording_mock(lcs, *args):
            received.append(args)
            if len(lcs['indexes']) == 0:
                return []
            mid = len(lcs['indexes']) // 2
            return [
                {'lc_indexes': lcs['indexes'][:mid],  'lc_energies': lcs['energy'][:mid]},
                {'lc_indexes': lcs['indexes'][mid:],  'lc_energies': lcs['energy'][mid:]},
            ]

        # Make event with both CEE and CHE LCs
        event = self._make_event()
        event['all_lcs']['subdet'] = np.array(['CEE', 'CEE', 'CHE', 'CHE'])
        event['sim_showers'][0]['lc_indexes'] = np.array([0, 1, 2, 3])
        event['sim_showers'][0]['true_energy'] = 4.0
        event['sim_showers'].pop(1)
        # Re-add second shower to keep 2 CPs
        event['sim_showers'].append({
            'shower_id': 1, 'lc_indexes': np.array([], dtype=np.int64),
            'lc_energy': np.array([], dtype=np.float32), 'true_energy': 0.01,
        })

        cee_params = [1.0, 2.0, 3.0, 4.0, 0.5]
        che_params = [5.0, 6.0, 7.0, 8.0, 2.0]
        params = cee_params + che_params

        events = [self._make_event()]   # use simple event
        with patch('objective.run_cluestering', side_effect=recording_mock):
            obj_fn = make_objective(events)
            obj_fn(params)

        # First call is CEE, should use params[:5]
        assert list(received[0]) == pytest.approx(cee_params)
