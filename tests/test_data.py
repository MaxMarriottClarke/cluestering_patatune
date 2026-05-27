"""
Tests for data.py — LC resolution logic and event validation.

Run with:
    cd cluestering_patatune
    pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import validate_events
from config import CEE_Z_BOUNDARY


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_event(particle_type='ee',
                shower0_idxs=None, shower0_energies=None,
                shower1_idxs=None, shower1_energies=None,
                all_z=None):
    """Build a minimal valid event dict."""
    if shower0_idxs    is None: shower0_idxs    = [0, 1]
    if shower0_energies is None: shower0_energies = [1.0, 1.0]
    if shower1_idxs    is None: shower1_idxs    = [2, 3]
    if shower1_energies is None: shower1_energies = [1.0, 1.0]

    all_idxs = sorted(set(shower0_idxs) | set(shower1_idxs))
    n        = len(all_idxs)
    if all_z is None:
        all_z = np.full(n, 320.0, dtype=np.float32)   # all in CEE by default

    subdet = np.where(np.abs(all_z) < CEE_Z_BOUNDARY, 'CEE', 'CHE')

    def _shower(sid, idxs, energies):
        e = np.array(energies, dtype=np.float32)
        return {
            'shower_id':   sid,
            'lc_indexes':  np.array(idxs, dtype=np.int64),
            'lc_energy':   e,
            'true_energy': float(e.sum()),
        }

    return {
        'particle_type': particle_type,
        'sim_showers': [
            _shower(0, shower0_idxs, shower0_energies),
            _shower(1, shower1_idxs, shower1_energies),
        ],
        'all_lcs': {
            'indexes': np.array(all_idxs, dtype=np.int64),
            'x':       np.zeros(n, dtype=np.float32),
            'y':       np.zeros(n, dtype=np.float32),
            'z':       all_z[:n],
            'energy':  np.ones(n, dtype=np.float32),
            'subdet':  subdet,
        },
    }


# ── validate_events ────────────────────────────────────────────────────────────

class TestValidateEvents:

    def test_valid_events_pass(self):
        events = [_make_event('ee'), _make_event('epion')]
        validate_events(events)   # should not raise

    def test_wrong_shower_count_raises(self):
        event = _make_event()
        event['sim_showers'].pop()   # remove one shower → only 1 left
        with pytest.raises(AssertionError, match="expected 2 sim_showers"):
            validate_events([event])

    def test_overlapping_lc_indexes_raises(self):
        """If the same LC index appears in both showers, validation should catch it."""
        event = _make_event(
            shower0_idxs=[0, 1, 2],
            shower1_idxs=[2, 3, 4],   # LC 2 is in both showers
        )
        with pytest.raises(AssertionError, match="LC 2"):
            validate_events([event])

    def test_zero_true_energy_raises(self):
        event = _make_event()
        event['sim_showers'][0]['true_energy'] = 0.0
        with pytest.raises(AssertionError, match="true_energy == 0"):
            validate_events([event])

    def test_bad_subdet_label_raises(self):
        event = _make_event()
        event['all_lcs']['subdet'] = np.array(['CEE', 'CEE', 'INVALID', 'CHE'])
        with pytest.raises(AssertionError, match="unexpected subdet"):
            validate_events([event])

    def test_multiple_valid_events(self):
        events = [_make_event('ee', [0,1],[1.,1.], [2,3],[2.,2.]) for _ in range(5)]
        validate_events(events)   # should not raise

    def test_che_lcs_accepted(self):
        """LCs with |z| >= 352 should be labelled CHE without error."""
        z_vals = np.array([360.0, 365.0, 370.0, 375.0], dtype=np.float32)
        event  = _make_event(all_z=z_vals)
        # subdet should be all CHE
        assert all(s == 'CHE' for s in event['all_lcs']['subdet'])
        validate_events([event])


# ── Subdetector boundary ───────────────────────────────────────────────────────

class TestSubdetBoundary:
    """Tests that the CEE/CHE z boundary is applied correctly."""

    def test_cee_boundary(self):
        z = np.array([0., 100., 351.9], dtype=np.float32)
        subdet = np.where(np.abs(z) < CEE_Z_BOUNDARY, 'CEE', 'CHE')
        assert all(s == 'CEE' for s in subdet)

    def test_che_boundary(self):
        z = np.array([352.0, 360.0, 400.0], dtype=np.float32)
        subdet = np.where(np.abs(z) < CEE_Z_BOUNDARY, 'CEE', 'CHE')
        assert all(s == 'CHE' for s in subdet)

    def test_negative_z(self):
        """Boundary is symmetric — negative z side treated identically."""
        z = np.array([-320., -360.], dtype=np.float32)
        subdet = np.where(np.abs(z) < CEE_Z_BOUNDARY, 'CEE', 'CHE')
        assert subdet[0] == 'CEE'
        assert subdet[1] == 'CHE'


# ── LC resolution logic ────────────────────────────────────────────────────────

class TestLcResolution:
    """
    Test the dominant-multiplicity resolution logic in isolation.
    Replicates the core of _build_event without needing awkward/uproot.
    """

    def _resolve(self, shower_data):
        """
        shower_data: list of (shower_id, lc_idxs, multiplicities)
        Returns: {shower_id: [resolved lc_idxs]}
        """
        lc_best_frac   = {}
        lc_best_shower = {}

        for shower_id, lc_idxs, mults in shower_data:
            for lc_idx, m in zip(lc_idxs, mults):
                if m <= 0:
                    continue
                frac = 1.0 / m
                if lc_idx not in lc_best_frac or frac > lc_best_frac[lc_idx]:
                    lc_best_frac[lc_idx]   = frac
                    lc_best_shower[lc_idx] = shower_id

        result = {sid: [] for sid, _, _ in shower_data}
        for lc_idx, winner in lc_best_shower.items():
            result[winner].append(lc_idx)
        return result

    def test_unambiguous_lcs(self):
        """Each LC exclusively in one shower → no change after resolution."""
        shower_data = [
            (0, [0, 1, 2], [1.0, 1.0, 1.0]),
            (1, [3, 4, 5], [1.0, 1.0, 1.0]),
        ]
        resolved = self._resolve(shower_data)
        assert sorted(resolved[0]) == [0, 1, 2]
        assert sorted(resolved[1]) == [3, 4, 5]

    def test_shared_lc_goes_to_dominant_shower(self):
        """LC 5 is in both showers; shower 0 has 1/mult=0.9, shower 1 has 1/mult=0.1."""
        shower_data = [
            (0, [0, 5], [1.0, 1/0.9]),   # mult for LC5 = 1/0.9, frac = 0.9
            (1, [1, 5], [1.0, 1/0.1]),   # mult for LC5 = 1/0.1, frac = 0.1
        ]
        resolved = self._resolve(shower_data)
        assert 5 in resolved[0]
        assert 5 not in resolved[1]

    def test_equal_fractions_deterministic(self):
        """When fractions are exactly equal, the last shower to set it wins (stable)."""
        shower_data = [
            (0, [99], [2.0]),   # frac = 0.5
            (1, [99], [2.0]),   # frac = 0.5 — tie, shower 1 processed second
        ]
        resolved = self._resolve(shower_data)
        all_assigned = resolved[0] + resolved[1]
        assert all_assigned.count(99) == 1   # exactly one shower gets it

    def test_no_lc_in_both_showers_after_resolution(self):
        """After resolution no LC index appears in more than one shower."""
        shower_data = [
            (0, [0, 1, 5, 6], [1.0, 1.0, 2.0, 3.0]),
            (1, [2, 3, 5, 6], [1.0, 1.0, 1.5, 4.0]),
        ]
        resolved = self._resolve(shower_data)
        all_lcs  = resolved[0] + resolved[1]
        assert len(all_lcs) == len(set(all_lcs)), "Duplicate LC after resolution"

    def test_zero_multiplicity_lcs_ignored(self):
        """LCs with multiplicity <= 0 are skipped."""
        shower_data = [
            (0, [0, 1], [1.0, 0.0]),   # LC 1 has zero multiplicity
            (1, [2, 3], [1.0, 1.0]),
        ]
        resolved = self._resolve(shower_data)
        assert 1 not in resolved[0]
        assert 1 not in resolved[1]
