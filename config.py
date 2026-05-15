# ── Data ──────────────────────────────────────────────────────────────────────
PATH = '/eos/user/m/mmarriot/SWAN_projects/cluestering_opt'

CONFIG_FILES = {
    'em':   'tune/e.root',
    'pion': 'tune/pi.root',
}

PARTICLE_TYPES = ['em', 'pion']

BRANCHES = [
    'vertices_indexes',
    'vertices_x',
    'vertices_y',
    'vertices_z',
    'vertices_energy',
    'vertices_multiplicity',
]

TRUTH_BRANCH = 'ticlDumper/simtrackstersCP'
CLUE_BRANCH  = 'ticlDumper/hltTiclTrackstersCLUE3DHigh'

# ── Default CLUEstering parameters ────────────────────────────────────────────
DC   = 3.0   # spatial density radius [cm]
RHOC = 5.0   # min local density to be a seed
DO   = 3.0   # outlier delta threshold
DS   = 3.0   # seed delta threshold
WZ   = 0.5   # z-weight: compress z relative to x/y so layers don't dominate

DEFAULT_PARAMS = [DC, RHOC, DO, DS, WZ]
PARAM_NAMES    = ['dc', 'rhoc', 'do', 'ds', 'wz']

# Parameter search bounds for MOPSO
LOWER_BOUNDS = [1.0,  0.1,  1.0,  1.0, 0.2]
UPPER_BOUNDS = [8.0, 10.0, 20.0, 20.0, 0.8]

# ── Objective weights ─────────────────────────────────────────────────────────
ALPHA  = 0.5   # EM weight; pion weight = 1 - ALPHA
LAMBDA = 2.0   # floor-penalty weight for mean N_T < 2

# ── MOPSO hyperparameters ─────────────────────────────────────────────────────
NUM_PARTICLES  = 30
NUM_ITERATIONS = 20
INERTIA        = 0.5
COGNITIVE      = 1.5
SOCIAL         = 1.5
MAX_PARETO     = 100
TOPOLOGY       = 'random'
RANDOM_SEED    = 42
