# ── Data ───────────────────────────────────────────────────────────────────────
DATA_DIR = 'Data'

FILES = {
    'ee':    'e_energy.root',   # 500 two-electron events
    'epion': 'pi_energy.root',  # 500 electron-pion events
}

TRUTH_BRANCH = 'ticlDumper/simtrackstersCP'

BRANCHES = [
    'vertices_indexes',
    'vertices_x',
    'vertices_y',
    'vertices_z',
    'vertices_energy',
    'vertices_multiplicity',
]

# ── Subdetector boundary ───────────────────────────────────────────────────────
# CLUEstering is run separately on each region (different detector geometry),
# but quality is scored globally across both.
CEE_Z_BOUNDARY = 352.0   # |z| < 352 cm  => CEE (electromagnetic)
                         # |z| >= 352 cm  => CHE (hadronic)
SUBDETS = ['CEE', 'CHE']

# ── Parameters — one joint 10-element vector [CEE×5, CHE×5] ───────────────────
PARAM_NAMES = [
    'density_radius_cee', 'min_density_cee', 'outlier_distance_cee',
    'seeding_distance_cee', 'w_z_cee',
    'density_radius_che', 'min_density_che', 'outlier_distance_che',
    'seeding_distance_che', 'w_z_che',
]

#                             CEE                              CHE
LOWER_BOUNDS = [0.5,  0.5,  0.5,  0.5,  0.1,    0.5,  0.5,  0.5,  0.5,  0.1]
UPPER_BOUNDS = [5.0, 10.0, 10.0, 10.0,  5.0,    8.0, 15.0, 15.0, 15.0,  5.0]
DEFAULT_PARAMS = [2.0,  2.0,  2.0,  2.0,  1.0,  3.0,  3.0,  3.0,  3.0,  1.0]

# ── MOPSO hyperparameters ──────────────────────────────────────────────────────
NUM_PARTICLES  = 30
NUM_ITERATIONS = 50
INERTIA        = 0.4
COGNITIVE      = 1.5
SOCIAL         = 2.0
MAX_PARETO     = 100
TOPOLOGY       = 'random'
RANDOM_SEED    = 42
