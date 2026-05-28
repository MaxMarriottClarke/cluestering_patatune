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

# ── Subdetector boundaries ─────────────────────────────────────────────────────
# CLUEstering is run separately on CEE and CHE (different detector geometry),
# but quality is scored globally across both.
#
# CEE/CHE boundary: midpoint between layer 26 (|z|=361.1 cm) and layer 27 (|z|=368.0 cm)
#
#   |z| <  364.55 cm   => CEE  (layers 1-26,  EM silicon)
#   |z| >= 364.55 cm   => CHE  (layers 27+,   hadronic silicon)
CEE_Z_BOUNDARY = 364.55   # cm — midpoint between layer 26 and 27

SUBDETS = ['CEE', 'CHE']

# ── Parameters — one joint 10-element vector [CEE×5, CHE×5] ──────────────────
PARAM_NAMES = [
    'density_radius_cee',  'min_density_cee',  'outlier_distance_cee',
    'seeding_distance_cee', 'w_z_cee',
    'density_radius_che',  'min_density_che',  'outlier_distance_che',
    'seeding_distance_che', 'w_z_che',
]

#                              CEE                       CHE
LOWER_BOUNDS  = [0.5,  0.5,  0.5,  0.5,  0.1,    0.5,  0.5,  0.5,  0.5,  0.1]
UPPER_BOUNDS  = [5.0, 10.0, 10.0, 10.0,  1.0,    8.0, 15.0, 15.0, 15.0,  1.0]
DEFAULT_PARAMS = [2.85, 3.31, 4.50, 4.50, 0.859, 2.38, 2.83, 4.60, 5.53, 0.477]

# ── Event subsampling ─────────────────────────────────────────────────────────
# Number of events used during optimisation (per particle type, so total = 2×).
# Kept balanced: N_EVENTS//2 ee + N_EVENTS//2 epion, interleaved.
# After optimisation, re-evaluate the Pareto front on the full dataset.
# Set to None to use all available events.
N_EVENTS = 1000

# ── Parallelism ───────────────────────────────────────────────────────────────
# Number of worker processes for parallel event evaluation within each
# objective call.  Set to match request_cpus in condor.sub.
# Rule of thumb: 8 workers × 1000 events → ~6.5h on workday flavour.
N_JOBS = 8

# ── MOPSO hyperparameters ──────────────────────────────────────────────────────
NUM_PARTICLES  = 100
NUM_ITERATIONS = 100
INERTIA        = 0.4
COGNITIVE      = 1.5
SOCIAL         = 2.0
MAX_PARETO     = 100
TOPOLOGY       = 'random'
RANDOM_SEED    = 42

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = 'results'
