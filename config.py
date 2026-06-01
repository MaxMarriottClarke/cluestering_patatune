DATA_DIR = 'Data'

FILES = {
    'ee':    'e_energy.root',   # 500 two-electron events
    'pi': 'pi_energy.root',  # 500 two-pion events
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

# CLUEstering is run separately on CEE and CHE (different detector geometry),
# but quality is scored globally across both.
#
# CEE/CHE boundary: midpoint between layer 26 (|z|=361.1 cm) and layer 27 (|z|=368.0 cm)
#
#   |z| <  364.55 cm   => CEE  (layers 1-26,  EM silicon)
#   |z| >= 364.55 cm   => CHE  (layers 27+,   hadronic silicon)
CEE_Z_BOUNDARY = 364.55   # cm — midpoint between layer 26 and 27

SUBDETS = ['CEE', 'CHE']

PARAM_NAMES = [
    'density_radius_cee',  'min_density_cee',  'outlier_distance_cee',
    'seeding_distance_cee', 'w_z_cee',
    'density_radius_che',  'min_density_che',  'outlier_distance_che',
    'seeding_distance_che', 'w_z_che',
]

#                              CEE                       CHE
LOWER_BOUNDS  = [1.0,  0.01,  3.0,  1.0,  0.4,    1.0,  0.01,  1.0,  1.0,  0.2]
UPPER_BOUNDS  = [6.0, 10.0, 20.0, 20.0,  1.0, 10.0, 10.0, 30.0, 30.0,  1.0]
DEFAULT_PARAMS = [2.85, 3.31, 4.50, 4.50, 0.859, 2.38, 2.83, 4.60, 5.53, 0.477]

# 6 ratio objectives: each raw metric divided by the corresponding CLUE3D value.
# Values < 1 = better than CLUE3D.  CLUE3D maps to the point (1,1,1,1,1,1).
OBJ_NAMES = [
    'R1_ee',    'R2_ee',    'R3_ee',
    'R1_pi', 'R2_pi', 'R3_pi',
]

# CLUE3D Baselin
# These are the CLUE3D values on the same 1000-event dataset used for tuning.
# Used as the normalisation denominator so each objective has target < 1.
CLUE3D_BASELINES = {
    'ee': {
        'f1': 0.08188021960670597,
        'f2': 0.07743506729522537,
        'f3': 0.64,
    },
    'pi': {
        'f1': 0.3004829949293515,
        'f2': 0.42645414425151495,
        'f3': 9.386,
    },
}

N_EVENTS = 1000

# Number of worker processes for parallel event evaluation within each
# objective call.  Set to match request_cpus in condor.sub.
# Rule of thumb: 8 workers × 1000 events → ~6.5h on workday flavour.
N_JOBS = 8

# MOPSO Hyperparameters
NUM_PARTICLES  = 100
NUM_ITERATIONS = 150
INERTIA        = 0.6
COGNITIVE      = 1.5
SOCIAL         = 1.5
MAX_PARETO     = 100
TOPOLOGY       = 'random'
RANDOM_SEED    = 42

OUTPUT_DIR = 'results'
