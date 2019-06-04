import os

# path
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
DATA_ORIGIN = os.path.join(DATA, 'origin')
DATA_LABEL = os.path.join(DATA, 'label')
DATA_PREP = os.path.join(DATA, 'prep')
DATA_AGGREGATED = os.path.join(DATA, 'aggregated')
DATA_FEATURES = os.path.join(DATA, 'features.json')
