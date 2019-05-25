import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
DATA_ORIGIN = os.path.join(DATA, 'origin')
DATA_LABEL = os.path.join(DATA, 'label')

print(DATA_ORIGIN)