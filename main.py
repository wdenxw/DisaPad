"""The main implementation of DisPad
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from train import run
from settings import settings

rn = run()
st = settings()
args, unknown = st.parser.parse_known_args()
if __name__ == "__main__":
    rn.train()
