import runpy, sys
from pathlib import Path
overlay=Path('/diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930/LLAMA2_UNIFORM_EB059_STRICT_MATCH_V1/engineering_fix/overlay')
repo=Path('/home/bai-project/jiaqizhang/EigenAttn/EigenAttn')
sys.path.insert(0,str(overlay))
sys.path.insert(1,str(repo))
runpy.run_path(str(repo/'main_eigen_attn.py'),run_name='__main__')
