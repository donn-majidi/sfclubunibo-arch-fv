import numpy as np
from arch.bootstrap.base import optimal_block_length
from arch.bootstrap.base import _single_optimal_block
from dataclasses import dataclass

@dataclass
class BootStrapBSResult:
    """Results container for optimal bootstrap block sizes."""
    b_sb: int       #
    b_sb_min: int   #
    b_cb: int       #
    b_cb_min: int   #

def optimal_block_size(x: Union[ArrayLike1D, ArrayLike2D]) -> pd.DataFrame:
    nobs = x.shape[0]
    b_max = np.ceil(min(3 * np.sqrt(nobs), nobs / 3))