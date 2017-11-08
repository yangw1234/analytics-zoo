from bigdl.util.common import callBigDlFunc
from bigdl.util.common import JTensor
import numpy as np

from bigdl.nn.criterion import Criterion


class VanillaPGCriterion(Criterion):

    def __init__(self,
                 size_average=True,
                 bigdl_type="float"):
        super(VanillaPGCriterion, self).__init__(None,bigdl_type,
                                                size_average)
