from bigdl.util.common import callBigDlFunc
from bigdl.util.common import JTensor
import numpy as np

from bigdl.nn.criterion import Criterion


class VanillaPGCriterion(Criterion):

    def __init__(self,
                 clipping = False,
                 clipping_range = 0.2,
                 size_average=True,
                 bigdl_type="float"):
        super(VanillaPGCriterion, self).__init__(None,bigdl_type,
                                                clipping, 
                                                clipping_range, 
                                                size_average)

class PGCriterion(Criterion):

    def __init__(self,
                 is_clip = False,
                 clip_pos = 1.2,
                 clip_neg = 0.8,
                 size_average=True,
                 bigdl_type="float"):
        super(PGCriterion, self).__init__(None,bigdl_type,
                                                 is_clip,
                                                 clip_pos,
                                                 clip_neg,
                                                 size_average)


class RFPGCriterion(Criterion):

    def __init__(self,
                 beta = 0.01,
                 epsilon = 1e-8,
                 bigdl_type="float"):
        super(RFPGCriterion, self).__init__(None,bigdl_type,
                                                 beta,
                                                 epsilon)