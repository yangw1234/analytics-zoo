from bigdl.util.common import callBigDlFunc
from bigdl.util.common import JTensor
import numpy as np

from bigdl.nn.criterion import Criterion

class DiscretePPOCriterion(Criterion):

    def __init__(self,
                 epsilon=0.3,
                 entropy_coeff=0.0,
                 kl_target=0.01,
                 init_beta=0.0,
                 bigdl_type="float"):
        super(DiscretePPOCriterion, self).__init__(None,
                                          bigdl_type,
                                          epsilon,
                                          entropy_coeff,
                                          kl_target,
                                          init_beta)

class ContinuousPPOCriterion(Criterion):

    def __init__(self,
                 epsilon=0.3,
                 entropy_coeff=0.0,
                 kl_target=0.01,
                 init_beta=0.0,
                 bigdl_type="float"):
        super(ContinuousPPOCriterion, self).__init__(None,
                                           bigdl_type,
                                           epsilon,
                                           entropy_coeff,
                                           kl_target,
                                           init_beta)