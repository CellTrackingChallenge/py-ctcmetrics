from ctc_metrics.metrics.validation.valid import valid
from ctc_metrics.metrics.biological.bc import bc
from ctc_metrics.metrics.biological.ct import ct
from ctc_metrics.metrics.biological.cca import cca
from ctc_metrics.metrics.biological.tf import tf
from ctc_metrics.metrics.technical.seg import seg
from ctc_metrics.metrics.technical.tra import tra
from ctc_metrics.metrics.technical.det import det
from ctc_metrics.metrics.technical.op_ctb import op_ctb
from ctc_metrics.metrics.technical.op_csb import op_csb
from ctc_metrics.metrics.biological.bio import bio
from ctc_metrics.metrics.biological.op_clb import op_clb
from ctc_metrics.metrics.technical.lnk import lnk

ALL_METRICS = [
    "Valid", "BC", "CT", "CCA", "TF", "SEG", "TRA", "DET", "LNK", "OP_CTB",
    "OP_CSB", "BIO", "OP_CLB"
]
