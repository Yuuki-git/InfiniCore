#ifndef NINETOOTHED_SS5_S3_GATE_H
#define NINETOOTHED_SS5_S3_GATE_H

#include "ninetoothed.h"

#ifdef __cplusplus
extern "C" NineToothedResult launch_ss5_s3_gate(NineToothedStream stream, NineToothedTensor residual, NineToothedTensor branch, NineToothedTensor gate, NineToothedTensor output);
#else
NineToothedResult launch_ss5_s3_gate(NineToothedStream stream, NineToothedTensor residual, NineToothedTensor branch, NineToothedTensor gate, NineToothedTensor output);
#endif

#endif
