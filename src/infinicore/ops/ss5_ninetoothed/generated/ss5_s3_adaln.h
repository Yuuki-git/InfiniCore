#ifndef NINETOOTHED_SS5_S3_ADALN_H
#define NINETOOTHED_SS5_S3_ADALN_H

#include "ninetoothed.h"

#ifdef __cplusplus
extern "C" NineToothedResult launch_ss5_s3_adaln(NineToothedStream stream, NineToothedTensor normalized, NineToothedTensor scale, NineToothedTensor shift, NineToothedTensor output);
#else
NineToothedResult launch_ss5_s3_adaln(NineToothedStream stream, NineToothedTensor normalized, NineToothedTensor scale, NineToothedTensor shift, NineToothedTensor output);
#endif

#endif
