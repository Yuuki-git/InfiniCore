#ifndef NINETOOTHED_SS5_S4_ATTENTION_H
#define NINETOOTHED_SS5_S4_ATTENTION_H

#include "ninetoothed.h"

#ifdef __cplusplus
extern "C" NineToothedResult launch_ss5_s4_attention(NineToothedStream stream, NineToothedTensor query, NineToothedTensor key, NineToothedTensor value, NineToothedTensor scale, NineToothedTensor output);
#else
NineToothedResult launch_ss5_s4_attention(NineToothedStream stream, NineToothedTensor query, NineToothedTensor key, NineToothedTensor value, NineToothedTensor scale, NineToothedTensor output);
#endif

#endif
