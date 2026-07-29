#ifndef MEL_FILTERBANK_H
#define MEL_FILTERBANK_H

#include <stdint.h>

#define SAMPLE_RATE 16000

void mel_filterbank_init(void);

void mel_filterbank_compute(
    const float *magnitude,
    float *mel_output
);

#endif