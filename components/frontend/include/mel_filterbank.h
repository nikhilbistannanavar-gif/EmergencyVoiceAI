#ifndef MEL_FILTERBANK_H
#define MEL_FILTERBANK_H

#ifdef __cplusplus
extern "C" {
#endif

void mel_filterbank_init(void);

void mel_filterbank_compute(
    const float *magnitude,
    float *mel_output);

#ifdef __cplusplus
}
#endif

#endif