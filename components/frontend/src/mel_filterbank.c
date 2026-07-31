#include "mel_filterbank.h"
#include "mel_weights.h"

#include <math.h>
#include <string.h>

void mel_filterbank_init(void)
{
    // No initialization required.
    // TensorFlow Mel weights are already stored in mel_weights.h
}

void mel_filterbank_compute(const float *magnitude,
                            float *mel_output)
{
    memset(mel_output, 0, sizeof(float) * MEL_BINS);

    /*
     * TensorFlow implementation:
     *
     * mel = spectrogram × mel_weight_matrix
     *
     * spectrogram : 513 bins
     * mel matrix  : 513 x 64
     * output      : 64 bins
     */

    for (int mel = 0; mel < MEL_BINS; mel++)
    {
        float sum = 0.0f;

        for (int fft = 0; fft < FFT_BINS; fft++)
        {
            sum += magnitude[fft] * tf_mel_weights[fft][mel];
        }

        /*
         * Match training notebook:
         *
         * tf.math.log(mel + 1e-6)
         */

        mel_output[mel] = logf(sum + 1e-6f);
    }
}