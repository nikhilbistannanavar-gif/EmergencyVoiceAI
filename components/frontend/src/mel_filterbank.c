#include "mel_filterbank.h"
#include "frontend.h"

#include <math.h>

#define MEL_LOW_FREQ      20.0f
#define MEL_HIGH_FREQ     4000.0f
#define NUM_FFT_BINS      (FFT_LENGTH / 2 + 1)

static int fft_bins[MEL_BINS + 2];

static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void mel_filterbank_init(void)
{
    float mel_low = hz_to_mel(MEL_LOW_FREQ);
    float mel_high = hz_to_mel(MEL_HIGH_FREQ);

    float mel_step = (mel_high - mel_low) / (MEL_BINS + 1);

    for (int i = 0; i < MEL_BINS + 2; i++)
    {
        float mel = mel_low + (float)i * mel_step;
        float hz = mel_to_hz(mel);

        fft_bins[i] = (int)((FFT_LENGTH + 1) * hz / SAMPLE_RATE);

        if (fft_bins[i] < 0)
            fft_bins[i] = 0;

        if (fft_bins[i] >= NUM_FFT_BINS)
            fft_bins[i] = NUM_FFT_BINS - 1;
    }
}

void mel_filterbank_compute(const float *magnitude,
                            float *mel_output)
{
    for (int m = 0; m < MEL_BINS; m++)
    {
        float sum = 0.0f;

        int left   = fft_bins[m];
        int center = fft_bins[m + 1];
        int right  = fft_bins[m + 2];

        if (center <= left)
            center = left + 1;

        if (right <= center)
            right = center + 1;

        for (int k = left; k < center; k++)
        {
            float weight =
                (float)(k - left) /
                (float)(center - left);

            sum += magnitude[k] * weight;
        }

        for (int k = center; k < right; k++)
        {
            float weight =
                (float)(right - k) /
                (float)(right - center);

            sum += magnitude[k] * weight;
        }

        mel_output[m] = logf(sum + 1e-6f);
    }
}