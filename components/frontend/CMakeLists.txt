idf_component_register(
    SRCS
    "src/frontend.c"
    "src/mel_filterbank.c"
    INCLUDE_DIRS "include"
    REQUIRES esp-dsp
)