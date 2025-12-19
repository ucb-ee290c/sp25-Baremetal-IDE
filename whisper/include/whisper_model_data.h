// In-memory version of the whisper model file. The actual data is provided by the
// file_to_c_data.py script whish is run automatically by the build system.

#include <stddef.h>

extern char whisper_model_file_data[];
extern size_t whisper_model_file_data_len;