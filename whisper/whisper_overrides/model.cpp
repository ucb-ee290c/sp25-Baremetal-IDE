#include <string.h>
#include <stddef.h>
#include "whisper.h"
#include "whisper_model_data.h"

static size_t file_pos = 0;

// This function is a drop-in replacement for whisper.cpp's model loader generator.
// This creates a custom loader that instead of using a std::ifstream uses a combination
// of memcpy and the in-memory model from whisper_model_data.h to essentially create
// an in-memory file.
struct whisper_context * whisper_init_from_file_with_params_no_state(const char * path_model, struct whisper_context_params params) {

    // auto fin = std::ifstream(path_model, std::ios::binary);

    whisper_model_loader loader = {};

    loader.context = &file_pos;

    // Copy from the in-memory file at the current offset and update the offset
    loader.read = [](void * ctx, void * output, size_t read_size) {
        size_t *pos = (size_t *) ctx;
        size_t remaining = whisper_model_file_data_len - *pos;
        size_t copy_size = (remaining > read_size) ? read_size : remaining;
        memcpy(output, whisper_model_file_data + *pos, copy_size);
        *pos += copy_size;
        return copy_size;
        // std::ifstream * fin = (std::ifstream*)ctx;
        // fin->read((char *)output, read_size);
        // return read_size;
    };

    // Check if the current offset is at or past the length of the file
    loader.eof = [](void * ctx) {
        size_t *pos = (size_t *) ctx;
        return *pos >= whisper_model_file_data_len;
        // std::ifstream * fin = (std::ifstream*)ctx;
        // return fin->eof();
    };

    // Reset the current offset. I don't know if this is actually needed or if the
    // file is only read once. Since pos is a static global variable, this also
    // means only one instance of the loader can work at a time, but this seems to
    // be fine.
    loader.close = [](void * ctx) {
        size_t *pos = (size_t *) ctx;
        *pos = 0;
        // std::ifstream * fin = (std::ifstream*)ctx;
        // fin->close();
    };

    auto ctx = whisper_init_with_params_no_state(&loader, params);

    // if (ctx) {
    //     ctx->path_model = path_model;
    // }

    return ctx;
}