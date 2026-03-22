#include "misc.h"

static inline float tensor_get_value(const Tensor *t, int32_t idx) {
    if (t->f_data != NULL) {
        return t->f_data[idx];
    }
    return (float)t->data[idx];
}

static inline float tensor_get_broadcast(const Tensor *t, int32_t idx) {
    if (t->size <= 1) {
        return tensor_get_value(t, 0);
    }
    return tensor_get_value(t, idx);
}

float compute_mean_abs(int32_t *w, int32_t len) {
    float sum = 0.0f;
    for (int32_t i = 0; i < len; i++) {
        sum += fabsf((float)w[i]);
    }
    return sum / (float)len;
}

int8_t clamp(int8_t val, int8_t min_val, int8_t max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

void quantize_weights(Tensor *w, Tensor *u, float *scale, u_int8_t retain_float) {
    float s = (scale != NULL) ? *scale : 1.0f;
    if (fabsf(s) < 1e-12f) {
        s = 1.0f;
    }

    for (int32_t i = 0; i < u->size; i++) {
        float q = roundf(w->f_data[i] / s);
        if (q < -127.0f) q = -127.0f;
        if (q > 127.0f) q = 127.0f;

        if (retain_float == CONVERT_FLOAT) {
            u->f_data[i] = q;
        } else {
            u->data[i] = (int8_t)q;
        }
    }
}

void dequantize_weights(Tensor *quantized_weights, Tensor *dequantized_weights, float scale) {
    for (int32_t i = 0; i < dequantized_weights->size; i++) {
        float q = tensor_get_value(quantized_weights, i);
        dequantized_weights->f_data[i] = q * scale;
    }
}

Tensor sigmoid(Tensor *tensor) {
    Tensor output = f_create_tensor(tensor->shape, 4);

    for (int32_t i = 0; i < tensor->size; i++) {
        float x = tensor_get_value(tensor, i);
        output.f_data[i] = 1.0f / (1.0f + expf(-x));
    }

    free_tensor(tensor);
    return output;
}

void attention(Tensor *residual, Tensor *S, Tensor *scale) {
    for (int32_t i = 0; i < S->size; i++) {
        float r = tensor_get_value(residual, i);
        float g = tensor_get_broadcast(scale, i);
        S->f_data[i] = r + (r * g * S->f_data[i]);
    }
}

float mean(int8_t *data, int size) {
    float sum = 0.0f;
    for (int32_t i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / (float)size;
}

void relu(Tensor *input) {
    if (input->data != NULL) {
        for (int32_t i = 0; i < input->size; i++) {
            if (input->data[i] < 0) {
                input->data[i] = 0;
            }
        }
        return;
    }

    for (int32_t i = 0; i < input->size; i++) {
        if (input->f_data[i] < 0.0f) {
            input->f_data[i] = 0.0f;
        }
    }
}
