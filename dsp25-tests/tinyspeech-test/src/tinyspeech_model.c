#include "tinyspeech_model.h"

#include "misc.h"
#include "modules.h"
#include "weights.h"

#include <stdint.h>

static inline Tensor *W(u_int8_t idx) {
    return model_weights[idx].address;
}

static Tensor attention_condenser(Tensor *input, u_int8_t *layer_id) {
    u_int8_t base = *layer_id; /* scale tensor index */

    Tensor q = maxpool2d(input, 2, 2);
    Tensor k = conv2d(&q, W(base + 1), W(base + 2), W(base + 3), 1, 1);
    k = conv2d(&k, W(base + 5), W(base + 6), W(base + 7), 1, 1);
    k = upsample_nearest(&k, 2);
    k = conv2d(&k, W(base + 9), W(base + 10), W(base + 11), 1, 1);
    k = sigmoid(&k);
    attention(input, &k, W(base));

    *layer_id = (u_int8_t)(base + 13); /* BN gamma index */
    return k;
}

static Tensor attn_bn_block(Tensor *input, u_int8_t *layer_id) {
    Tensor x = attention_condenser(input, layer_id);
    free_tensor(input);

    x = batchnorm2d(&x, W(*layer_id), W(*layer_id + 1), W(*layer_id + 2), W(*layer_id + 4), W(*layer_id + 5));
    *layer_id = (u_int8_t)(*layer_id + 6); /* next attention scale index */

    Tensor y = attention_condenser(&x, layer_id);
    free_tensor(&x);

    y = batchnorm2d(&y, W(*layer_id), W(*layer_id + 1), W(*layer_id + 2), W(*layer_id + 4), W(*layer_id + 5));
    *layer_id = (u_int8_t)(*layer_id + 6); /* next block scale index */

    return y;
}

Tensor tinyspeech_run_inference(Tensor *input) {
    Tensor x = conv2d(input, W(0), W(1), W(2), 1, 1);
    relu(&x);

    u_int8_t layer_id = 4; /* first block scale tensor */

    x = attn_bn_block(&x, &layer_id);
    x = attn_bn_block(&x, &layer_id);
    x = attn_bn_block(&x, &layer_id);
    x = attn_bn_block(&x, &layer_id);

    x = conv2d(&x, W(layer_id), W(layer_id + 1), W(layer_id + 2), 1, 1);
    relu(&x);

    Tensor pooled = adaptive_avg_pool2d(&x);
    free_tensor(&x);

    Tensor probs = fc_layer(&pooled, W(layer_id + 4));
    free_tensor(&pooled);

    softmax(&probs);
    return probs;
}

int32_t tinyspeech_argmax(const Tensor *probs, float *max_prob) {
    int32_t best_idx = 0;
    float best_val = probs->f_data[0];

    for (int32_t i = 1; i < probs->size; i++) {
        if (probs->f_data[i] > best_val) {
            best_val = probs->f_data[i];
            best_idx = i;
        }
    }

    if (max_prob != NULL) {
        *max_prob = best_val;
    }

    return best_idx;
}
