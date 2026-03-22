#include "main.h"

#include "tinyspeech_model.h"
#include "tinyspeech_test_inputs.h"

static const char *k_labels[TINYSPEECH_NUM_CLASSES] = {
    "yes", "no", "on", "off", "stop", "go"
};

static inline uint64_t rdcycle64(void) {
    uint64_t x;
    __asm__ volatile("rdcycle %0" : "=r"(x));
    return x;
}

static Tensor make_input_tensor(const int8_t *flat_data) {
    u_int8_t shape[4] = {1, 1, TINYSPEECH_TEST_INPUT_H, TINYSPEECH_TEST_INPUT_W};
    Tensor t = create_tensor(shape, 4);
    memcpy(t.data, flat_data, (size_t)TINYSPEECH_TEST_INPUT_SIZE * sizeof(int8_t));
    return t;
}

static void print_input_preview(const int8_t *flat_data) {
    printf("    input[0:23] =");
    for (int32_t i = 0; i < 24; i++) {
        printf(" %d", flat_data[i]);
    }
    printf("\n");
}

static int output_is_valid(const Tensor *probs, float *sum_out) {
    float sum = 0.0f;
    for (int32_t i = 0; i < probs->size; i++) {
        float v = probs->f_data[i];
        if (!isfinite(v) || (v < -1e-5f) || (v > 1.0005f)) {
            if (sum_out != NULL) {
                *sum_out = sum;
            }
            return 0;
        }
        sum += v;
    }

    if (sum_out != NULL) {
        *sum_out = sum;
    }

    return fabsf(sum - 1.0f) < 0.02f;
}

#if TINYSPEECH_DEBUG_TRACE
static void print_debug_trace(void) {
    const tinyspeech_debug_trace_t *trace = tinyspeech_debug_last_trace();

    printf("    logits(pre-softmax) =");
    for (int32_t i = 0; i < trace->logits_len; i++) {
        printf(" %.6f", trace->logits[i]);
    }
    printf("\n");

    printf("    layer-trace (%ld stages)\n", (long)trace->num_stages);
    for (int32_t i = 0; i < trace->num_stages; i++) {
        const tinyspeech_stage_checksum_t *s = &trace->stages[i];
        printf("      [%02ld] %-18s size=%ld sum=%.6f abs=%.6f min=%.6f max=%.6f\n",
               (long)i,
               s->name,
               (long)s->size,
               s->sum,
               s->abs_sum,
               s->min,
               s->max);
    }
}
#endif

void app_init(void) {
}

void app_main(void) {
    printf("=== DSP25 TinySpeech Inference Test ===\n");
    printf("  input shape: [1,1,%d,%d]\n", TINYSPEECH_TEST_INPUT_H, TINYSPEECH_TEST_INPUT_W);
    printf("  classes: %d (yes/no/on/off/stop/go)\n", TINYSPEECH_NUM_CLASSES);
    printf("  cases: %d\n", TINYSPEECH_TEST_NUM_CASES);

    uint32_t pass = 0;
    uint32_t fail = 0;
    uint32_t labeled_total = 0;
    uint32_t labeled_match = 0;

    for (uint32_t tc = 0; tc < TINYSPEECH_TEST_NUM_CASES; tc++) {
        const tinyspeech_test_input_case_t *c = &g_tinyspeech_test_inputs[tc];

        Tensor input = make_input_tensor(c->data);
        print_input_preview(c->data);

        uint64_t c0 = rdcycle64();
        Tensor probs = tinyspeech_run_inference(&input);
        uint64_t c1 = rdcycle64();

        float max_prob = 0.0f;
        int32_t pred = tinyspeech_argmax(&probs, &max_prob);
        float sum = 0.0f;
        int ok = output_is_valid(&probs, &sum);

        printf("[CASE %lu] %s\n", (unsigned long)tc, c->name);
        if ((c->expected_label >= 0) && (c->expected_label < TINYSPEECH_NUM_CLASSES)) {
            printf("    expected  = %s (%ld)\n", k_labels[c->expected_label], (long)c->expected_label);
        } else {
            printf("    expected  = <background/unknown>\n");
        }
        printf("    probs     =");
        for (int32_t i = 0; i < probs.size; i++) {
            printf(" %.6f", probs.f_data[i]);
        }
        printf("\n");
        if ((pred >= 0) && (pred < TINYSPEECH_NUM_CLASSES)) {
            printf("    predict   = %s (%ld), conf=%.6f\n", k_labels[pred], (long)pred, max_prob);
        } else {
            printf("    predict   = <out-of-range> (%ld), conf=%.6f\n", (long)pred, max_prob);
        }
        printf("    prob_sum  = %.6f\n", sum);
        printf("    cycles    = %lu\n", (unsigned long)(c1 - c0));

#if TINYSPEECH_DEBUG_TRACE
        print_debug_trace();
#endif

        if ((c->expected_label >= 0) && (c->expected_label < TINYSPEECH_NUM_CLASSES)) {
            labeled_total++;
            if (pred == c->expected_label) {
                labeled_match++;
            }
        }

        if (ok) {
            pass++;
            printf("    status    = PASS\n\n");
        } else {
            fail++;
            printf("    status    = FAIL (invalid probability vector)\n\n");
        }

        free_tensor(&probs);
    }

    printf("TinySpeech summary: pass=%lu fail=%lu\n",
           (unsigned long)pass,
           (unsigned long)fail);
    printf("Label-match summary: %lu/%lu labeled cases\n",
           (unsigned long)labeled_match,
           (unsigned long)labeled_total);
}

int main(void) {
    app_init();
    app_main();
    return 0;
}
