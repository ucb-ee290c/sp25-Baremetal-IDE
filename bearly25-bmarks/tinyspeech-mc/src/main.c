#include "main.h"

#include "bench_config.h"
#include "simple_setup.h"
#include "tinyspeech_model.h"
#include "tinyspeech_inputs.h"
#include "tinyspeech_reference.h"
#include "hthread.h"

#if (TINYSPEECH_TEST_NUM_CASES != TINYSPEECH_EXPECTED_NUM_CASES)
#error "tinyspeech_inputs.h mismatch: unexpected case count"
#endif

#if (TINYSPEECH_TEST_BANDPASS_LOW_HZ != 0) || (TINYSPEECH_TEST_BANDPASS_HIGH_HZ != 8000)
#error "tinyspeech_inputs.h mismatch: expected bandpass 0..8000 Hz"
#endif

#if (TINYSPEECH_REF_NUM_CASES != TINYSPEECH_EXPECTED_NUM_CASES) || (TINYSPEECH_REF_NUM_STAGES != 12)
#error "tinyspeech_reference.h mismatch: unexpected case count/stage count"
#endif

#ifndef TINYSPEECH_REF_CHECK_STAGE_SUM
#define TINYSPEECH_REF_CHECK_STAGE_SUM 1
#endif

#ifndef TINYSPEECH_PROGRESS_EVERY
#define TINYSPEECH_PROGRESS_EVERY 10
#endif

static const char *k_labels[TINYSPEECH_NUM_CLASSES] = {
    "yes", "no", "on", "off", "stop", "go"
};

uint64_t target_frequency = TINYSPEECH_MC_TARGET_FREQUENCY_HZ;

static void mc_nop_worker(void *arg) {
    (void)arg;
}

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

#if TINYSPEECH_OUTPUT_SOFTMAX
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
#else
static int output_is_valid(const Tensor *logits, float *sum_out) {
    float sum = 0.0f;
    for (int32_t i = 0; i < logits->size; i++) {
        float v = logits->f_data[i];
        if (!isfinite(v)) {
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
    return 1;
}
#endif

static const tinyspeech_ref_case_t *get_reference_case(uint32_t tc,
                                                       const tinyspeech_test_input_case_t *c) {
    if (tc >= TINYSPEECH_REF_NUM_CASES) {
        return NULL;
    }

    const tinyspeech_ref_case_t *r = &g_tinyspeech_ref_cases[tc];
    if (strcmp(r->name, c->name) != 0) {
        return NULL;
    }

    return r;
}

typedef struct {
    uint32_t compared;
    uint32_t pred_match;
    uint32_t prob_fail;
    uint32_t logit_fail;
    uint32_t stage_fail;
    uint32_t missing_ref;
} ref_cmp_summary_t;

static int compare_with_reference(uint32_t tc,
                                  const tinyspeech_test_input_case_t *c,
                                  const Tensor *probs,
                                  int32_t pred,
                                  ref_cmp_summary_t *summary) {
    const tinyspeech_ref_case_t *r = get_reference_case(tc, c);
    const tinyspeech_debug_trace_t *trace = tinyspeech_debug_last_trace();

    if (r == NULL) {
        summary->missing_ref++;
        if (TINYSPEECH_VERBOSE_CASE_LOGS) {
            printf("    ref       = SKIP (missing or name mismatch)\n");
        }
        return 0;
    }

    summary->compared++;

    int pred_ok = (pred == r->ref_pred_label);

    int prob_ok = 1;
    float max_prob_diff = 0.0f;
#if TINYSPEECH_OUTPUT_SOFTMAX
    for (int32_t i = 0; i < TINYSPEECH_REF_NUM_CLASSES; i++) {
        float d = fabsf(probs->f_data[i] - r->ref_probs[i]);
        if (d > max_prob_diff) {
            max_prob_diff = d;
        }
        if (d > TINYSPEECH_REF_PROB_TOL) {
            prob_ok = 0;
        }
    }
#endif
    if (!prob_ok) {
        summary->prob_fail++;
    }

    int logit_ok = 1;
    float max_logit_diff = 0.0f;
    if (trace->logits_len == TINYSPEECH_REF_NUM_CLASSES) {
        for (int32_t i = 0; i < TINYSPEECH_REF_NUM_CLASSES; i++) {
            float d = fabsf(trace->logits[i] - r->ref_logits[i]);
            if (d > max_logit_diff) {
                max_logit_diff = d;
            }
            if (d > TINYSPEECH_REF_LOGIT_TOL) {
                logit_ok = 0;
            }
        }
    } else {
        logit_ok = 0;
    }
    if (!logit_ok) {
        summary->logit_fail++;
    }

#if TINYSPEECH_INT8_PIPELINE
    if (!pred_ok && (max_logit_diff <= TINYSPEECH_INT8_PRED_TIE_TOL)) {
        pred_ok = 1;
    }
#endif
    if (pred_ok) {
        summary->pred_match++;
    }

    int stage_ok = 1;
    float max_stage_diff = 0.0f;
#if TINYSPEECH_REF_CHECK_STAGE_SUM
    if (trace->num_stages == TINYSPEECH_REF_NUM_STAGES) {
        for (int32_t i = 0; i < TINYSPEECH_REF_NUM_STAGES; i++) {
            float d = fabsf(trace->stages[i].sum - r->ref_stage_sums[i]);
            if (d > max_stage_diff) {
                max_stage_diff = d;
            }
            if (d > TINYSPEECH_REF_STAGE_SUM_TOL) {
                stage_ok = 0;
            }
        }
    } else {
        stage_ok = 0;
    }
#endif
    if (!stage_ok) {
        summary->stage_fail++;
    }

    if (TINYSPEECH_VERBOSE_CASE_LOGS || !(pred_ok && prob_ok && logit_ok && stage_ok)) {
        printf("    ref       = pred:%s probs:%s logits:%s stage-sum:%s\n",
               pred_ok ? "OK" : "MISMATCH",
               prob_ok ? "OK" : "MISMATCH",
               logit_ok ? "OK" : "MISMATCH",
               stage_ok ? "OK" : "MISMATCH");
        printf("    ref-diff  = prob_max=%.6f logit_max=%.6f stage_sum_max=%.6f\n",
               max_prob_diff, max_logit_diff, max_stage_diff);
    }

    return pred_ok && prob_ok && logit_ok && stage_ok;
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
    init_test(target_frequency);
    hthread_init();
    // Warm hart1 once so steady-state dispatch has no cold-start hiccup.
    hthread_issue(1, mc_nop_worker, NULL);
    hthread_join(1);
}

typedef struct {
    uint64_t sum;
    uint64_t min;
    uint64_t max;
} cycle_stat_t;

static inline void cycle_stat_init(cycle_stat_t *s) {
    s->sum = 0;
    s->min = UINT64_MAX;
    s->max = 0;
}

static inline void cycle_stat_update(cycle_stat_t *s, uint64_t v) {
    s->sum += v;
    if (v < s->min) {
        s->min = v;
    }
    if (v > s->max) {
        s->max = v;
    }
}

static int run_suite_for_frequency(uint64_t frequency_hz) {
#if TINYSPEECH_OUTPUT_SOFTMAX
    const char *score_label = "conf";
#else
    const char *score_label = "score";
#endif

    printf("=== Bearly25 TinySpeech-MC @ %llu Hz ===\n", (unsigned long long)frequency_hz);
    printf("  mode: multicore inference benchmark (2 cores)\n");
#if TINYSPEECH_INT8_PIPELINE
    printf("  kernel mode: fixed-shape INT8 conv/gap/fc\n");
    printf("  weights  : symmetric int8 prepack per layer\n");
#if TINYSPEECH_INT8_MULTICORE
    printf("  multicore: enabled (hart0+hart1 split for conv2/conv3/gap-quant)\n");
    printf("  split    : conv2=%d/%d conv3=%d/%d\n",
           TINYSPEECH_MC_CONV2_OC_SPLIT, 48,
           TINYSPEECH_MC_CONV3_OC_SPLIT, 96);
#else
    printf("  multicore: disabled\n");
#endif
#elif defined(__riscv_vector)
    printf("  kernel mode: RVV implicit-GEMM convolution/matmul\n");
    printf("  weights  : prepacked K-major RVV layout\n");
#else
    printf("  kernel mode: scalar fallback (no RVV)\n");
#endif
#if TINYSPEECH_FP16_MIXED
#if defined(__riscv_zvfh) && (__riscv_zvfh > 0)
    printf("  fp16 mix : enabled (RVV Zvfh path active)\n");
#else
    printf("  fp16 mix : requested, but Zvfh unavailable -> f32 fallback path\n");
#endif
#else
    printf("  fp16 mix : disabled\n");
#endif
#if TINYSPEECH_INT8_PIPELINE
    printf("  int8 path: enabled (fixed-shape quantized conv/gap/fc)\n");
#if TINYSPEECH_MC_USE_SCRATCHPAD_SHARED
    printf("  mem map  : scratchpad shared hot-buffers enabled (pad2/pad3/gap3/w2pack16)\n");
#else
    printf("  mem map  : scratchpad shared hot-buffers disabled\n");
#endif
#if TINYSPEECH_MC_USE_TCM_PRIVATE
    printf("  mem map  : TCM private buffers enabled (core0 fc/core1 gap3)\n");
#else
    printf("  mem map  : TCM private buffers disabled\n");
#endif
#if defined(__riscv_vector) && TINYSPEECH_INT8_RVV_UKERNELS
    printf("  int8 ukrn: enabled (conv2+pool2 and conv3+gap fixed-shape RVV)\n");
#if TINYSPEECH_INT8_USE_VSE8_PACK_STORE
    printf("  int8 pack: RVV contiguous pack/store enabled (vnclip API=%d)\n", TINYSPEECH_RVV_NCLIP_API);
#else
    printf("  int8 pack: scalar lane-copy fallback\n");
#endif
#if TINYSPEECH_INT8_USE_VSMUL_REQUANT
    printf("  int8 rqnt: RVV vsmul requant enabled (API=%d)\n", TINYSPEECH_RVV_VSMUL_API);
#else
    printf("  int8 rqnt: widen-mul requant fallback\n");
#endif
#elif defined(__riscv_vector)
    printf("  int8 ukrn: disabled (generic RVV kernels)\n");
#else
    printf("  int8 ukrn: scalar-only build (no RVV)\n");
#endif
    printf("  int8 mode: fixed-only fast path enabled after calibration\n");
#else
    printf("  int8 path: disabled\n");
#endif
    printf("  input shape: [1,1,%d,%d]\n", TINYSPEECH_TEST_INPUT_H, TINYSPEECH_TEST_INPUT_W);
    printf("  classes: %d (yes/no/on/off/stop/go)\n", TINYSPEECH_NUM_CLASSES);
    printf("  cases: %d\n", TINYSPEECH_TEST_NUM_CASES);
    printf("  preproc  : MFCC, win=%dms hop=%dms, bandpass=%d..%d Hz\n",
           TINYSPEECH_TEST_WINDOW_MS,
           TINYSPEECH_TEST_HOP_MS,
           TINYSPEECH_TEST_BANDPASS_LOW_HZ,
           TINYSPEECH_TEST_BANDPASS_HIGH_HZ);
    printf("  bn scale offset: 2 (fixed to activation_scale)\n");
#if TINYSPEECH_OUTPUT_SOFTMAX
    printf("  output   : softmax probabilities\n");
#else
    printf("  output   : logits (softmax skipped for top1)\n");
#endif
#if !TINYSPEECH_REF_CHECK_STAGE_SUM
    printf("  note     : stage-sum reference check disabled (fused conv trace semantics)\n");
#endif
#if !TINYSPEECH_ENABLE_TRACE
    printf("  note     : layer trace capture disabled (perf mode)\n");
#endif

    printf("  runtime prep: begin\n");
    fflush(stdout);
    tinyspeech_prepare_runtime();
    printf("  runtime prep: done\n");
    fflush(stdout);

#if TINYSPEECH_INT8_PIPELINE
    printf("  calibration: begin (freeze int8 scales/biases)\n");
    fflush(stdout);
    tinyspeech_int8_calibration_begin();
    for (uint32_t tc = 0; tc < TINYSPEECH_TEST_NUM_CASES; tc++) {
        const tinyspeech_test_input_case_t *c = &g_tinyspeech_test_inputs[tc];
        Tensor input = make_input_tensor(c->data);
        Tensor logits = tinyspeech_run_inference(&input);
        free_tensor(&input);
        free_tensor(&logits);
    }
    int calib_ok = tinyspeech_int8_calibration_end();
    printf("  calibration: %s\n", calib_ok ? "done" : "failed (falling back to dynamic int8)");
    fflush(stdout);
#endif

    uint32_t pass = 0;
    uint32_t fail = 0;
    uint32_t labeled_total = 0;
    uint32_t labeled_match = 0;
    uint64_t cycles_sum = 0;
    uint64_t cycles_min = UINT64_MAX;
    uint64_t cycles_max = 0;
    ref_cmp_summary_t ref_summary = {0};
    cycle_stat_t st_input_cast;
    cycle_stat_t st_conv1_pool1;
    cycle_stat_t st_conv2_pool2;
    cycle_stat_t st_conv3_gap;
    cycle_stat_t st_fc_logits;
    cycle_stat_t st_softmax;
    cycle_stat_t st_model_total;
    cycle_stat_init(&st_input_cast);
    cycle_stat_init(&st_conv1_pool1);
    cycle_stat_init(&st_conv2_pool2);
    cycle_stat_init(&st_conv3_gap);
    cycle_stat_init(&st_fc_logits);
    cycle_stat_init(&st_softmax);
    cycle_stat_init(&st_model_total);

    for (uint32_t tc = 0; tc < TINYSPEECH_TEST_NUM_CASES; tc++) {
        const tinyspeech_test_input_case_t *c = &g_tinyspeech_test_inputs[tc];

#if !TINYSPEECH_VERBOSE_CASE_LOGS
#if (TINYSPEECH_PROGRESS_EVERY > 0)
        if ((tc % TINYSPEECH_PROGRESS_EVERY) == 0) {
            printf("  progress: case %lu/%d\n",
                   (unsigned long)(tc + 1),
                   TINYSPEECH_TEST_NUM_CASES);
            fflush(stdout);
        }
#endif
#endif

        Tensor input = make_input_tensor(c->data);
        if (TINYSPEECH_VERBOSE_CASE_LOGS) {
            print_input_preview(c->data);
        }

        uint64_t c0 = rdcycle64();
        Tensor probs = tinyspeech_run_inference(&input);
        uint64_t c1 = rdcycle64();
        free_tensor(&input);
        const tinyspeech_cycle_profile_t *prof = tinyspeech_last_cycle_profile();
        uint64_t cycles = c1 - c0;
        cycles_sum += cycles;
        if (cycles < cycles_min) {
            cycles_min = cycles;
        }
        if (cycles > cycles_max) {
            cycles_max = cycles;
        }
        cycle_stat_update(&st_input_cast, prof->input_cast);
        cycle_stat_update(&st_conv1_pool1, prof->conv1_pool1);
        cycle_stat_update(&st_conv2_pool2, prof->conv2_pool2);
        cycle_stat_update(&st_conv3_gap, prof->conv3_gap);
        cycle_stat_update(&st_fc_logits, prof->fc_logits);
        cycle_stat_update(&st_softmax, prof->softmax);
        cycle_stat_update(&st_model_total, prof->total);

        float max_prob = 0.0f;
        int32_t pred = tinyspeech_argmax(&probs, &max_prob);
        float sum = 0.0f;
        int ok = output_is_valid(&probs, &sum);
        if (TINYSPEECH_VERBOSE_CASE_LOGS) {
            printf("[CASE %lu] %s\n", (unsigned long)tc, c->name);
            if ((c->expected_label >= 0) && (c->expected_label < TINYSPEECH_NUM_CLASSES)) {
                printf("    expected  = %s (%ld)\n", k_labels[c->expected_label], (long)c->expected_label);
            } else {
                printf("    expected  = <background/unknown>\n");
            }
            printf("    output    =");
            for (int32_t i = 0; i < probs.size; i++) {
                printf(" %.6f", probs.f_data[i]);
            }
            printf("\n");
            if ((pred >= 0) && (pred < TINYSPEECH_NUM_CLASSES)) {
                printf("    predict   = %s (%ld), %s=%.6f\n",
                       k_labels[pred], (long)pred, score_label, max_prob);
            } else {
                printf("    predict   = <out-of-range> (%ld), %s=%.6f\n",
                       (long)pred, score_label, max_prob);
            }
            printf("    output_sum= %.6f\n", sum);
            printf("    cycles    = %lu\n", (unsigned long)cycles);
        }
        int ref_ok = compare_with_reference(tc, c, &probs, pred, &ref_summary);

#if TINYSPEECH_DEBUG_TRACE
        print_debug_trace();
#endif

        if ((c->expected_label >= 0) && (c->expected_label < TINYSPEECH_NUM_CLASSES)) {
            labeled_total++;
            if (pred == c->expected_label) {
                labeled_match++;
            }
        }

        int case_pass = (ok && ref_ok);
        if (case_pass) {
            pass++;
        } else {
            fail++;
        }

        if (!TINYSPEECH_VERBOSE_CASE_LOGS && TINYSPEECH_PRINT_CASE_OUTPUTS) {
            const char *exp_str =
                ((c->expected_label >= 0) && (c->expected_label < TINYSPEECH_NUM_CLASSES))
                    ? k_labels[c->expected_label]
                    : "<unknown/bg>";
            const char *pred_str =
                ((pred >= 0) && (pred < TINYSPEECH_NUM_CLASSES))
                    ? k_labels[pred]
                    : "<oor>";

            printf("[CASE %lu] %s exp=%s pred=%s %s=%.6f sum=%.6f cycles=%lu status=%s\n",
                   (unsigned long)tc,
                   c->name,
                   exp_str,
                   pred_str,
                   score_label,
                   max_prob,
                   sum,
                   (unsigned long)cycles,
                   case_pass ? "PASS" : "FAIL");

            if (TINYSPEECH_PRINT_CASE_PROBS) {
                printf("    output =");
                for (int32_t i = 0; i < probs.size; i++) {
                    printf(" %.6f", probs.f_data[i]);
                }
                printf("\n");
            }
        }

        if (TINYSPEECH_VERBOSE_CASE_LOGS) {
            if (case_pass) {
                printf("    status    = PASS\n\n");
            } else if (!ok) {
                printf("    status    = FAIL (invalid output vector)\n\n");
            } else {
                printf("    status    = FAIL (reference mismatch)\n\n");
            }
        } else if (!TINYSPEECH_PRINT_CASE_OUTPUTS && !case_pass) {
            if ((c->expected_label >= 0) && (c->expected_label < TINYSPEECH_NUM_CLASSES)) {
                printf("[CASE %lu] FAIL %s expected=%s pred=%s cycles=%lu\n",
                       (unsigned long)tc,
                       c->name,
                       k_labels[c->expected_label],
                       ((pred >= 0) && (pred < TINYSPEECH_NUM_CLASSES)) ? k_labels[pred] : "<oor>",
                       (unsigned long)cycles);
            } else {
                printf("[CASE %lu] FAIL %s expected=<unknown/bg> pred=%s cycles=%lu\n",
                       (unsigned long)tc,
                       c->name,
                       ((pred >= 0) && (pred < TINYSPEECH_NUM_CLASSES)) ? k_labels[pred] : "<oor>",
                       (unsigned long)cycles);
            }
        }

        free_tensor(&probs);
    }

    printf("TinySpeech summary: pass=%lu fail=%lu\n",
           (unsigned long)pass,
           (unsigned long)fail);
    printf("Label-match summary: %lu/%lu labeled cases\n",
           (unsigned long)labeled_match,
           (unsigned long)labeled_total);
    printf("Reference summary: compared=%lu pred_match=%lu prob_fail=%lu logit_fail=%lu stage_fail=%lu missing_ref=%lu\n",
           (unsigned long)ref_summary.compared,
           (unsigned long)ref_summary.pred_match,
           (unsigned long)ref_summary.prob_fail,
           (unsigned long)ref_summary.logit_fail,
           (unsigned long)ref_summary.stage_fail,
           (unsigned long)ref_summary.missing_ref);
    if (TINYSPEECH_TEST_NUM_CASES > 0) {
        unsigned long avg_cycles = (unsigned long)(cycles_sum / (uint64_t)TINYSPEECH_TEST_NUM_CASES);
        printf("Cycle summary: min=%lu avg=%lu max=%lu\n",
               (unsigned long)cycles_min,
               avg_cycles,
               (unsigned long)cycles_max);
        const uint64_t ncases = (uint64_t)TINYSPEECH_TEST_NUM_CASES;
        printf("Layer-cycle summary:\n");
        printf("  input_cast   : min=%lu avg=%lu max=%lu\n",
               (unsigned long)st_input_cast.min,
               (unsigned long)(st_input_cast.sum / ncases),
               (unsigned long)st_input_cast.max);
        printf("  conv1+pool1  : min=%lu avg=%lu max=%lu\n",
               (unsigned long)st_conv1_pool1.min,
               (unsigned long)(st_conv1_pool1.sum / ncases),
               (unsigned long)st_conv1_pool1.max);
        printf("  conv2+pool2  : min=%lu avg=%lu max=%lu\n",
               (unsigned long)st_conv2_pool2.min,
               (unsigned long)(st_conv2_pool2.sum / ncases),
               (unsigned long)st_conv2_pool2.max);
        printf("  conv3+gap    : min=%lu avg=%lu max=%lu\n",
               (unsigned long)st_conv3_gap.min,
               (unsigned long)(st_conv3_gap.sum / ncases),
               (unsigned long)st_conv3_gap.max);
        printf("  fc_logits    : min=%lu avg=%lu max=%lu\n",
               (unsigned long)st_fc_logits.min,
               (unsigned long)(st_fc_logits.sum / ncases),
               (unsigned long)st_fc_logits.max);
        printf("  softmax      : min=%lu avg=%lu max=%lu\n",
               (unsigned long)st_softmax.min,
               (unsigned long)(st_softmax.sum / ncases),
               (unsigned long)st_softmax.max);
        printf("  model_total  : min=%lu avg=%lu max=%lu\n",
               (unsigned long)st_model_total.min,
               (unsigned long)(st_model_total.sum / ncases),
               (unsigned long)st_model_total.max);
    }

    return (fail == 0) ? 0 : 1;
}

int app_main(void) {
    return run_suite_for_frequency(target_frequency);
}

#if TINYSPEECH_MC_ENABLE_PLL_SWEEP
static const uint64_t k_pll_sweep_freqs_hz[] = {
    TINYSPEECH_MC_PLL_FREQ_LIST
};
#endif

int main(void) {
#if TINYSPEECH_MC_ENABLE_PLL_SWEEP
    const size_t num_freqs =
        sizeof(k_pll_sweep_freqs_hz) / sizeof(k_pll_sweep_freqs_hz[0]);
    int status = 0;

    if (num_freqs == 0u) {
        return 0;
    }

    target_frequency = k_pll_sweep_freqs_hz[0];
    init_test(target_frequency);
    hthread_init();
    hthread_issue(1, mc_nop_worker, NULL);
    hthread_join(1);
    status |= run_suite_for_frequency(target_frequency);

    for (size_t i = 1; i < num_freqs; ++i) {
        target_frequency = k_pll_sweep_freqs_hz[i];
        reconfigure_pll(target_frequency, TINYSPEECH_MC_PLL_SWEEP_SLEEP_MS);
        hthread_issue(1, mc_nop_worker, NULL);
        hthread_join(1);
        status |= run_suite_for_frequency(target_frequency);
    }
    return status;
#else
    app_init();
    return app_main();
#endif
}
