/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  * @authors        :
  ******************************************************************************
  * @attention
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  *             
  *          _______                        ______   ______           
  *         /       \                      /      \ /      |          
  *         $$$$$$$  |  ______    ______  /$$$$$$  |$$$$$$/   ______  
  *         $$ |__$$ | /      \  /      \ $$ |__$$ |  $$ |   /      \ 
  *         $$    $$< /$$$$$$  |/$$$$$$  |$$    $$ |  $$ |  /$$$$$$  |
  *         $$$$$$$  |$$ |  $$ |$$ |  $$/ $$$$$$$$ |  $$ |  $$ |  $$ |
  *         $$ |__$$ |$$ \__$$ |$$ |      $$ |  $$ | _$$ |_ $$ \__$$ |
  *         $$    $$/ $$    $$/ $$ |      $$ |  $$ |/ $$   |$$    $$ |
  *         $$$$$$$/   $$$$$$/  $$/       $$/   $$/ $$$$$$/  $$$$$$$ |
  *                                                               $$ |
  *                                                               $$ |
  *                                                               $$/ 
  *
  *             Powered by Llama2 based upon llama2.c by @karpathy
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

#if defined(TRANSPOSED_WEIGHTS) || defined(VEC_SOFTMAX)
#include "layers.h"
#endif
#ifdef BORAIQ_TINY_SHAPE_GEMM
#include "tiny_gemm_i8_rvv.h"
#endif
#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif
#ifdef PREFILL_MULTICORE
#include <thread-lib/hthread.h>
#endif

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
// #define ENABLE_BORAVOICE_INTEG
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
const unsigned char *ASCII_CRLF = (const unsigned char *) "\r\n";
const unsigned char *ASCII_BEL = (const unsigned char *) "\a";

// int32_t GS = 64; // group size global for quantization of the weights

uint64_t target_frequency = 500000000l;

// #ifdef ENABLE_DMA_MATVEC
// int32_t GS_MATVEC_BOUND = 0;

// #endif

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

typedef struct Tokenizer Tokenizer;
typedef struct Sampler Sampler;

char* decode(Tokenizer* t, int prev_token, int token);
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
int sample(Sampler* sampler, float* logits);
void safe_printf(char *piece);


/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN PUC */

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    int8_t* q;    // quantized values
    float s; // scaling factors
} QuantizedTensor;

typedef struct {
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    float* token_embedding_table; // same, but dequantized

    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    float *rope_freq; // RoPE inverse frequencies per (head_dim/2,)
    float *rope_cos;  // RoPE cos cache for current position
    float *rope_sin;  // RoPE sin cache for current position
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
#ifdef TRANSPOSED_WEIGHTS
    TransformerWeightsT weights_t; // transposed B_pack copies for vectorized inference
#endif
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = p->dim / p->n_heads;
    int rope_terms = head_size / 2;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = calloc(p->dim, sizeof(int8_t)), .s = 0.0f };
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = 0.0f };
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->rope_freq = rope_terms > 0 ? calloc((size_t)rope_terms, sizeof(float)) : NULL;
    s->rope_cos = rope_terms > 0 ? calloc((size_t)rope_terms, sizeof(float)) : NULL;
    s->rope_sin = rope_terms > 0 ? calloc((size_t)rope_terms, sizeof(float)) : NULL;
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    if (rope_terms > 0 && s->rope_freq != NULL) {
        for (int i = 0; i < rope_terms; i++) {
            int head_dim = i * 2;
            s->rope_freq[i] = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        }
    }
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache || (rope_terms > 0 && (!s->rope_freq || !s->rope_cos || !s->rope_sin))) {
        printf("STDERR: malloc failed!\r\n");
        printf("size: %d\r\n", p->n_layers * p->seq_len * kv_dim * sizeof(float));
        printf("s->q: %x\r\n", s->q);
        printf("key_cache: %x\r\n", s->key_cache);

        // exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->hq.q);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->rope_freq);
    free(s->rope_cos);
    free(s->rope_sin);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s;
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    float Q_MAX = 127.0f;

    // find the max absolute value
    float wmax = 0.0;
    for (int i = 0; i < n; i++) {
        float val = fabs(x[i]);
        if (val > wmax) {
            wmax = val;
        }
    }

    if (wmax == 0.0f) {
        qx->s = 1.0f;
        memset(qx->q, 0, (size_t)n * sizeof(int8_t));
        return;
    }

    // calculate and write the scaling factor
    float scale = wmax / Q_MAX;
    float inv_scale = 1.0f / scale;
    qx->s = scale;

    // calculate and write the quantized values
    for (int i = 0; i < n; i++) {
        int q = (int)roundf(x[i] * inv_scale);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qx->q[i] = (int8_t)q;
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    for(int i=0; i<n; i++) {
        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = *(float*)p;
        p = (float*)p + 1;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float* fptr = (float*) ptr; // cast our pointer to float*
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    // now read all the quantized weights
    ptr = (void*)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // dequantize token embedding table
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}


// void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
//                      int* fd, float** data, ssize_t* file_size) {
//     FILE *file = fopen(checkpoint, "rb");
//     if (!file) { printf("STDERR: Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
//     // read in the config header
//     if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
//     // negative vocab size is hacky way of signaling unshared weights. bit yikes.
//     int shared_weights = config->vocab_size > 0 ? 1 : 0;
//     config->vocab_size = abs(config->vocab_size);
//     // figure out the file size
//     fseek(file, 0, SEEK_END); // move file pointer to end of file
//     *file_size = ftell(file); // get the file size, in bytes
//     fclose(file);
//     // memory map the Transformer weights into the data pointer
//     *fd = open(checkpoint, O_RDONLY); // open in read only mode
//     if (*fd == -1) { printf("STDERR: open failed!\n"); exit(EXIT_FAILURE); }
//     *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
//     if (*data == MAP_FAILED) { printf("STDERR: mmap failed!\n"); exit(EXIT_FAILURE); }
//     float* weights_ptr = *data + sizeof(Config)/sizeof(float);
//     memory_map_weights(weights, config, weights_ptr, shared_weights);
// }

void read_checkpoint_from_header(Config* config, TransformerWeights* weights, float** data, size_t* file_size) {
  size_t cumulative_offset = 0;
  printf("Loading checkpoint data...\r\n");
  // Check magic number
  uint32_t magic = *((uint32_t*)(WEIGHTS + cumulative_offset));
  if (magic != MODEL_MAGIC_NUMBER) {
    printf("Model magic number does not match! Please preprocess with export.py.\r\n");
  }
  printf("Magic number verified\r\n");
  cumulative_offset += sizeof(uint32_t);

  uint32_t version = *((uint32_t*)(WEIGHTS + cumulative_offset));
  if (version != MODEL_VERSION_INT8) {
    printf("Model version is not an Int8 Quantized model. Version (hex) = %x", version);
  }
  printf("Model is properly formatted as Int8 Quantized (Version 2).\r\n");
  cumulative_offset += sizeof(uint32_t);
  
  // load from weights.h WEIGHTS
  memcpy(config, (WEIGHTS + cumulative_offset), sizeof(Config));
  cumulative_offset += sizeof(Config);
  printf("Successfully loaded configuration structure.\r\n");
  printf("\tTransformer Dimension:\t%d\r\n", config->dim);
  printf("\tFFN Layer Dimension:\t%d\r\n", config->hidden_dim);
  printf("\tLayer Count:\t%d\r\n", config->n_layers);
  printf("\tQuery Head Count:\t%d\r\n", config->n_heads);
  printf("\tKey/Value Head Count:\t%d\r\n", config->n_kv_heads);
  printf("\tByte-Level Vocabulary Size:\t%d\r\n", config->vocab_size);
  printf("\tMaximum Sequence Length:\t%d\r\n", config->seq_len);

  // Check shared classifier byte
  uint8_t shared_classifier = *((uint8_t*)(WEIGHTS + cumulative_offset));
  if (shared_classifier != 1) {
    printf("Non-shared classifier detected. Shared classifier byte value = %x", shared_classifier);
  }
  printf("Proper shared classifier detected.\r\n");
  cumulative_offset += sizeof(uint8_t);

  // Read group size
  //int32_t group_size = *((int32_t*)(WEIGHTS + cumulative_offset));
  //GS = group_size;
  //cumulative_offset += sizeof(int32_t);
  //printf("\tGroup Size:\t%d\r\n", GS);

  printf("Accelerator Status:\r\n");
  printf("\tBearly24 DMA MatVec:\t");
#ifdef ENABLE_DMA_MATVEC
  printf("Enabled\r\n");
#else
  printf("Disabled\r\n");
#endif

  printf("\tBearly24 QTrans DotProd:\t");
#ifdef ENABLE_QT_DOTPROD
  GS_QTDP_BOUND = GS / 8 * 8;
  printf("Enabled\r\n");
#else
  printf("Disabled\r\n");
#endif

  printf("\tDSP'24 Saturn-V Vector:\t");
#ifdef ENABLE_SATURNV_VEC
  printf("Enabled\r\n");
#else
  printf("Disabled\r\n");
#endif

  // int shared_weights = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs(config->vocab_size);
  *file_size = sizeof(Config) + sizeof(WEIGHTS);

  // Point to WEIGHTS array as data
  *data = (float*)WEIGHTS;
  void* weights_ptr = ((char*)WEIGHTS) + MODEL_V2_HEADER_SIZE;
  memory_map_weights(weights, config, weights_ptr, shared_classifier);
}

#ifdef TRANSPOSED_WEIGHTS
void alloc_and_transpose_weights_i8(
    TransformerWeightsT* wt,
    const TransformerWeights* w,
    const Config* p);
void free_transposed_weights_i8(TransformerWeightsT* wt);
#endif

void build_transformer(Transformer *t) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint_from_header(&t->config, &t->weights, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
#ifdef TRANSPOSED_WEIGHTS
    // transpose all weight matrices once; forward() will use weights_t exclusively
    alloc_and_transpose_weights_i8(&t->weights_t, &t->weights, &t->config);
#endif
}

void free_transformer(Transformer* t) {
#ifdef TRANSPOSED_WEIGHTS
    free_transposed_weights_i8(&t->weights_t);
#endif
  // Free quantized tensors
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }
    // close the memory mapping
    // if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    // if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

#ifndef ATTN_VEC_SOFTMAX_MIN
#define ATTN_VEC_SOFTMAX_MIN 48
#endif

static inline void softmax_attn(float *att, int len) {
#ifdef VEC_SOFTMAX
    if (len >= ATTN_VEC_SOFTMAX_MIN) {
        softmax_vec(att, att, 1, (size_t)len);
    } else {
        softmax(att, len);
    }
#else
    softmax(att, len);
#endif
}

static inline float dot_qk_head(const float *q, const float *k, int n) {
#if defined(__riscv_vector)
    int i = 0;
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(n - i));
        vfloat32m4_t vq = __riscv_vle32_v_f32m4(q + i, vl);
        vfloat32m4_t vk = __riscv_vle32_v_f32m4(k + i, vl);
        vfloat32m4_t vm = __riscv_vfmul_vv_f32m4(vq, vk, vl);
        acc = __riscv_vfredusum_vs_f32m4_f32m1(vm, acc, vl);
        i += (int)vl;
    }
    return __riscv_vfmv_f_s_f32m1_f32(acc);
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += q[i] * k[i];
    return sum;
#endif
}

static inline void axpy_v_head(float *dst, const float *v, float a, int n) {
#if defined(__riscv_vector)
    int i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(n - i));
        vfloat32m4_t vd = __riscv_vle32_v_f32m4(dst + i, vl);
        vfloat32m4_t vv = __riscv_vle32_v_f32m4(v + i, vl);
        vd = __riscv_vfmacc_vf_f32m4(vd, a, vv, vl);
        __riscv_vse32_v_f32m4(dst + i, vd, vl);
        i += (int)vl;
    }
#else
    for (int i = 0; i < n; i++) dst[i] += a * v[i];
#endif
}

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    int i;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul
        int j;
        for (j = 0; j <= n; j++) {
            ival += ((int32_t) x->q[j]) * ((int32_t) w->q[in + j]);
        }

        xout[i] = ((float) ival) * w->s * x->s;
    }
}

#ifdef TRANSPOSED_WEIGHTS
/* ---------------------------------------------------------------------------
 * make_b_pack_i8 — build a B_pack from a quantized weight matrix.
 *
 * W is stored as W[n_out × n_in] row-major int8 (from QuantizedTensor.q).
 * Output B_pack is [(n_in+1) × n_out] bytes:
 *   Row 0       : n_out zero bytes  (zero bias)
 *   Rows 1..K   : rows of W_T as int8  (W_T[k][j] = W[j*n_in+k])
 *
 * The weights are already signed int8 from QuantizedTensor, so no bias-128
 * subtraction is needed (unlike uint8-biased formats).
 * ------------------------------------------------------------------------- */
static void make_b_pack_i8(
    unsigned char* pack,
    const int8_t* W,
    int n_out, int n_in)
{
    int8_t* ipack = (int8_t*)pack;
    /* bias row: zeros */
    for (int j = 0; j < n_out; j++)
        ipack[j] = 0;
    /* weight rows: W_T */
    for (int k = 0; k < n_in; k++)
        for (int j = 0; j < n_out; j++)
            ipack[(k + 1) * n_out + j] = W[j * n_in + k];
}

/* ---------------------------------------------------------------------------
 * alloc_and_transpose_weights_i8 — allocate and fill transposed B_packs.
 * Called once in build_transformer().
 * ------------------------------------------------------------------------- */
void alloc_and_transpose_weights_i8(
    TransformerWeightsT* wt,
    const TransformerWeights* w,
    const Config* p)
{
    int dim        = p->dim;
    int hidden_dim = p->hidden_dim;
    int n_layers   = p->n_layers;
    int kv_dim     = (p->dim * p->n_kv_heads) / p->n_heads;

    /* wq: W[dim × dim], B_pack [(dim+1) × dim] per layer */
    wt->wq_T = (unsigned char*)malloc(n_layers * (size_t)(dim + 1) * dim);
    for (int l = 0; l < n_layers; l++)
        make_b_pack_i8(wt->wq_T + l * (size_t)(dim + 1) * dim,
                       w->wq[l].q, dim, dim);

    /* wk: W[kv_dim × dim], B_pack [(dim+1) × kv_dim] per layer */
    wt->wk_T = (unsigned char*)malloc(n_layers * (size_t)(dim + 1) * kv_dim);
    for (int l = 0; l < n_layers; l++)
        make_b_pack_i8(wt->wk_T + l * (size_t)(dim + 1) * kv_dim,
                       w->wk[l].q, kv_dim, dim);

    /* wv: same layout as wk */
    wt->wv_T = (unsigned char*)malloc(n_layers * (size_t)(dim + 1) * kv_dim);
    for (int l = 0; l < n_layers; l++)
        make_b_pack_i8(wt->wv_T + l * (size_t)(dim + 1) * kv_dim,
                       w->wv[l].q, kv_dim, dim);

    /* wo: W[dim × dim], same layout as wq */
    wt->wo_T = (unsigned char*)malloc(n_layers * (size_t)(dim + 1) * dim);
    for (int l = 0; l < n_layers; l++)
        make_b_pack_i8(wt->wo_T + l * (size_t)(dim + 1) * dim,
                       w->wo[l].q, dim, dim);

    /* w1: W[hidden_dim × dim], B_pack [(dim+1) × hidden_dim] per layer */
    wt->w1_T = (unsigned char*)malloc(n_layers * (size_t)(dim + 1) * hidden_dim);
    for (int l = 0; l < n_layers; l++)
        make_b_pack_i8(wt->w1_T + l * (size_t)(dim + 1) * hidden_dim,
                       w->w1[l].q, hidden_dim, dim);

    /* w2: W[dim × hidden_dim], B_pack [(hidden_dim+1) × dim] per layer */
    wt->w2_T = (unsigned char*)malloc(n_layers * (size_t)(hidden_dim + 1) * dim);
    for (int l = 0; l < n_layers; l++)
        make_b_pack_i8(wt->w2_T + l * (size_t)(hidden_dim + 1) * dim,
                       w->w2[l].q, dim, hidden_dim);

    /* w3: same layout as w1 */
    wt->w3_T = (unsigned char*)malloc(n_layers * (size_t)(dim + 1) * hidden_dim);
    for (int l = 0; l < n_layers; l++)
        make_b_pack_i8(wt->w3_T + l * (size_t)(dim + 1) * hidden_dim,
                       w->w3[l].q, hidden_dim, dim);

    /* wcls split for multicore-friendly logits: both halves keep B_pack layout */
    wt->wcls0_n = p->vocab_size / 2;
    wt->wcls1_n = p->vocab_size - wt->wcls0_n;
    wt->wcls0_T = wt->wcls0_n > 0 ? (unsigned char*)malloc((size_t)(dim + 1) * wt->wcls0_n) : NULL;
    wt->wcls1_T = wt->wcls1_n > 0 ? (unsigned char*)malloc((size_t)(dim + 1) * wt->wcls1_n) : NULL;
    if (wt->wcls0_n > 0) {
        make_b_pack_i8(wt->wcls0_T, w->wcls[0].q, wt->wcls0_n, dim);
    }
    if (wt->wcls1_n > 0) {
        make_b_pack_i8(
            wt->wcls1_T,
            w->wcls[0].q + (size_t)wt->wcls0_n * dim,
            wt->wcls1_n,
            dim);
    }
}

void free_transposed_weights_i8(TransformerWeightsT* wt) {
    free(wt->wq_T);
    free(wt->wk_T);
    free(wt->wv_T);
    free(wt->wo_T);
    free(wt->w1_T);
    free(wt->w2_T);
    free(wt->w3_T);
    free(wt->wcls0_T);
    free(wt->wcls1_T);
}

/* ---------------------------------------------------------------------------
 * matmul_t — vectorized matmul using pre-transposed B_pack.
 *
 * Computes xout(n_out,) = W(n_out,n_in) @ xq(n_in,) exactly like matmul(),
 * but w_t_pack is W stored as B_pack (transposed + bias row).
 *
 * Reformulated as: xout(1,n_out) = xq(1,n_in) @ W_T(n_in,n_out)
 *   → int8_qgemm_fout(M=1, N=n_out, K=n_in) — vectorises over n_out.
 * ------------------------------------------------------------------------- */
static void matmul_t(
    float* xout,
    QuantizedTensor* xq,
    const void* w_t_pack,
    float w_scale,
    int n_in, int n_out)
{
    float scale = xq->s * w_scale;
#ifdef BORAIQ_TINY_SHAPE_GEMM
    if (borai_tiny_matmul_t_i8_fout(
            xq->q,
            (const int8_t*)w_t_pack,
            xout,
            n_in,
            n_out,
            scale)) {
        return;
    }
#endif
    quant_fully_connected_int8_t(
        (size_t)n_in, (size_t)n_out, 1,
        xq->q, w_t_pack, xout, scale);
}
#endif /* TRANSPOSED_WEIGHTS */

// ----------------------------------------------------------------------------
// Multicore prefill support
#ifdef PREFILL_MULTICORE

static Transformer  _mc_transformer;   /* shared transformer, built by hart 0 */

static volatile int _mc_token_val    = 0;
static volatile int _mc_pos_val      = 0;
static volatile int _mc_fwd_req      = 0;
static volatile int _mc_fwd_done     = 0;
static volatile int _mc_worker_ready = 0;
static volatile int _mc_swiglu_h1_done = 0;

static float* forward_mc(Transformer* transformer, int token, int pos, int hartid);

/* 2-hart sense-reversing barrier.
 * Hart 0: waits for hart 1 to arrive, then flips the shared sense to release.
 * Hart 1: signals arrival, then waits for sense to flip. */
static volatile int _bar_h1_arrived = 0;
static volatile int _bar_sense      = 0;

static void barrier2(int hartid) {
    int s = _bar_sense;
    if (hartid == 0) {
        while (!_bar_h1_arrived) {}
        _bar_h1_arrived = 0;
        __sync_synchronize();
        _bar_sense = !s;
        __sync_synchronize();
    } else {
        _bar_h1_arrived = 1;
        __sync_synchronize();
        while (_bar_sense == s) {}
        __sync_synchronize();
    }
}

/* Scalar matmul over output rows [row_start, row_end).
 * Computes xout[i] = W[i,:] dot x  for i in [row_start, row_end). */
static void matmul_rows(float* xout, QuantizedTensor* x, QuantizedTensor* w,
                        int n, int d, int row_start, int row_end) {
    (void)d;
    for (int i = row_start; i < row_end; i++) {
        int32_t ival = 0;
        int in = i * n;
        for (int j = 0; j < n; j++) {
            ival += (int32_t)x->q[j] * (int32_t)w->q[in + j];
        }
        xout[i] = (float)ival * w->s * x->s;
    }
}

static void mc_forward_worker(void *arg) {
    Transformer *transformer = (Transformer *)arg;
    _mc_worker_ready = 1;
    __sync_synchronize();
    while (1) {
        while (_mc_fwd_req == 0) {}
        __sync_synchronize();
        _mc_fwd_req = 0;
        int token = _mc_token_val;
        int pos   = _mc_pos_val;
        forward_mc(transformer, token, pos, 1);
        __sync_synchronize();
        _mc_fwd_done = 1;
    }
}

static void mc_start_worker(Transformer *transformer) {
    _mc_fwd_req = 0;
    _mc_fwd_done = 0;
    _mc_worker_ready = 0;
    _mc_swiglu_h1_done = 0;
    __sync_synchronize();
    hthread_issue(1, mc_forward_worker, transformer);
    while (_mc_worker_ready == 0) {}
    __sync_synchronize();
}

/* forward_mc — parallel forward pass for two harts.
 *
 * Mutual-exclusion assignment: each independent matmul is owned exclusively
 * by one hart, so both harts run full-size (optionally vectorized) kernels
 * simultaneously rather than each computing half-rows of a single matrix.
 *
 * Assignment per layer:
 *   QKV  : hart 0 → wq + wk  |  hart 1 → wv
 *   Attn : both — each handles n_heads/2 heads
 *   FFN  : hart 0 → w1       |  hart 1 → w3  (key parallelism win)
 *   wcls : both harts split vocab work (TRANSPOSED_WEIGHTS split packs,
 *          or scalar row split)
 *
 * Mostly-sequential work (rmsnorm, RoPE, kv-store, residual, w2) stays on
 * hart 0; hart 1 additionally handles half of SwiGLU between FFN matmuls.
 *
 * When TRANSPOSED_WEIGHTS is also defined, w1/w3/wq/wk/wv/wo all use the
 * vectorized matmul_t kernel on their full matrices.
 *
 * Returns logits (hart 0) or NULL (hart 1). */
static float* forward_mc(Transformer* transformer, int token, int pos, int hartid) {
    Config* p             = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s           = &transformer->state;
#ifdef TRANSPOSED_WEIGHTS
    TransformerWeightsT* wt = &transformer->weights_t;
#endif
    float* x        = s->x;
    int dim         = p->dim;
    int kv_dim      = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul      = p->n_heads / p->n_kv_heads;
    int hidden_dim  = p->hidden_dim;
    int head_size   = dim / p->n_heads;
    int rope_terms  = head_size / 2;
    float inv_sqrt_head_size = 1.0f / sqrtf((float)head_size);

    /* embedding copy (hart 0 only; hart 1 waits at first barrier in the loop) */
    if (hartid == 0) {
        memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));
        for (int i = 0; i < rope_terms; i++) {
            float val = pos * s->rope_freq[i];
            s->rope_cos[i] = cosf(val);
            s->rope_sin[i] = sinf(val);
        }
    }

    for (int l = 0; l < p->n_layers; l++) {

        /* rmsnorm + quantize input (hart 0) */
        if (hartid == 0) {
            rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);
            quantize(&s->xq, s->xb, dim);
        }
        barrier2(hartid);

        /* QKV projections — mutual exclusion:
         *   hart 0 → wq (s->q) + wk (s->k)
         *   hart 1 → wv (s->v)                              */
        if (hartid == 0) {
#ifdef TRANSPOSED_WEIGHTS
            matmul_t(s->q, &s->xq, wt->wq_T + l*(size_t)(dim+1)*dim,    w->wq[l].s, dim, dim);
            matmul_t(s->k, &s->xq, wt->wk_T + l*(size_t)(dim+1)*kv_dim, w->wk[l].s, dim, kv_dim);
#else
            matmul(s->q, &s->xq, w->wq + l, dim, dim);
            matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
#endif
        } else {
#ifdef TRANSPOSED_WEIGHTS
            matmul_t(s->v, &s->xq, wt->wv_T + l*(size_t)(dim+1)*kv_dim, w->wv[l].s, dim, kv_dim);
#else
            matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);
#endif
        }
        barrier2(hartid);

        /* RoPE relative positional encoding + KV cache store (hart 0) */
        if (hartid == 0) {
            for (int i = 0; i < dim; i += 2) {
                int rope_idx = (i % head_size) >> 1;
                float fcr = s->rope_cos[rope_idx];
                float fci = s->rope_sin[rope_idx];
                int rotn = i < kv_dim ? 2 : 1;
                for (int v = 0; v < rotn; v++) {
                    float* vec = v == 0 ? s->q : s->k;
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i]     = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }
            int loff = l * p->seq_len * kv_dim;
            memcpy(s->key_cache   + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
            memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));
        }
        barrier2(hartid);

        /* multihead attention — each hart owns n_heads/2 heads exclusively */
        {
            int h_start = hartid * (p->n_heads / 2);
            int h_end   = h_start + p->n_heads / 2;
            int loff    = l * p->seq_len * kv_dim;
            for (int h = h_start; h < h_end; h++) {
                float* q   = s->q   + h * head_size;
                float* att = s->att + h * p->seq_len;
                int kv_head_off = (h / kv_mul) * head_size;
                for (int t = 0; t <= pos; t++) {
                    float* k = s->key_cache + loff + t * kv_dim + kv_head_off;
                    float score = dot_qk_head(q, k, head_size);
                    att[t] = score * inv_sqrt_head_size;
                }
                softmax_attn(att, pos + 1);
                float* xb = s->xb + h * head_size;
                memset(xb, 0, head_size * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    float* v = s->value_cache + loff + t * kv_dim + kv_head_off;
                    float a  = att[t];
                    axpy_v_head(xb, v, a, head_size);
                }
            }
        }
        barrier2(hartid);

        /* wo matmul + residual + FFN rmsnorm + quantize (hart 0) */
        if (hartid == 0) {
            quantize(&s->xq, s->xb, dim);
#ifdef TRANSPOSED_WEIGHTS
            matmul_t(s->xb2, &s->xq, wt->wo_T + l*(size_t)(dim+1)*dim, w->wo[l].s, dim, dim);
#else
            matmul(s->xb2, &s->xq, w->wo + l, dim, dim);
#endif
            for (int i = 0; i < dim; i++) x[i] += s->xb2[i];
            rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);
            quantize(&s->xq, s->xb, dim);
        }
        barrier2(hartid);

        /* FFN w1 + w3 — mutual exclusion (biggest parallelism win):
         *   hart 0 → w1 (s->hb)   hart 1 → w3 (s->hb2)
         * Both run simultaneously on their full output buffer. */
        if (hartid == 0) {
#ifdef TRANSPOSED_WEIGHTS
            matmul_t(s->hb,  &s->xq, wt->w1_T + l*(size_t)(dim+1)*hidden_dim, w->w1[l].s, dim, hidden_dim);
#else
            matmul(s->hb,  &s->xq, w->w1 + l, dim, hidden_dim);
#endif
        } else {
#ifdef TRANSPOSED_WEIGHTS
            matmul_t(s->hb2, &s->xq, wt->w3_T + l*(size_t)(dim+1)*hidden_dim, w->w3[l].s, dim, hidden_dim);
#else
            matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
#endif
        }
        barrier2(hartid);

        /* SwiGLU split: both harts process disjoint hidden ranges, then hart 0
         * runs quantize + w2 + residual to avoid extra barriers on matmul. */
        if (hartid == 0) {
            int h_half = hidden_dim / 2;
            for (int i = 0; i < h_half; i++) {
                float val = s->hb[i];
                val *= 1.0f / (1.0f + expf(-val));
                val *= s->hb2[i];
                s->hb[i] = val;
            }
            while (_mc_swiglu_h1_done == 0) {}
            __sync_synchronize();
            _mc_swiglu_h1_done = 0;
            quantize(&s->hq, s->hb, hidden_dim);
#ifdef TRANSPOSED_WEIGHTS
            matmul_t(s->xb, &s->hq, wt->w2_T + l*(size_t)(hidden_dim+1)*dim, w->w2[l].s, hidden_dim, dim);
#else
            matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);
#endif
            for (int i = 0; i < dim; i++) x[i] += s->xb[i];
        } else {
            int h_half = hidden_dim / 2;
            for (int i = h_half; i < hidden_dim; i++) {
                float val = s->hb[i];
                val *= 1.0f / (1.0f + expf(-val));
                val *= s->hb2[i];
                s->hb[i] = val;
            }
            __sync_synchronize();
            _mc_swiglu_h1_done = 1;
        }
        barrier2(hartid);
    }

    /* final rmsnorm + quantize (hart 0) */
    if (hartid == 0) {
        rmsnorm(x, x, w->rms_final_weight, dim);
        quantize(&s->xq, x, dim);
    }
    barrier2(hartid);

    /* wcls:
     *   TRANSPOSED_WEIGHTS → both harts compute disjoint vocab slices with
     *                        split B_pack buffers (wcls0_T/wcls1_T).
     *   scalar             → both harts split vocab_size rows (mutual exclusion
     *                        via non-overlapping [0, vs_half) / [vs_half, end)). */
#ifdef TRANSPOSED_WEIGHTS
    if (hartid == 0) {
        if (wt->wcls0_n > 0) {
            matmul_t(s->logits, &s->xq, wt->wcls0_T, w->wcls[0].s, dim, wt->wcls0_n);
        }
    } else {
        if (wt->wcls1_n > 0) {
            matmul_t(
                s->logits + wt->wcls0_n,
                &s->xq,
                wt->wcls1_T,
                w->wcls[0].s,
                dim,
                wt->wcls1_n);
        }
    }
#else
    {
        int vs_half  = p->vocab_size / 2;
        int vs_start = hartid * vs_half;
        int vs_end   = vs_start + vs_half;
        matmul_rows(s->logits, &s->xq, w->wcls, dim, p->vocab_size, vs_start, vs_end);
    }
#endif
    barrier2(hartid);

    return hartid == 0 ? s->logits : NULL;
}

/* generate_mc — same as generate() but uses a persistent hart 1 worker issued
 * once through threadlib; each token is handed off via shared flags. */
void generate_mc(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                 char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        printf("STDERR: something is wrong, expected at least 1 prompt token\r\n");
    }

    unsigned long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos   = 0;
    _bar_h1_arrived = 0;
    _bar_sense = 0;
    _mc_swiglu_h1_done = 0;
    while (pos < steps) {
        _mc_token_val = token;
        _mc_pos_val = pos;
        _mc_fwd_done = 0;
        __sync_synchronize();
        _mc_fwd_req = 1;
        __sync_synchronize();
        float* logits = forward_mc(transformer, token, pos, 0);
        while (_mc_fwd_done == 0) {}
        __sync_synchronize();

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        if (next < 0 || next >= transformer->config.vocab_size) {
            printf("STDERR: sampled token out of range (%d), vocab=%d at pos=%d\r\n",
                   next, transformer->config.vocab_size, pos);
            next = 1;
        }
        pos++;

        if (next == 1) { break; }

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) { start = READ_CSR("mcycle"); }
    }
    printf("\r\n");

    if (pos > 1) {
        unsigned long end = READ_CSR("mcycle");
        printf("\r\nBENCHMARK: Total cycles: %lu\r\n", end - start);
        printf("BENCHMARK: Total tokens:\t%d\r\n", pos - 1);
        printf("BENCHMARK: Cycles per token:\t%lu\r\n", (unsigned long)(end - start) / (pos - 1));
        printf("BENCHMARK: Seconds per token:\t%lu\r\n", (unsigned long)((end - start) / target_frequency) / (pos - 1));
        printf("BENCHMARK: Seconds per token (float):\t%f\r\n", ((float)(end - start) / (float)target_frequency) / (float)(pos - 1));
        printf("BENCHMARK: CLOCK Frequency:\t%llu\r\n", (unsigned long long)target_frequency);
        printf("STDERR: achieved tok/s: %f\r\n", (pos - 1) / (((double)(end - start)) / target_frequency));
    }

    free(prompt_tokens);
}

#endif /* PREFILL_MULTICORE */

float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
#ifdef TRANSPOSED_WEIGHTS
    TransformerWeightsT* wt = &transformer->weights_t;
#endif
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    int rope_terms = head_size / 2;
    float inv_sqrt_head_size = 1.0f / sqrtf((float)head_size);

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));
    for (int i = 0; i < rope_terms; i++) {
        float val = pos * s->rope_freq[i];
        s->rope_cos[i] = cosf(val);
        s->rope_sin[i] = sinf(val);
    }

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
#ifdef TRANSPOSED_WEIGHTS
        matmul_t(s->q, &s->xq, wt->wq_T + l*(size_t)(dim+1)*dim,    w->wq[l].s, dim, dim);
        matmul_t(s->k, &s->xq, wt->wk_T + l*(size_t)(dim+1)*kv_dim, w->wk[l].s, dim, kv_dim);
        matmul_t(s->v, &s->xq, wt->wv_T + l*(size_t)(dim+1)*kv_dim, w->wv[l].s, dim, kv_dim);
#else
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);
#endif

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int rope_idx = (i % head_size) >> 1;
            float fcr = s->rope_cos[rope_idx];
            float fci = s->rope_sin[rope_idx];
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        // multihead attention. iterate over all heads
        int h;
#ifdef _OPENMP
        #pragma omp parallel for private(h)
#endif
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            int kv_head_off = (h / kv_mul) * head_size;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + kv_head_off;
                // calculate the attention score as the dot product of q and k
                float score = dot_qk_head(q, k, head_size);
                score *= inv_sqrt_head_size;
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax_attn(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + kv_head_off;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                axpy_v_head(xb, v, a, head_size);
            }
        }

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
#ifdef TRANSPOSED_WEIGHTS
        matmul_t(s->xb2, &s->xq, wt->wo_T + l*(size_t)(dim+1)*dim, w->wo[l].s, dim, dim);
#else
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);
#endif

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
#ifdef TRANSPOSED_WEIGHTS
        matmul_t(s->hb,  &s->xq, wt->w1_T + l*(size_t)(dim+1)*hidden_dim,    w->w1[l].s, dim, hidden_dim);
        matmul_t(s->hb2, &s->xq, wt->w3_T + l*(size_t)(dim+1)*hidden_dim,    w->w3[l].s, dim, hidden_dim);
#else
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
#endif

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
#ifdef TRANSPOSED_WEIGHTS
        matmul_t(s->xb, &s->hq, wt->w2_T + l*(size_t)(hidden_dim+1)*dim, w->w2[l].s, hidden_dim, dim);
#else
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);
#endif

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
#ifdef TRANSPOSED_WEIGHTS
    if (wt->wcls0_n > 0) {
        matmul_t(s->logits, &s->xq, wt->wcls0_T, w->wcls[0].s, dim, wt->wcls0_n);
    }
    if (wt->wcls1_n > 0) {
        matmul_t(
            s->logits + wt->wcls0_n,
            &s->xq,
            wt->wcls1_T,
            w->wcls[0].s,
            dim,
            wt->wcls1_n);
    }
#else
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
#endif
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct Tokenizer {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer_from_header(Tokenizer* t, int vocab_size) {
  // Potential point of improvement: Write the vocab size into the tokenizer file.
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (char**)malloc(vocab_size * sizeof(char*));
  t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; // initialized lazily
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }

  void *tok_ptr = TOKENIZER;
  // read in from TOKENIZER in tokenizer.h
  t->max_token_length = *(int *)tok_ptr;
  tok_ptr += sizeof(int);

  int len;
  for (int i = 0; i < vocab_size; i++) {
    // read into (t->vocab_scores + i)
    memcpy(t->vocab_scores + i, tok_ptr, sizeof(float));
    tok_ptr += sizeof(float);

    // read into len
    memcpy(&len, tok_ptr, sizeof(int));
    tok_ptr += sizeof(int);
    
    t->vocab[i] = (char *)malloc(len + 1);
    memcpy(t->vocab[i], tok_ptr, len);
    tok_ptr += len;
    t->vocab[i][len] = '\0'; // add the string null terminator
  }
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size) {
        return "";
    }
    if (prev_token < 0 || prev_token >= t->vocab_size) {
        prev_token = 0;
    }
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte

    // sscanf replaced with not sscanf
    unsigned char byte_val;
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
        char hex[3];
        hex[0] = piece[3];
        hex[1] = piece[4];
        hex[2] = '\0';
        byte_val = (unsigned char) strtol(hex, NULL, 16);
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { printf("STDERR: cannot encode NULL text\r\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct ProbIndex {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct Sampler {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

static inline void heap_swap_probindex(ProbIndex* a, ProbIndex* b) {
    ProbIndex t = *a;
    *a = *b;
    *b = t;
}

/* max-heapify at index i for ProbIndex.prob */
static inline void heapify_down_probindex(ProbIndex* heap, int size, int i) {
    while (1) {
        int left = (i << 1) + 1;
        int right = left + 1;
        int largest = i;
        if (left < size && heap[left].prob > heap[largest].prob) largest = left;
        if (right < size && heap[right].prob > heap[largest].prob) largest = right;
        if (largest == i) break;
        heap_swap_probindex(&heap[i], &heap[largest]);
        i = largest;
    }
}

/* pop max element from heap[0..*size-1], decrements *size */
static inline ProbIndex heap_pop_max_probindex(ProbIndex* heap, int* size) {
    ProbIndex out = heap[0];
    (*size)--;
    if (*size > 0) {
        heap[0] = heap[*size];
        heapify_down_probindex(heap, *size, 0);
    }
    return out;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    if (n <= 1) return 0;

    int n0 = 0;
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before heap building
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        float p = probabilities[i];
        if (p >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = p;
            n0++;
        }
    }
    if (n0 <= 0) {
        return sample_argmax(probabilities, n);
    }

    // Build max-heap in-place: O(n0)
    for (int i = (n0 / 2) - 1; i >= 0; i--) {
        heapify_down_probindex(probindex, n0, i);
    }

    // Pop only as many top tokens as needed to exceed topp cumulative mass.
    // Store popped candidates at the tail [selected_start, n0) for sampling.
    int heap_size = n0;
    int selected_start = n0;
    float cumulative_prob = 0.0f;
    while (heap_size > 0) {
        ProbIndex top = heap_pop_max_probindex(probindex, &heap_size);
        probindex[--selected_start] = top;
        cumulative_prob += top.prob;
        if (cumulative_prob > topp) break;
    }

    // sample from the selected candidates
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = selected_start; i < n0; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[n0 - 1].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

#define CURRENT_TIME_IN_SECONDS (READ_CSR("mcycle") / MTIME_FREQ)

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    return (READ_CSR("mcycle") / MTIME_FREQ);
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        printf("STDERR: something is wrong, expected at least 1 prompt token\r\n");
        // exit(EXIT_FAILURE);
    }

    // start the main loop
    unsigned long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = READ_CSR("mcycle"); }
    }
    printf("\r\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        unsigned long end = READ_CSR("mcycle");
        printf("\r\nBENCHMARK: Total cycles: %lu\r\n", end-start);
        printf("BENCHMARK: Total tokens:\t%d\r\n", pos-1);
        printf("BENCHMARK: Cycles per token:\t%lu\r\n", (unsigned long)(end-start)/(pos-1));
        printf("BENCHMARK: Seconds per token:\t%lu\r\n", (unsigned long)((end-start)/target_frequency)/(pos-1));
        printf("BENCHMARK: Seconds per token (float):\t%f\r\n", ((float)(end-start)/(float)target_frequency)/(float)(pos-1));

        printf("BENCHMARK: CLOCK Frequency:\t%llu\r\n", (unsigned long long)target_frequency);
        printf("STDERR: achieved tok/s: %f\r\n", (pos-1) / (((double)(end-start))/target_frequency));
    }

    free(prompt_tokens);
}


void read_stdin(const char* guide, char* buffer, size_t bufsize) {
  // read a line from stdin, up to but not including \n
  // printf("%s", guide);
  // if (fgets(buffer, bufsize, stdin) != NULL) {
  //     size_t len = strlen(buffer);
  //     if (len > 0 && buffer[len - 1] == '\n') {
  //         buffer[len - 1] = '\0'; // strip newline
  //     }
  // }
  size_t char_offset = 0;
  size_t upper_bufsize_bound = bufsize - 1;
  printf("%s", guide);
  fflush(stdout);

  while(1) {
    unsigned char input_char = '\0';
    uart_receive(UART0, &input_char, 1, 100);

    if (input_char == '\b' && char_offset > 0) {
      // Backspace handling
      uart_transmit(UART0, &input_char, 1, 100);
      char_offset--;
      *(buffer + char_offset) = '\0';
    } else if (input_char == '\r') {
      // Newline (submit) handling
      uart_transmit(UART0, ASCII_CRLF, 2, 100);
      *(buffer + char_offset) = '\0';
      break;
    } else if (input_char > 31 && input_char < 127 && char_offset < upper_bufsize_bound) {
      // Any printable ascii character
      uart_transmit(UART0, &input_char, 1, 100);
      *(buffer + char_offset) = input_char;
      char_offset++;
    } else {
      // Send a bell to the terminal if some other condition.
      uart_transmit(UART0, ASCII_BEL, 1, 100);
    }
  }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    unsigned long start = 0;  // used to time our code, only initialized after first iteration

    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\r\n%s\r\n<</SYS>>\r\n\r\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        // if (token == 2) {
        //     unsigned long end = READ_CSR("mcycle");
        //     printf("\r\nBENCHMARK: Total cycles: %lu\r\n", end-start);
        //     printf("BENCHMARK: Total tokens:\t%d\r\n", pos-1);
        //     printf("BENCHMARK: Cycles per token:\t%lu\r\n", (unsigned long)(end-start)/(pos-1));
        //     printf("BENCHMARK: Seconds per token:\t%lu\r\n", (unsigned long)((end-start)/SYS_CLK_FREQ)/(pos-1));
        //     printf("BENCHMARK: Seconds per token (float):\t%f\r\n", ((float)(end-start)/(float)SYS_CLK_FREQ)/(float)(pos-1));

        //     printf("BENCHMARK: MTIME Frequency:\t%lu\r\n", MTIME_FREQ);
        //     user_turn = 1;
        //     start = 0;
        // }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;
        
        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
            if (start == 0) { start = READ_CSR("mcycle"); }
        }
        if (next == 2) { printf("\r\n"); }
    }

    unsigned long end = READ_CSR("mcycle");
    printf("\r\nBENCHMARK: Total cycles: %lu\r\n", end-start);
    printf("BENCHMARK: Total tokens:\t%d\r\n", pos-1);
    printf("BENCHMARK: Cycles per token:\t%lu\r\n", (unsigned long)(end-start)/(pos-1));
    printf("BENCHMARK: Seconds per token:\t%lu\r\n", (unsigned long)((end-start)/SYS_CLK_FREQ)/(pos-1));
    printf("BENCHMARK: Seconds per token (float):\t%f\r\n", ((float)(end-start)/(float)SYS_CLK_FREQ)/(float)(pos-1));

    printf("BENCHMARK: MTIME Frequency:\t%u\r\n", MTIME_FREQ);
    user_turn = 1;
    start = 0;
    printf("\r\n");
    free(prompt_tokens);
}


void app_main() {
  uint64_t mhartid = READ_CSR("mhartid");
  printf("Started BorAIq (Int8 Quantized) Inference Engine on hart ID %lu\r\n", mhartid);
#if defined(__riscv_vector)
  printf("Build flags: RVV attention kernels ON (__riscv_vector=1)\r\n");
#else
  printf("Build flags: RVV attention kernels OFF (__riscv_vector=0)\r\n");
#endif
#if defined(PREFILL_MULTICORE)
  printf("Build flags: PREFILL_MULTICORE ON\r\n");
#else
  printf("Build flags: PREFILL_MULTICORE OFF\r\n");
#endif
#if defined(VEC_SOFTMAX)
  printf("Build flags: VEC_SOFTMAX ON\r\n");
#else
  printf("Build flags: VEC_SOFTMAX OFF\r\n");
#endif
#if defined(TRANSPOSED_WEIGHTS)
  printf("Build flags: TRANSPOSED_WEIGHTS ON\r\n");
#else
  printf("Build flags: TRANSPOSED_WEIGHTS OFF\r\n");
#endif
#if defined(BORAIQ_TINY_SHAPE_GEMM)
  printf("Build flags: BORAIQ_TINY_SHAPE_GEMM ON\r\n");
#else
  printf("Build flags: BORAIQ_TINY_SHAPE_GEMM OFF\r\n");
#endif

  // Parameters //
  float temperature = 0.8f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 4;            // number of steps to run for (default 512)
  char *prompt = NULL;        // prompt string
  unsigned long long rng_seed = CLINT->MTIME; // seed rng with time by default
  GenMode mode = GENERATE;    // generate|chat
  char *system_prompt = NULL; // the (optional) system prompt to use in chat mode (I have it set up to ask screen if not given)

  // Parameter validation and overrides
  // if (rng_seed <= 0) rng_seed = CLINT->MTIME;
  if (temperature < 0.0) temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp) topp = 0.9;
  if (steps < 0) steps = 0;

  // Import from transformer binary
#ifdef PREFILL_MULTICORE
  Transformer* p_tfm = &_mc_transformer;
#else
  Transformer _tfm_buf;
  Transformer* p_tfm = &_tfm_buf;
#endif
  build_transformer(p_tfm);
  if (steps == 0 || steps > p_tfm->config.seq_len) steps = p_tfm->config.seq_len;

  // Import the tokenizer binary
  Tokenizer tokenizer;
  build_tokenizer_from_header(&tokenizer, p_tfm->config.vocab_size);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, p_tfm->config.vocab_size, temperature, topp, rng_seed);

#ifdef PREFILL_MULTICORE
  hthread_init();
  mc_start_worker(p_tfm);
#endif

  while (1) {
    // Disabled for testing. Should uncomment when ready for random stuff each run
    sampler.rng_state = CLINT->MTIME;
    printf("\r\nMTIME RNG State: %llu\r\n\r\n", sampler.rng_state);

    // run!
    if (mode == GENERATE) {
#ifdef PREFILL_MULTICORE
        generate_mc(p_tfm, &tokenizer, &sampler, prompt, steps);
#else
        generate(p_tfm, &tokenizer, &sampler, prompt, steps);
#endif
    } else {
        chat(p_tfm, &tokenizer, &sampler, NULL, NULL, steps);
    }

    printf("========================================\r\n");
    //msleep(1000);
  }
}

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(int argc, char **argv) {
  /* MCU Configuration--------------------------------------------------------*/
  
  configure_pll(PLL, target_frequency/50000000, 0);
  set_all_clocks(CLOCK_SELECTOR, 1);

  /* USER CODE BEGIN SysInit */
  // Initialize UART0 for Serial Monitor
  UART_InitType UART0_init_config;
  UART0_init_config.baudrate = 115200;
  UART0_init_config.mode = UART_MODE_TX_RX;
  UART0_init_config.stopbits = UART_STOPBITS_2;
  uart_init(UART0, &UART0_init_config);
  UART0->DIV = (target_frequency / 115200) - 1;

  /* USER CODE END SysInit */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  app_main();
  /* USER CODE END WHILE */
}


// Alternative HART runner. In multicore mode, threadlib provides __main.
#ifndef PREFILL_MULTICORE
void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
    asm volatile ("wfi");
  }
}
#endif
