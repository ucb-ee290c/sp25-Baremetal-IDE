/*
 * main_zerocopy.c - MobileNetV2 ImageNet Demo
 *
 * Entry point for running MobileNetV2 inference on the Bearly25 chip.
 * Uses the onnx2c-generated model with zero-copy weight access.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "model.h"
#include "test_inputs.h"
#include "imagenet_labels.h"

// #define USE_CAT_IMAGE            /* Expected: 282 (tiger cat) */
// #define USE_DOG_IMAGE            /* Expected: 208 (Labrador retriever) */
// #define USE_TIGER_IMAGE          /* Expected: 292 (tiger) */
#define USE_BACKPACK_IMAGE       /* Expected: 414 (backpack) */
// #define USE_BENCH_IMAGE             /* Expected: 703 (park bench) */

/* Output buffer */
static float output[1][1000];

/* Find the index of the maximum value (argmax) */
static int argmax(const float *arr, int n) {
    int max_idx = 0;
    float max_val = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int main(void) {
    printf("MobileNetV2 ImageNet Demo\n");
    printf("==============================================\n\n");

    /* Select input based on which image is enabled */
#if defined(USE_CAT_IMAGE)
    const float (*input)[3][224][224] = test_input_cat;
    const char *image_name = "cat";
    int expected_class = 282;
#elif defined(USE_DOG_IMAGE)
    const float (*input)[3][224][224] = test_input_dog;
    const char *image_name = "dog";
    int expected_class = 208;
#elif defined(USE_TIGER_IMAGE)
    const float (*input)[3][224][224] = test_input_tiger;
    const char *image_name = "tiger";
    int expected_class = 292;
#elif defined(USE_BACKPACK_IMAGE)
    const float (*input)[3][224][224] = test_input_backpack;
    const char *image_name = "backpack";
    int expected_class = 414;
#elif defined(USE_BENCH_IMAGE)
    const float (*input)[3][224][224] = test_input_park_bench;
    const char *image_name = "park_bench";
    int expected_class = 703;
#else
    #error "No test image selected! Uncomment one of the USE_*_IMAGE defines."
#endif

    printf("Test image: %s\n", image_name);
    printf("Expected class: %d\n\n", expected_class);

    printf("Running inference...\n");

    /* Run inference */
    entry(*input, output);

    printf("Inference complete!\n\n");

    /* Find predicted class */
    int predicted_class = argmax(output[0], 1000);
    float confidence = output[0][predicted_class];

    /* Display results */
    printf("Results:\n");
    printf("  Predicted class: %d (%s)\n", predicted_class, imagenet_labels[predicted_class]);
    printf("  Expected class:  %d (%s)\n", expected_class, imagenet_labels[expected_class]);
    printf("  Logit value:     %f\n", confidence);
    printf("  Match: %s\n", 
           predicted_class == expected_class ? "✓ YES" : "✗ NO");

    return (predicted_class == expected_class) ? 0 : 1;
}
