/*
 * main_int8.c - Test harness for INT8 quantized MobileNetV2
 * 
 * Compares INT8 model output against expected ImageNet classes.
 */

#include <stdio.h>

// Include the INT8 model (split into separate files)
#include "model_int8.c"

// ImageNet class labels
#include "imagenet_labels.h"

// Test images
#include "test_inputs.h"

#include "simple_setup.h"

uint64_t target_frequency = 500000000l;

// Find the index of maximum value in an array
int argmax(const float* arr, int n) {
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

void run_inference(const char* name, const float input[3][224][224], int expected_class) {
    float output[1][1000];
    
    printf("\nTest: %s\n", name);
    printf("  Expected class: %d (%s)\n", expected_class, imagenet_labels[expected_class]);
    
    // Run inference - the INT8 model handles quant/dequant internally
    entry((const float(*)[3][224][224])input, output);
    
    // Find predicted class
    int predicted = argmax(output[0], 1000);
    float logit = output[0][predicted];
    
    printf("  Predicted class: %d (%s)\n", predicted, imagenet_labels[predicted]);
    printf("  Logit value: %.4f\n", logit);
    
    if (predicted == expected_class) {
        printf("  Result: ✓ PASS\n");
    } else {
        printf("  Result: ✗ FAIL (expected %d, got %d)\n", expected_class, predicted);
    }
}

int main(int argc, char** argv) {
    init_test(target_frequency);

    printf("==============================================\n");
    printf("MobileNetV2 INT8 Quantized Model Test\n");
    printf("==============================================\n");
    
    printf("\nModel info:\n");
    printf("  Precision: INT8 weights, UINT8 activations\n");
    printf("  Input: float [1, 3, 224, 224]\n");
    printf("  Output: float [1, 1000] (logits)\n");
    
    // Run tests with embedded test images
    run_inference("Cat", test_input_cat, 282);               // tabby cat
    run_inference("Dog", test_input_dog, 208);               // Samoyed
    run_inference("Tiger", test_input_tiger, 292);           // tiger
    run_inference("Park Bench", test_input_park_bench, 703); // park bench
    run_inference("Backpack", test_input_backpack, 414);     // backpack
    
    printf("\n==============================================\n");
    printf("INT8 Quantization Test Complete\n");
    printf("==============================================\n");
    
    return 0;
}
