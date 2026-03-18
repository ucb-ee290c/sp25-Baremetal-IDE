#ifndef MODEL_H
#define MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Run MobileNetV2 inference
 *
 * @param tensor_input  Input image in NCHW format [1][3][224][224]
 *                      Normalized: (pixel/255 - mean) / std
 *                      mean = [0.485, 0.456, 0.406] (RGB)
 *                      std  = [0.229, 0.224, 0.225] (RGB)
 *
 * @param tensor_logits Output logits [1][1000] for ImageNet classes
 *                      Apply softmax for probabilities
 */
void entry(const float tensor_input[1][3][224][224], 
           float tensor_logits[1][1000]);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_H */
