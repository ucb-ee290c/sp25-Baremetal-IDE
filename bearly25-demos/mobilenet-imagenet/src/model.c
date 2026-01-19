#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define MAX(X,Y) ( X > Y ? X : Y)
#define MIN(X,Y) ( X < Y ? X : Y)
#define CLIP(X,L) ( MAX(MIN(X,L), -L) )

#if __STDC_VERSION__ < 199901L
#define FUNC_PREFIX
#else
#define FUNC_PREFIX static inline
#endif


// Include all weights from header
#include "model_weights.h"

static union tensor_union_2 tu2;


/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.0/features.0.0/Conv
 */
FUNC_PREFIX void node__features_features_0_features_0_0_Conv( const float x[1][3][224][224], const float w[32][3][3][3], const float bias[32], float y[1][32][112][112] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<32; m++) {
		for( int32_t o0=0, i0=-1; o0<112; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<112; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<3; c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii1 >= 0 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.0/features.0.2/Clip
 */
FUNC_PREFIX void node__features_features_0_features_0_2_Clip( const float input[1][32][112][112], const float *min_val, const float *max_val, float output[1][32][112][112] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<32; i1++) {
	for (unsigned i2=0; i2<112; i2++) {
	for (unsigned i3=0; i3<112; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.1/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_1_conv_conv_0_conv_0_0_Conv( const float x[1][32][112][112], const float w[32][1][3][3], const float bias[32], float y[1][32][112][112] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 32
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<32; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<112; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<112; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 112 && ii1 >= 0 && ii1 < 112 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.1/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_1_conv_conv_0_conv_0_2_Clip( const float input[1][32][112][112], const float *min_val, const float *max_val, float output[1][32][112][112] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<32; i1++) {
	for (unsigned i2=0; i2<112; i2++) {
	for (unsigned i3=0; i3<112; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.1/conv/conv.1/Conv
 */
FUNC_PREFIX void node__features_features_1_conv_conv_1_Conv( const float x[1][32][112][112], const float w[16][32][1][1], const float bias[16], float y[1][16][112][112] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<16; m++) {
		for( int32_t o0=0, i0=0; o0<112; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<112; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<32; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.2/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_2_conv_conv_0_conv_0_0_Conv( const float x[1][16][112][112], const float w[96][16][1][1], const float bias[96], float y[1][96][112][112] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<112; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<112; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<16; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.2/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_2_conv_conv_0_conv_0_2_Clip( const float input[1][96][112][112], const float *min_val, const float *max_val, float output[1][96][112][112] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<96; i1++) {
	for (unsigned i2=0; i2<112; i2++) {
	for (unsigned i3=0; i3<112; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.2/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_2_conv_conv_1_conv_1_0_Conv( const float x[1][96][112][112], const float w[96][1][3][3], const float bias[96], float y[1][96][56][56] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 96
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<96; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<56; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<56; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii1 >= 0 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.2/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_2_conv_conv_1_conv_1_2_Clip( const float input[1][96][56][56], const float *min_val, const float *max_val, float output[1][96][56][56] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<96; i1++) {
	for (unsigned i2=0; i2<56; i2++) {
	for (unsigned i3=0; i3<56; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.2/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_2_conv_conv_2_Conv( const float x[1][96][56][56], const float w[24][96][1][1], const float bias[24], float y[1][24][56][56] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<24; m++) {
		for( int32_t o0=0, i0=0; o0<56; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<56; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.3/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_3_conv_conv_0_conv_0_0_Conv( const float x[1][24][56][56], const float w[144][24][1][1], const float bias[144], float y[1][144][56][56] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<144; m++) {
		for( int32_t o0=0, i0=0; o0<56; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<56; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<24; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.3/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_3_conv_conv_0_conv_0_2_Clip( const float input[1][144][56][56], const float *min_val, const float *max_val, float output[1][144][56][56] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<144; i1++) {
	for (unsigned i2=0; i2<56; i2++) {
	for (unsigned i3=0; i3<56; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.3/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_3_conv_conv_1_conv_1_0_Conv( const float x[1][144][56][56], const float w[144][1][3][3], const float bias[144], float y[1][144][56][56] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 144
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<144; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<56; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<56; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 56 && ii1 >= 0 && ii1 < 56 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.3/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_3_conv_conv_1_conv_1_2_Clip( const float input[1][144][56][56], const float *min_val, const float *max_val, float output[1][144][56][56] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<144; i1++) {
	for (unsigned i2=0; i2<56; i2++) {
	for (unsigned i3=0; i3<56; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.3/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_3_conv_conv_2_Conv( const float x[1][144][56][56], const float w[24][144][1][1], const float bias[24], float y[1][24][56][56] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<24; m++) {
		for( int32_t o0=0, i0=0; o0<56; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<56; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<144; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.3/Add
 */
FUNC_PREFIX void node__features_features_3_Add( const float A[1][24][56][56], const float B[1][24][56][56], float C[1][24][56][56] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<24; i1++)
	for (unsigned i2=0; i2<56; i2++)
	for (unsigned i3=0; i3<56; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.4/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_4_conv_conv_0_conv_0_0_Conv( const float x[1][24][56][56], const float w[144][24][1][1], const float bias[144], float y[1][144][56][56] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<144; m++) {
		for( int32_t o0=0, i0=0; o0<56; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<56; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<24; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.4/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_4_conv_conv_0_conv_0_2_Clip( const float input[1][144][56][56], const float *min_val, const float *max_val, float output[1][144][56][56] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<144; i1++) {
	for (unsigned i2=0; i2<56; i2++) {
	for (unsigned i3=0; i3<56; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.4/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_4_conv_conv_1_conv_1_0_Conv( const float x[1][144][56][56], const float w[144][1][3][3], const float bias[144], float y[1][144][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 144
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<144; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<28; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<28; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii1 >= 0 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.4/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_4_conv_conv_1_conv_1_2_Clip( const float input[1][144][28][28], const float *min_val, const float *max_val, float output[1][144][28][28] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<144; i1++) {
	for (unsigned i2=0; i2<28; i2++) {
	for (unsigned i3=0; i3<28; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.4/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_4_conv_conv_2_Conv( const float x[1][144][28][28], const float w[32][144][1][1], const float bias[32], float y[1][32][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<32; m++) {
		for( int32_t o0=0, i0=0; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<144; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.5/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_5_conv_conv_0_conv_0_0_Conv( const float x[1][32][28][28], const float w[192][32][1][1], const float bias[192], float y[1][192][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<32; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.5/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_5_conv_conv_0_conv_0_2_Clip( const float input[1][192][28][28], const float *min_val, const float *max_val, float output[1][192][28][28] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<28; i2++) {
	for (unsigned i3=0; i3<28; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.5/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_5_conv_conv_1_conv_1_0_Conv( const float x[1][192][28][28], const float w[192][1][3][3], const float bias[192], float y[1][192][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 192
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<192; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 28 && ii1 >= 0 && ii1 < 28 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.5/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_5_conv_conv_1_conv_1_2_Clip( const float input[1][192][28][28], const float *min_val, const float *max_val, float output[1][192][28][28] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<28; i2++) {
	for (unsigned i3=0; i3<28; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.5/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_5_conv_conv_2_Conv( const float x[1][192][28][28], const float w[32][192][1][1], const float bias[32], float y[1][32][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<32; m++) {
		for( int32_t o0=0, i0=0; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.5/Add
 */
FUNC_PREFIX void node__features_features_5_Add( const float A[1][32][28][28], const float B[1][32][28][28], float C[1][32][28][28] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<32; i1++)
	for (unsigned i2=0; i2<28; i2++)
	for (unsigned i3=0; i3<28; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.6/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_6_conv_conv_0_conv_0_0_Conv( const float x[1][32][28][28], const float w[192][32][1][1], const float bias[192], float y[1][192][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<32; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.6/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_6_conv_conv_0_conv_0_2_Clip( const float input[1][192][28][28], const float *min_val, const float *max_val, float output[1][192][28][28] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<28; i2++) {
	for (unsigned i3=0; i3<28; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.6/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_6_conv_conv_1_conv_1_0_Conv( const float x[1][192][28][28], const float w[192][1][3][3], const float bias[192], float y[1][192][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 192
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<192; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 28 && ii1 >= 0 && ii1 < 28 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.6/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_6_conv_conv_1_conv_1_2_Clip( const float input[1][192][28][28], const float *min_val, const float *max_val, float output[1][192][28][28] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<28; i2++) {
	for (unsigned i3=0; i3<28; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.6/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_6_conv_conv_2_Conv( const float x[1][192][28][28], const float w[32][192][1][1], const float bias[32], float y[1][32][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<32; m++) {
		for( int32_t o0=0, i0=0; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.6/Add
 */
FUNC_PREFIX void node__features_features_6_Add( const float A[1][32][28][28], const float B[1][32][28][28], float C[1][32][28][28] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<32; i1++)
	for (unsigned i2=0; i2<28; i2++)
	for (unsigned i3=0; i3<28; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.7/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_7_conv_conv_0_conv_0_0_Conv( const float x[1][32][28][28], const float w[192][32][1][1], const float bias[192], float y[1][192][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<32; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.7/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_7_conv_conv_0_conv_0_2_Clip( const float input[1][192][28][28], const float *min_val, const float *max_val, float output[1][192][28][28] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<28; i2++) {
	for (unsigned i3=0; i3<28; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.7/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_7_conv_conv_1_conv_1_0_Conv( const float x[1][192][28][28], const float w[192][1][3][3], const float bias[192], float y[1][192][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 192
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<192; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<14; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<14; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii1 >= 0 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.7/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_7_conv_conv_1_conv_1_2_Clip( const float input[1][192][14][14], const float *min_val, const float *max_val, float output[1][192][14][14] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<14; i2++) {
	for (unsigned i3=0; i3<14; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.7/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_7_conv_conv_2_Conv( const float x[1][192][14][14], const float w[64][192][1][1], const float bias[64], float y[1][64][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<64; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.8/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_8_conv_conv_0_conv_0_0_Conv( const float x[1][64][14][14], const float w[384][64][1][1], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<64; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.8/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_8_conv_conv_0_conv_0_2_Clip( const float input[1][384][14][14], const float *min_val, const float *max_val, float output[1][384][14][14] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<14; i2++) {
	for (unsigned i3=0; i3<14; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.8/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_8_conv_conv_1_conv_1_0_Conv( const float x[1][384][14][14], const float w[384][1][3][3], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 384
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<384; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 14 && ii1 >= 0 && ii1 < 14 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.8/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_8_conv_conv_1_conv_1_2_Clip( const float input[1][384][14][14], const float *min_val, const float *max_val, float output[1][384][14][14] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<14; i2++) {
	for (unsigned i3=0; i3<14; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.8/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_8_conv_conv_2_Conv( const float x[1][384][14][14], const float w[64][384][1][1], const float bias[64], float y[1][64][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<64; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.8/Add
 */
FUNC_PREFIX void node__features_features_8_Add( const float A[1][64][14][14], const float B[1][64][14][14], float C[1][64][14][14] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<64; i1++)
	for (unsigned i2=0; i2<14; i2++)
	for (unsigned i3=0; i3<14; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.9/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_9_conv_conv_0_conv_0_0_Conv( const float x[1][64][14][14], const float w[384][64][1][1], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<64; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.9/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_9_conv_conv_0_conv_0_2_Clip( const float input[1][384][14][14], const float *min_val, const float *max_val, float output[1][384][14][14] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<14; i2++) {
	for (unsigned i3=0; i3<14; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.9/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_9_conv_conv_1_conv_1_0_Conv( const float x[1][384][14][14], const float w[384][1][3][3], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 384
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<384; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 14 && ii1 >= 0 && ii1 < 14 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.9/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_9_conv_conv_1_conv_1_2_Clip( const float input[1][384][14][14], const float *min_val, const float *max_val, float output[1][384][14][14] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<14; i2++) {
	for (unsigned i3=0; i3<14; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.9/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_9_conv_conv_2_Conv( const float x[1][384][14][14], const float w[64][384][1][1], const float bias[64], float y[1][64][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<64; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.9/Add
 */
FUNC_PREFIX void node__features_features_9_Add( const float A[1][64][14][14], const float B[1][64][14][14], float C[1][64][14][14] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<64; i1++)
	for (unsigned i2=0; i2<14; i2++)
	for (unsigned i3=0; i3<14; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.10/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_10_conv_conv_0_conv_0_0_Conv( const float x[1][64][14][14], const float w[384][64][1][1], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<64; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.10/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_10_conv_conv_0_conv_0_2_Clip( const float input[1][384][14][14], const float *min_val, const float *max_val, float output[1][384][14][14] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<14; i2++) {
	for (unsigned i3=0; i3<14; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.10/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_10_conv_conv_1_conv_1_0_Conv( const float x[1][384][14][14], const float w[384][1][3][3], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 384
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<384; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 14 && ii1 >= 0 && ii1 < 14 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.10/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_10_conv_conv_1_conv_1_2_Clip( const float input[1][384][14][14], const float *min_val, const float *max_val, float output[1][384][14][14] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<14; i2++) {
	for (unsigned i3=0; i3<14; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.10/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_10_conv_conv_2_Conv( const float x[1][384][14][14], const float w[64][384][1][1], const float bias[64], float y[1][64][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<64; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.10/Add
 */
FUNC_PREFIX void node__features_features_10_Add( const float A[1][64][14][14], const float B[1][64][14][14], float C[1][64][14][14] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<64; i1++)
	for (unsigned i2=0; i2<14; i2++)
	for (unsigned i3=0; i3<14; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.11/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_11_conv_conv_0_conv_0_0_Conv( const float x[1][64][14][14], const float w[384][64][1][1], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<64; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.11/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_11_conv_conv_0_conv_0_2_Clip( const float input[1][384][14][14], const float *min_val, const float *max_val, float output[1][384][14][14] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<14; i2++) {
	for (unsigned i3=0; i3<14; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.11/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_11_conv_conv_1_conv_1_0_Conv( const float x[1][384][14][14], const float w[384][1][3][3], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 384
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<384; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 14 && ii1 >= 0 && ii1 < 14 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.11/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_11_conv_conv_1_conv_1_2_Clip( const float input[1][384][14][14], const float *min_val, const float *max_val, float output[1][384][14][14] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<14; i2++) {
	for (unsigned i3=0; i3<14; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.11/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_11_conv_conv_2_Conv( const float x[1][384][14][14], const float w[96][384][1][1], const float bias[96], float y[1][96][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.12/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_12_conv_conv_0_conv_0_0_Conv( const float x[1][96][14][14], const float w[576][96][1][1], const float bias[576], float y[1][576][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<576; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.12/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_12_conv_conv_0_conv_0_2_Clip( const float input[1][576][14][14], const float *min_val, const float *max_val, float output[1][576][14][14] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<576; i1++) {
	for (unsigned i2=0; i2<14; i2++) {
	for (unsigned i3=0; i3<14; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.12/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_12_conv_conv_1_conv_1_0_Conv( const float x[1][576][14][14], const float w[576][1][3][3], const float bias[576], float y[1][576][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 576
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<576; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 14 && ii1 >= 0 && ii1 < 14 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.12/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_12_conv_conv_1_conv_1_2_Clip( const float input[1][576][14][14], const float *min_val, const float *max_val, float output[1][576][14][14] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<576; i1++) {
	for (unsigned i2=0; i2<14; i2++) {
	for (unsigned i3=0; i3<14; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.12/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_12_conv_conv_2_Conv( const float x[1][576][14][14], const float w[96][576][1][1], const float bias[96], float y[1][96][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<576; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.12/Add
 */
FUNC_PREFIX void node__features_features_12_Add( const float A[1][96][14][14], const float B[1][96][14][14], float C[1][96][14][14] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<96; i1++)
	for (unsigned i2=0; i2<14; i2++)
	for (unsigned i3=0; i3<14; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.13/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_13_conv_conv_0_conv_0_0_Conv( const float x[1][96][14][14], const float w[576][96][1][1], const float bias[576], float y[1][576][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<576; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.13/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_13_conv_conv_0_conv_0_2_Clip( const float input[1][576][14][14], const float *min_val, const float *max_val, float output[1][576][14][14] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<576; i1++) {
	for (unsigned i2=0; i2<14; i2++) {
	for (unsigned i3=0; i3<14; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.13/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_13_conv_conv_1_conv_1_0_Conv( const float x[1][576][14][14], const float w[576][1][3][3], const float bias[576], float y[1][576][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 576
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<576; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 14 && ii1 >= 0 && ii1 < 14 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.13/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_13_conv_conv_1_conv_1_2_Clip( const float input[1][576][14][14], const float *min_val, const float *max_val, float output[1][576][14][14] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<576; i1++) {
	for (unsigned i2=0; i2<14; i2++) {
	for (unsigned i3=0; i3<14; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.13/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_13_conv_conv_2_Conv( const float x[1][576][14][14], const float w[96][576][1][1], const float bias[96], float y[1][96][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<576; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.13/Add
 */
FUNC_PREFIX void node__features_features_13_Add( const float A[1][96][14][14], const float B[1][96][14][14], float C[1][96][14][14] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<96; i1++)
	for (unsigned i2=0; i2<14; i2++)
	for (unsigned i3=0; i3<14; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.14/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_14_conv_conv_0_conv_0_0_Conv( const float x[1][96][14][14], const float w[576][96][1][1], const float bias[576], float y[1][576][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<576; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.14/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_14_conv_conv_0_conv_0_2_Clip( const float input[1][576][14][14], const float *min_val, const float *max_val, float output[1][576][14][14] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<576; i1++) {
	for (unsigned i2=0; i2<14; i2++) {
	for (unsigned i3=0; i3<14; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.14/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_14_conv_conv_1_conv_1_0_Conv( const float x[1][576][14][14], const float w[576][1][3][3], const float bias[576], float y[1][576][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 576
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<576; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<7; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<7; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii1 >= 0 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.14/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_14_conv_conv_1_conv_1_2_Clip( const float input[1][576][7][7], const float *min_val, const float *max_val, float output[1][576][7][7] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<576; i1++) {
	for (unsigned i2=0; i2<7; i2++) {
	for (unsigned i3=0; i3<7; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.14/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_14_conv_conv_2_Conv( const float x[1][576][7][7], const float w[160][576][1][1], const float bias[160], float y[1][160][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<160; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<576; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.15/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_15_conv_conv_0_conv_0_0_Conv( const float x[1][160][7][7], const float w[960][160][1][1], const float bias[960], float y[1][960][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<960; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<160; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.15/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_15_conv_conv_0_conv_0_2_Clip( const float input[1][960][7][7], const float *min_val, const float *max_val, float output[1][960][7][7] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<960; i1++) {
	for (unsigned i2=0; i2<7; i2++) {
	for (unsigned i3=0; i3<7; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.15/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_15_conv_conv_1_conv_1_0_Conv( const float x[1][960][7][7], const float w[960][1][3][3], const float bias[960], float y[1][960][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 960
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<960; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 7 && ii1 >= 0 && ii1 < 7 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.15/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_15_conv_conv_1_conv_1_2_Clip( const float input[1][960][7][7], const float *min_val, const float *max_val, float output[1][960][7][7] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<960; i1++) {
	for (unsigned i2=0; i2<7; i2++) {
	for (unsigned i3=0; i3<7; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.15/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_15_conv_conv_2_Conv( const float x[1][960][7][7], const float w[160][960][1][1], const float bias[160], float y[1][160][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<160; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<960; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.15/Add
 */
FUNC_PREFIX void node__features_features_15_Add( const float A[1][160][7][7], const float B[1][160][7][7], float C[1][160][7][7] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<160; i1++)
	for (unsigned i2=0; i2<7; i2++)
	for (unsigned i3=0; i3<7; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.16/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_16_conv_conv_0_conv_0_0_Conv( const float x[1][160][7][7], const float w[960][160][1][1], const float bias[960], float y[1][960][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<960; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<160; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.16/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_16_conv_conv_0_conv_0_2_Clip( const float input[1][960][7][7], const float *min_val, const float *max_val, float output[1][960][7][7] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<960; i1++) {
	for (unsigned i2=0; i2<7; i2++) {
	for (unsigned i3=0; i3<7; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.16/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_16_conv_conv_1_conv_1_0_Conv( const float x[1][960][7][7], const float w[960][1][3][3], const float bias[960], float y[1][960][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 960
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<960; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 7 && ii1 >= 0 && ii1 < 7 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.16/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_16_conv_conv_1_conv_1_2_Clip( const float input[1][960][7][7], const float *min_val, const float *max_val, float output[1][960][7][7] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<960; i1++) {
	for (unsigned i2=0; i2<7; i2++) {
	for (unsigned i3=0; i3<7; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.16/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_16_conv_conv_2_Conv( const float x[1][960][7][7], const float w[160][960][1][1], const float bias[160], float y[1][160][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<160; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<960; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.16/Add
 */
FUNC_PREFIX void node__features_features_16_Add( const float A[1][160][7][7], const float B[1][160][7][7], float C[1][160][7][7] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<160; i1++)
	for (unsigned i2=0; i2<7; i2++)
	for (unsigned i3=0; i3<7; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.17/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_17_conv_conv_0_conv_0_0_Conv( const float x[1][160][7][7], const float w[960][160][1][1], const float bias[960], float y[1][960][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<960; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<160; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.17/conv/conv.0/conv.0.2/Clip
 */
FUNC_PREFIX void node__features_features_17_conv_conv_0_conv_0_2_Clip( const float input[1][960][7][7], const float *min_val, const float *max_val, float output[1][960][7][7] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<960; i1++) {
	for (unsigned i2=0; i2<7; i2++) {
	for (unsigned i3=0; i3<7; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.17/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_17_conv_conv_1_conv_1_0_Conv( const float x[1][960][7][7], const float w[960][1][3][3], const float bias[960], float y[1][960][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 960
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<960; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 7 && ii1 >= 0 && ii1 < 7 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.17/conv/conv.1/conv.1.2/Clip
 */
FUNC_PREFIX void node__features_features_17_conv_conv_1_conv_1_2_Clip( const float input[1][960][7][7], const float *min_val, const float *max_val, float output[1][960][7][7] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<960; i1++) {
	for (unsigned i2=0; i2<7; i2++) {
	for (unsigned i3=0; i3<7; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.17/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_17_conv_conv_2_Conv( const float x[1][960][7][7], const float w[320][960][1][1], const float bias[320], float y[1][320][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<320; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<960; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.18/features.18.0/Conv
 */
FUNC_PREFIX void node__features_features_18_features_18_0_Conv( const float x[1][320][7][7], const float w[1280][320][1][1], const float bias[1280], float y[1][1280][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<1280; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<320; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Clip
 * Name in ONNX file: /features/features.18/features.18.2/Clip
 */
FUNC_PREFIX void node__features_features_18_features_18_2_Clip( const float input[1][1280][7][7], const float *min_val, const float *max_val, float output[1][1280][7][7] )
{
	/* Clip */
	float minv = *min_val;
	float maxv = *max_val;
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1280; i1++) {
	for (unsigned i2=0; i2<7; i2++) {
	for (unsigned i3=0; i3<7; i3++) {
		output[i0][i1][i2][i3] = MAX( MIN( input[i0][i1][i2][i3], maxv), minv);
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name in ONNX file: /GlobalAveragePool
 */
FUNC_PREFIX void node__GlobalAveragePool( const float input[1][1280][7][7], float output[1][1280][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<1280; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<7; d0++ ) {
		for( int32_t d1 = 0; d1<7; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 49;
	}
	}
}

/*
 * Operand:           Flatten
 * Name in ONNX file: /Flatten
 */
FUNC_PREFIX void node__Flatten( const float input[1][1280][1][1], float output[1][1280] )
{
	/* Flatten*/
	float *input_ = (float*)input;
	float *output_ = (float*)output;
	for( uint32_t i=0; i<1280; i++ )
		output_[i] = input_[i];

}

/*
 * Operand:           Gemm
 * Name in ONNX file: /classifier/classifier.1/Gemm
 */
FUNC_PREFIX void node__classifier_classifier_1_Gemm( const float A[1][1280], const float B[1000][1280], const float C[1000], float Y[1][1000] )
{
	/* Gemm */
	/* alpha   = 1.00000000000000000000
	   beta    = 1.00000000000000000000
	   transA  = 0
	   transB  = 1
	 */
	const int M = 1;
	const int K = 1280;
	const int N = 1000;
	float alpha = 1.00000000000000000000;
	float beta = 1.00000000000000000000;
	float (*C_)[1000]  = (float(*)[1000])C;
	for( uint32_t r=0; r<M; r++ )
		for( uint32_t c=0; c<N; c++ ) {
			float ABrc = 0;
			for( uint32_t i=0; i<K; i++ ) {
				float B_el = B[c][i];
				ABrc += A[r][i] * B_el;
			}
			float tmp = ABrc * alpha;
			tmp += C_[0][c] * beta;
			Y[r][c] = tmp;
	}
}


void entry(const float tensor_input[1][3][224][224], float tensor_logits[1][1000]){
	node__features_features_0_features_0_0_Conv( tensor_input, tensor_onnx__Conv_538, tensor_onnx__Conv_539, tu0.tensor__features_features_0_features_0_0_Conv_output_0);
	node__features_features_0_features_0_2_Clip( tu0.tensor__features_features_0_features_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_0_features_0_2_Clip_output_0);
	node__features_features_1_conv_conv_0_conv_0_0_Conv( tu1.tensor__features_features_0_features_0_2_Clip_output_0, tensor_onnx__Conv_541, tensor_onnx__Conv_542, tu0.tensor__features_features_1_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_1_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_1_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_1_conv_conv_1_Conv( tu1.tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_544, tensor_onnx__Conv_545, tu0.tensor__features_features_1_conv_conv_1_Conv_output_0);
	node__features_features_2_conv_conv_0_conv_0_0_Conv( tu0.tensor__features_features_1_conv_conv_1_Conv_output_0, tensor_onnx__Conv_547, tensor_onnx__Conv_548, tu1.tensor__features_features_2_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_2_conv_conv_0_conv_0_2_Clip( tu1.tensor__features_features_2_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu0.tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_2_conv_conv_1_conv_1_0_Conv( tu0.tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_550, tensor_onnx__Conv_551, tu1.tensor__features_features_2_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_2_conv_conv_1_conv_1_2_Clip( tu1.tensor__features_features_2_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu0.tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_2_conv_conv_2_Conv( tu0.tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_553, tensor_onnx__Conv_554, tu1.tensor__features_features_2_conv_conv_2_Conv_output_0);
	node__features_features_3_conv_conv_0_conv_0_0_Conv( tu1.tensor__features_features_2_conv_conv_2_Conv_output_0, tensor_onnx__Conv_556, tensor_onnx__Conv_557, tu0.tensor__features_features_3_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_3_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_3_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu2.tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_3_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_559, tensor_onnx__Conv_560, tu0.tensor__features_features_3_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_3_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_3_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu2.tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_3_conv_conv_2_Conv( tu2.tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_562, tensor_onnx__Conv_563, tu0.tensor__features_features_3_conv_conv_2_Conv_output_0);
	node__features_features_3_Add( tu1.tensor__features_features_2_conv_conv_2_Conv_output_0, tu0.tensor__features_features_3_conv_conv_2_Conv_output_0, tu2.tensor__features_features_3_Add_output_0);
	node__features_features_4_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_3_Add_output_0, tensor_onnx__Conv_565, tensor_onnx__Conv_566, tu0.tensor__features_features_4_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_4_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_4_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_4_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_568, tensor_onnx__Conv_569, tu0.tensor__features_features_4_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_4_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_4_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_4_conv_conv_2_Conv( tu1.tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_571, tensor_onnx__Conv_572, tu0.tensor__features_features_4_conv_conv_2_Conv_output_0);
	node__features_features_5_conv_conv_0_conv_0_0_Conv( tu0.tensor__features_features_4_conv_conv_2_Conv_output_0, tensor_onnx__Conv_574, tensor_onnx__Conv_575, tu1.tensor__features_features_5_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_5_conv_conv_0_conv_0_2_Clip( tu1.tensor__features_features_5_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu2.tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_5_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_577, tensor_onnx__Conv_578, tu1.tensor__features_features_5_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_5_conv_conv_1_conv_1_2_Clip( tu1.tensor__features_features_5_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu2.tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_5_conv_conv_2_Conv( tu2.tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_580, tensor_onnx__Conv_581, tu1.tensor__features_features_5_conv_conv_2_Conv_output_0);
	node__features_features_5_Add( tu0.tensor__features_features_4_conv_conv_2_Conv_output_0, tu1.tensor__features_features_5_conv_conv_2_Conv_output_0, tu2.tensor__features_features_5_Add_output_0);
	node__features_features_6_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_5_Add_output_0, tensor_onnx__Conv_583, tensor_onnx__Conv_584, tu0.tensor__features_features_6_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_6_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_6_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_6_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_586, tensor_onnx__Conv_587, tu0.tensor__features_features_6_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_6_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_6_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_6_conv_conv_2_Conv( tu1.tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_589, tensor_onnx__Conv_590, tu0.tensor__features_features_6_conv_conv_2_Conv_output_0);
	node__features_features_6_Add( tu2.tensor__features_features_5_Add_output_0, tu0.tensor__features_features_6_conv_conv_2_Conv_output_0, tu1.tensor__features_features_6_Add_output_0);
	node__features_features_7_conv_conv_0_conv_0_0_Conv( tu1.tensor__features_features_6_Add_output_0, tensor_onnx__Conv_592, tensor_onnx__Conv_593, tu0.tensor__features_features_7_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_7_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_7_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_7_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_595, tensor_onnx__Conv_596, tu0.tensor__features_features_7_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_7_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_7_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_7_conv_conv_2_Conv( tu1.tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_598, tensor_onnx__Conv_599, tu0.tensor__features_features_7_conv_conv_2_Conv_output_0);
	node__features_features_8_conv_conv_0_conv_0_0_Conv( tu0.tensor__features_features_7_conv_conv_2_Conv_output_0, tensor_onnx__Conv_601, tensor_onnx__Conv_602, tu1.tensor__features_features_8_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_8_conv_conv_0_conv_0_2_Clip( tu1.tensor__features_features_8_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu2.tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_8_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_604, tensor_onnx__Conv_605, tu1.tensor__features_features_8_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_8_conv_conv_1_conv_1_2_Clip( tu1.tensor__features_features_8_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu2.tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_8_conv_conv_2_Conv( tu2.tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_607, tensor_onnx__Conv_608, tu1.tensor__features_features_8_conv_conv_2_Conv_output_0);
	node__features_features_8_Add( tu0.tensor__features_features_7_conv_conv_2_Conv_output_0, tu1.tensor__features_features_8_conv_conv_2_Conv_output_0, tu2.tensor__features_features_8_Add_output_0);
	node__features_features_9_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_8_Add_output_0, tensor_onnx__Conv_610, tensor_onnx__Conv_611, tu0.tensor__features_features_9_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_9_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_9_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_9_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_613, tensor_onnx__Conv_614, tu0.tensor__features_features_9_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_9_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_9_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_9_conv_conv_2_Conv( tu1.tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_616, tensor_onnx__Conv_617, tu0.tensor__features_features_9_conv_conv_2_Conv_output_0);
	node__features_features_9_Add( tu2.tensor__features_features_8_Add_output_0, tu0.tensor__features_features_9_conv_conv_2_Conv_output_0, tu1.tensor__features_features_9_Add_output_0);
	node__features_features_10_conv_conv_0_conv_0_0_Conv( tu1.tensor__features_features_9_Add_output_0, tensor_onnx__Conv_619, tensor_onnx__Conv_620, tu0.tensor__features_features_10_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_10_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_10_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu2.tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_10_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_622, tensor_onnx__Conv_623, tu0.tensor__features_features_10_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_10_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_10_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu2.tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_10_conv_conv_2_Conv( tu2.tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_625, tensor_onnx__Conv_626, tu0.tensor__features_features_10_conv_conv_2_Conv_output_0);
	node__features_features_10_Add( tu1.tensor__features_features_9_Add_output_0, tu0.tensor__features_features_10_conv_conv_2_Conv_output_0, tu2.tensor__features_features_10_Add_output_0);
	node__features_features_11_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_10_Add_output_0, tensor_onnx__Conv_628, tensor_onnx__Conv_629, tu0.tensor__features_features_11_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_11_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_11_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_11_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_631, tensor_onnx__Conv_632, tu0.tensor__features_features_11_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_11_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_11_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_11_conv_conv_2_Conv( tu1.tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_634, tensor_onnx__Conv_635, tu0.tensor__features_features_11_conv_conv_2_Conv_output_0);
	node__features_features_12_conv_conv_0_conv_0_0_Conv( tu0.tensor__features_features_11_conv_conv_2_Conv_output_0, tensor_onnx__Conv_637, tensor_onnx__Conv_638, tu1.tensor__features_features_12_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_12_conv_conv_0_conv_0_2_Clip( tu1.tensor__features_features_12_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu2.tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_12_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_640, tensor_onnx__Conv_641, tu1.tensor__features_features_12_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_12_conv_conv_1_conv_1_2_Clip( tu1.tensor__features_features_12_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu2.tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_12_conv_conv_2_Conv( tu2.tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_643, tensor_onnx__Conv_644, tu1.tensor__features_features_12_conv_conv_2_Conv_output_0);
	node__features_features_12_Add( tu0.tensor__features_features_11_conv_conv_2_Conv_output_0, tu1.tensor__features_features_12_conv_conv_2_Conv_output_0, tu2.tensor__features_features_12_Add_output_0);
	node__features_features_13_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_12_Add_output_0, tensor_onnx__Conv_646, tensor_onnx__Conv_647, tu0.tensor__features_features_13_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_13_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_13_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_13_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_649, tensor_onnx__Conv_650, tu0.tensor__features_features_13_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_13_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_13_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_13_conv_conv_2_Conv( tu1.tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_652, tensor_onnx__Conv_653, tu0.tensor__features_features_13_conv_conv_2_Conv_output_0);
	node__features_features_13_Add( tu2.tensor__features_features_12_Add_output_0, tu0.tensor__features_features_13_conv_conv_2_Conv_output_0, tu1.tensor__features_features_13_Add_output_0);
	node__features_features_14_conv_conv_0_conv_0_0_Conv( tu1.tensor__features_features_13_Add_output_0, tensor_onnx__Conv_655, tensor_onnx__Conv_656, tu0.tensor__features_features_14_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_14_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_14_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_14_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_658, tensor_onnx__Conv_659, tu0.tensor__features_features_14_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_14_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_14_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_14_conv_conv_2_Conv( tu1.tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_661, tensor_onnx__Conv_662, tu0.tensor__features_features_14_conv_conv_2_Conv_output_0);
	node__features_features_15_conv_conv_0_conv_0_0_Conv( tu0.tensor__features_features_14_conv_conv_2_Conv_output_0, tensor_onnx__Conv_664, tensor_onnx__Conv_665, tu1.tensor__features_features_15_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_15_conv_conv_0_conv_0_2_Clip( tu1.tensor__features_features_15_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu2.tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_15_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_667, tensor_onnx__Conv_668, tu1.tensor__features_features_15_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_15_conv_conv_1_conv_1_2_Clip( tu1.tensor__features_features_15_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu2.tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_15_conv_conv_2_Conv( tu2.tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_670, tensor_onnx__Conv_671, tu1.tensor__features_features_15_conv_conv_2_Conv_output_0);
	node__features_features_15_Add( tu0.tensor__features_features_14_conv_conv_2_Conv_output_0, tu1.tensor__features_features_15_conv_conv_2_Conv_output_0, tu2.tensor__features_features_15_Add_output_0);
	node__features_features_16_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_15_Add_output_0, tensor_onnx__Conv_673, tensor_onnx__Conv_674, tu0.tensor__features_features_16_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_16_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_16_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_16_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_676, tensor_onnx__Conv_677, tu0.tensor__features_features_16_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_16_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_16_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_16_conv_conv_2_Conv( tu1.tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_679, tensor_onnx__Conv_680, tu0.tensor__features_features_16_conv_conv_2_Conv_output_0);
	node__features_features_16_Add( tu2.tensor__features_features_15_Add_output_0, tu0.tensor__features_features_16_conv_conv_2_Conv_output_0, tu1.tensor__features_features_16_Add_output_0);
	node__features_features_17_conv_conv_0_conv_0_0_Conv( tu1.tensor__features_features_16_Add_output_0, tensor_onnx__Conv_682, tensor_onnx__Conv_683, tu0.tensor__features_features_17_conv_conv_0_conv_0_0_Conv_output_0);
	node__features_features_17_conv_conv_0_conv_0_2_Clip( tu0.tensor__features_features_17_conv_conv_0_conv_0_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_17_conv_conv_1_conv_1_0_Conv( tu1.tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0, tensor_onnx__Conv_685, tensor_onnx__Conv_686, tu0.tensor__features_features_17_conv_conv_1_conv_1_0_Conv_output_0);
	node__features_features_17_conv_conv_1_conv_1_2_Clip( tu0.tensor__features_features_17_conv_conv_1_conv_1_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu1.tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_17_conv_conv_2_Conv( tu1.tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0, tensor_onnx__Conv_688, tensor_onnx__Conv_689, tu0.tensor__features_features_17_conv_conv_2_Conv_output_0);
	node__features_features_18_features_18_0_Conv( tu0.tensor__features_features_17_conv_conv_2_Conv_output_0, tensor_onnx__Conv_691, tensor_onnx__Conv_692, tu1.tensor__features_features_18_features_18_0_Conv_output_0);
	node__features_features_18_features_18_2_Clip( tu1.tensor__features_features_18_features_18_0_Conv_output_0, &tensor__features_features_0_features_0_2_Constant_output_0, &tensor__features_features_0_features_0_2_Constant_1_output_0, tu0.tensor__features_features_18_features_18_2_Clip_output_0);
	node__GlobalAveragePool( tu0.tensor__features_features_18_features_18_2_Clip_output_0, tu1.tensor__GlobalAveragePool_output_0);
	node__Flatten( tu1.tensor__GlobalAveragePool_output_0, tu0.tensor__Flatten_output_0);
	node__classifier_classifier_1_Gemm( tu0.tensor__Flatten_output_0, tensor_classifier_1_weight, tensor_classifier_1_bias, tensor_logits);
}
