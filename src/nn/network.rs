use ndarray::Array;

use crate::util::dtypes::*;
use super::layer::Layer;
use super::LearningRatePolicy;


pub struct UpdateArgs {
    cbatch: Int,
    lr: Float,
    momentum: Float,
    decay: Float,
    adam: Int,
    B1: Float,
    B2: Float,
    eps: Float,
    t: Int
}

pub struct Network {
    batch: usize,

    layers: Vec<Layer>,

    n_outputs: Int,
    outputs: Array<Float, usize>,

    n_inputs: Int,
    h: Int, w: Int, c: Int,

    policy: LearningRatePolicy,
    update_args: UpdateArgs,

    time_steps: Uchar,
    steps: Vec<u32>,

    max_batches: Int,
    cur_iter: Int,

    batch_size: Int,
    subdivisions: Int,
    seen: Int
    
}

/*
typedef struct network{
    float *workspace;
    int n;
    int batch;
	uint64_t *seen;
    float epoch;
    int subdivisions;
    float momentum;
    float decay;
    layer *layers;
    int outputs;
    float *output;
    learning_rate_policy policy;
    float learning_rate;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;
    int cudnn_half;
    int adam;
    float B1;
    float B2;
    float eps;
    int inputs;
    int h, w, c;
    int max_crop;
    int min_crop;
    int flip; // horizontal flip 50% probability augmentaiont for classifier training (default = 1)
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
	int small_object;
    int gpu_index;
    tree *hierarchy;
    #ifdef GPU
    float *input_state_gpu;
    float **input_gpu;
    float **truth_gpu;
	float **input16_gpu;
	float **output16_gpu;
	size_t *max_input16_size;
	size_t *max_output16_size;
	int wait_stream;
    #endif
} network;
*/