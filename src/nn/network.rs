use ndarray::{Array, Ix, IxDyn, Shape};

use super::layer::Layer;
use super::LearningRatePolicy;
use crate::util::dtypes::*;

pub struct UpdateArgs {
    cbatch: Int,
    lr: Float,
    momentum: Float,
    decay: Float,
    adam: Int,
    B1: Float,
    B2: Float,
    eps: Float,
    t: Int,
}

impl Default for UpdateArgs {
    fn default() -> Self {
        UpdateArgs {
            cbatch: 4,
            lr: 0.001,
            momentum: 0.21,
            decay: 0.0,
            adam: 0,
            B1: 0.0,
            B2: 0.0,
            eps: 0.00001,
            t: 0,
        }
    }
}

pub struct Network {
    pub batch: usize,

    pub layers: Vec<Layer>,

    pub n_outputs: Int,
    pub outputs: FloatArr,

    pub n_inputs: Int,
    pub inputs: FloatArr,

    pub h: Int,
    pub w: Int,
    pub c: Int,

    pub policy: LearningRatePolicy,
    pub update_args: UpdateArgs,

    pub time_steps: Uchar,
    pub steps: Vec<Int>,

    pub max_batches: Int,
    pub cur_iter: Int,

    pub batch_size: Int,
    pub subdivisions: Int,
    pub seen: Int,
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

impl Network {
    pub fn add_layer(&mut self, l: Layer) -> () {
        self.layers.push(l)
    }

    pub fn new(
        batch: Int,
        subdivisions: Int,
        n_inputs: Int,
        n_outputs: Int,
        h: Int,
        w: Int,
        c: Int,
        time_steps: Uchar,
        i_shape: Shape<IxDyn>,
        o_shape: Shape<IxDyn>,
        policy: LearningRatePolicy,
        ua: UpdateArgs,
        steps: Vec<Int>,
        max_batches: Int,
    ) -> Network {
        Network {
            batch: 0,

            layers: Vec::<Layer>::new(),

            n_outputs: n_outputs,
            outputs: FloatArr::zeros(o_shape),

            n_inputs: n_inputs,
            inputs: FloatArr::zeros(i_shape),

            batch_size: batch,
            h: h,
            w: w,
            c: c,

            policy: policy,
            update_args: ua,

            time_steps: time_steps,
            steps: steps,

            max_batches: max_batches,
            cur_iter: 0,

            subdivisions: subdivisions,
            seen: 0,
        }
    }
}

impl Default for Network {
    fn default() -> Network {
        Network {
            batch: 0,

            layers: Vec::<Layer>::new(),

            n_outputs: 1,
            outputs: FloatArr::zeros(IxDyn(&[1])),

            n_inputs: 1,
            inputs: FloatArr::zeros(IxDyn(&[1])),

            batch_size: 16,
            subdivisions: 4,

            h: 0,
            w: 0,
            c: 0,

            policy: LearningRatePolicy::Steps,
            update_args: UpdateArgs::default(),

            time_steps: 0,
            steps: vec![3600, 4800, 6000],

            max_batches: 6000,
            cur_iter: 0,

            seen: 0,
        }
    }
}
