use ndarray::{Array, Ix, IxDyn, Shape};

use super::layer::{Layer};
use super::{LearningRatePolicy, DefaultBatch};
use crate::blas::convolutional_out_height;
use crate::util::dtypes::*;

use super::state::NetworkState;

pub struct UpdateArgs {
    pub cbatch: usize,
    pub lr: Float,
    pub momentum: Float,
    pub decay: Float,
    pub adam: usize,
    pub B1: Float,
    pub B2: Float,
    pub eps: Float,
    pub t: usize,
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
    
    pub data_path: String, 

    pub layers: Vec<Layer>,

    pub n_outputs: usize,
    pub outputs: Array2F,

    pub n_inputs: usize,
    pub inputs: Array4F,

    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,

    pub policy: LearningRatePolicy,
    pub update_args: UpdateArgs,

    pub steps: Vec<usize>,

    pub max_batches: usize,
    pub cur_iter: usize,

    pub batch_size: usize,
    pub subdivisions: usize,
    pub seen: usize,

    pub workspace: Array4F
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
        batch: usize,
        data_path: String,
        subdivisions: usize,
        n_inputs: usize,
        n_outputs: usize,
        t: usize,
        h: usize,
        w: usize,
        c: usize,
        policy: LearningRatePolicy,
        ua: UpdateArgs,
        steps: Vec<usize>,
        max_batches: usize,
    ) -> Network {
        let ishape = [batch, h, w, c];

        Network {
            batch: 0,

            data_path: data_path,

            layers: Vec::<Layer>::new(),

            n_outputs: n_outputs,
            outputs: Array2F::zeros([n_outputs, 5]),

            n_inputs: n_inputs,
            inputs: Array4F::zeros(ishape),

            batch_size: batch,
            t: t,
            h: h,
            w: w,
            c: c,

            policy: policy,
            update_args: ua,

            steps: steps,

            max_batches: max_batches,
            cur_iter: 0,

            subdivisions: subdivisions,
            seen: 0,

            workspace: Array4F::zeros([0, 0, 0, 0]),
        }
    }

    fn update_workspace(&mut self) -> (){}

    fn train_batch(&mut self) -> (){}

    fn forward(&mut self, state: &mut NetworkState) -> () {
        self.workspace.clone_from(&state.workspace);

        let n = self.layers.len();

        for i in 0..n {
            state.index = i as i32;
            let l = &mut self.layers[i];

            l.forward(state);

            state.input.clone_from(&l.outputs);
        }
    }

    fn backward(&mut self, state: &mut NetworkState) -> () {
        /*
        store original state input and delta

        For each i = n-1 .. 0:
            state index = i
            l = self.layers[i]

            (don't modify the 0th layer's )

            l.backward()
        */
    }

    pub fn epoch(&mut self, state: &mut NetworkState) -> () {
        self.forward(state);
        self.backward(state);
    }

}

impl Default for Network {
    fn default() -> Network {
        Network {
            batch: 0, // batch_num = (*net.seen)/(net.batch*net.subdivisions);

            data_path: String::new(),

            layers: Vec::<Layer>::new(),

            n_outputs: 1,
            outputs: Array2F::zeros([0, 0]),

            n_inputs: 1,
            inputs: Array4F::zeros([0, 0, 0, 0]),

            batch_size: DefaultBatch,
            subdivisions: 4,

            t: 0,
            h: 0,
            w: 0,
            c: 0,

            policy: LearningRatePolicy::Steps,
            update_args: UpdateArgs::default(),

            steps: vec![3600, 4800, 6000],

            max_batches: 6000,
            cur_iter: 0,
            seen: 0,

            workspace: Array4F::zeros([0, 0, 0, 0]),
        }
    }
}
