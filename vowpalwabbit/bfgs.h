/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD
license as described in the file LICENSE.
 */
#ifndef BFGS_H
#define BFGS_H

#define LEARN_OK 0
#define LEARN_CURV 1
#define LEARN_CONV 2

#include <sys/timeb.h>

namespace BFGS {
  LEARNER::learner* setup(vw& all, po::variables_map& vm);

  struct bfgs {
    vw* all;
    double wolfe1_bound;
    
    size_t final_pass;
    struct timeb t_start, t_end;
    double net_comm_time;
    
    struct timeb t_start_global, t_end_global;
    double net_time;
    
    v_array<float> predictions;
    size_t example_number;
    size_t current_pass;
    size_t no_win_counter;
    size_t early_stop_thres;
    
    // default transition behavior
    bool first_hessian_on;
    bool backstep_on;
    
    // set by initializer
    int mem_stride;
    bool output_regularizer;
    float* mem;
    double* rho;
    double* alpha;
    
    weight* regularizers;
    // the below needs to be included when resetting, in addition to preconditioner and derivative
    int lastj, origin;
    double loss_sum, previous_loss_sum;
    float step_size;
    double importance_weight_sum;
    double curvature;
    
    // first pass specification
    bool first_pass;
    bool gradient_pass;
    bool preconditioner_pass;
  };

  int process_pass(vw& all, bfgs& b);
  void reset_state(vw& all, bfgs& b, bool zero);
  void save_load_regularizer(vw& all, bfgs& b, io_buf& model_file, bool read, bool text);
  void preconditioner_to_regularizer(vw& all, bfgs& b, float regularization);
}

#endif
