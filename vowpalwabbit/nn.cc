/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <stdlib.h>
#include <sys/time.h>

#include "reductions.h"
#include "constant.h"
#include "simple_label.h"
#include "rand48.h"
#include "gd.h"
#include "bfgs.h"
#include "allreduce.h"
#include "accumulate.h"

using namespace std;
using namespace LEARNER;
using namespace BFGS;

namespace NN {
  const float hidden_min_activation = -3;
  const float hidden_max_activation = 3;
  const int nn_constant = 533357803;
  //  const float min_prob = 0.0069f;
  const float min_prob = 0.003f;

  struct nn {
    uint32_t k;
    loss_function* squared_loss;
    example output_layer;
    float prediction;
    size_t increment;
    bool dropout;
    uint64_t xsubi;
    uint64_t save_xsubi;
    bool inpass;
    bool finished_setup;

    // diagnostics
    double train_time, subsample_time;
    time_t subsample_start_time;

    // active flags
    bool active;
    bool active_pool_greedy;
    bool para_active;
    bool active_bfgs;
    size_t active_passes;
    float active_reg_base; // initial regularization -- this is kind of hacky right now.

    // pool maintainence
    size_t pool_size;
    size_t pool_pos;
    size_t subsample;
    example** pool;
    size_t numqueries;
    size_t numexamples;
    size_t local_begin, local_end;
    float current_t;
    int save_interval;

    // para-active state
    std::string* span_server;
    size_t unique_id; // unique id for each node in the network id == 0 means extra io
    size_t total; //total number of nodes
    size_t node; // node id number
    node_socks* socks;
    bool all_done;
    bool local_done;
    time_t start_time;

    vw* all;
  };

#define cast_uint32_t static_cast<uint32_t>

  static inline float
  fastpow2 (float p)
  {
    float offset = (p < 0) ? 1.0f : 0.0f;
    float clipp = (p < -126) ? -126.0f : p;
    int w = (int)clipp;
    float z = clipp - w + offset;
    union { uint32_t i; float f; } v = { cast_uint32_t ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

    return v.f;
  }

  static inline float
  fastexp (float p)
  {
    return fastpow2 (1.442695040f * p);
  }

  static inline float
  fasttanh (float p)
  {
    return -1.0f + 2.0f / (1.0f + fastexp (-2.0f * p));
  }

  void finish_setup (nn& n, vw& all)
  {
    // TODO: output_layer audit

    memset (&n.output_layer, 0, sizeof (n.output_layer));
    n.output_layer.indices.push_back(nn_output_namespace);
    feature output = {1., nn_constant << all.reg.stride_shift};

    for (unsigned int i = 0; i < n.k; ++i)
      {
        n.output_layer.atomics[nn_output_namespace].push_back(output);
        ++n.output_layer.num_features;
        output.weight_index += (uint32_t)n.increment;
      }

    if (! n.inpass) 
      {
        n.output_layer.atomics[nn_output_namespace].push_back(output);
        ++n.output_layer.num_features;
      }

    n.output_layer.in_use = true;

    n.finished_setup = true;
  }

  void end_pass(nn& n)
  {
    if (n.all->bfgs)
      n.xsubi = n.save_xsubi;
    if (n.active_bfgs) {
      cerr << "Total training time " << n.train_time << endl;
      cerr << "Total subsampling time " << n.subsample_time << endl;
    }
  }

  template <bool is_learn>
  void passive_predict_or_learn(nn& n, learner& base, example& ec)
  {
    bool shouldOutput = n.all->raw_prediction > 0;
    if (! n.finished_setup)
      finish_setup (n, *(n.all));

    label_data* ld = (label_data*)ec.ld;
    float save_label = ld->label;
    void (*save_set_minmax) (shared_data*, float) = n.all->set_minmax;
    float save_min_label;
    float save_max_label;
    float dropscale = n.dropout ? 2.0f : 1.0f;
    loss_function* save_loss = n.all->loss;

    float* hidden_units = (float*) alloca (n.k * sizeof (float));
    bool* dropped_out = (bool*) alloca (n.k * sizeof (bool));
  
    string outputString;
    stringstream outputStringStream(outputString);

    n.all->set_minmax = noop_mm;
    n.all->loss = n.squared_loss;
    save_min_label = n.all->sd->min_label;
    n.all->sd->min_label = hidden_min_activation;
    save_max_label = n.all->sd->max_label;
    n.all->sd->max_label = hidden_max_activation;
    //ld->label = FLT_MAX;
    for (unsigned int i = 0; i < n.k; ++i)
      {
        uint32_t biasindex = (uint32_t) constant * (n.all->wpp << n.all->reg.stride_shift) + i * (uint32_t)n.increment + ec.ft_offset;
        weight* w = &n.all->reg.weight_vector[biasindex & n.all->reg.weight_mask];
        
        // avoid saddle point at 0
        if (*w == 0)
          {
            w[0] = (float) (frand48 () - 0.5);

            if (n.dropout && n.all->normalized_updates)
              w[n.all->normalized_idx] = 1e-4f;
          }

	base.predict(ec, i);

        hidden_units[i] = ld->prediction;

        dropped_out[i] = (n.dropout && merand48 (n.xsubi) < 0.5);

        if (shouldOutput) {
          if (i > 0) outputStringStream << ' ';
          outputStringStream << i << ':' << ec.partial_prediction << ',' << fasttanh (hidden_units[i]);
        }
      }
    //ld->label = save_label;
    n.all->loss = save_loss;
    n.all->set_minmax = save_set_minmax;
    n.all->sd->min_label = save_min_label;
    n.all->sd->max_label = save_max_label;

    bool converse = false;
    float save_partial_prediction = 0;
    float save_final_prediction = 0;
    float save_ec_loss = 0;

CONVERSE: // That's right, I'm using goto.  So sue me.

    n.output_layer.total_sum_feat_sq = 1;
    n.output_layer.sum_feat_sq[nn_output_namespace] = 1;

    for (unsigned int i = 0; i < n.k; ++i)
      {
        float sigmah = 
          (dropped_out[i]) ? 0.0f : dropscale * fasttanh (hidden_units[i]);
        n.output_layer.atomics[nn_output_namespace][i].x = sigmah;
        n.output_layer.total_sum_feat_sq += sigmah * sigmah;
        n.output_layer.sum_feat_sq[nn_output_namespace] += sigmah * sigmah;

        uint32_t nuindex = n.output_layer.atomics[nn_output_namespace][i].weight_index + (n.k * (uint32_t)n.increment) + ec.ft_offset;
        weight* w = &n.all->reg.weight_vector[nuindex & n.all->reg.weight_mask];
        
        // avoid saddle point at 0
        if (*w == 0)
          {
            float sqrtk = sqrt ((float)n.k);
            w[0] = (float) (frand48 () - 0.5) / sqrtk;

            if (n.dropout && n.all->normalized_updates)
              w[n.all->normalized_idx] = 1e-4f;
          }
      }
    if (n.inpass) {
      // TODO: this is not correct if there is something in the 
      // nn_output_namespace but at least it will not leak memory
      // in that case
      ec.indices.push_back (nn_output_namespace);
      v_array<feature> save_nn_output_namespace = ec.atomics[nn_output_namespace];
      ec.atomics[nn_output_namespace] = n.output_layer.atomics[nn_output_namespace];
      ec.sum_feat_sq[nn_output_namespace] = n.output_layer.sum_feat_sq[nn_output_namespace];
      ec.total_sum_feat_sq += n.output_layer.sum_feat_sq[nn_output_namespace];
      if (is_learn)
	base.learn(ec, n.k);
      else
	base.predict(ec, n.k);
      n.output_layer.partial_prediction = ec.partial_prediction;
      n.output_layer.loss = ec.loss;
      ec.total_sum_feat_sq -= n.output_layer.sum_feat_sq[nn_output_namespace];
      ec.sum_feat_sq[nn_output_namespace] = 0;
      ec.atomics[nn_output_namespace] = save_nn_output_namespace;
      ec.indices.pop ();
    }
    else {
      n.output_layer.ft_offset = ec.ft_offset;
      n.output_layer.ld = ec.ld;
      n.output_layer.partial_prediction = 0;
      n.output_layer.eta_round = ec.eta_round;
      n.output_layer.eta_global = ec.eta_global;
      n.output_layer.example_t = ec.example_t;
      if (is_learn)
	base.learn(n.output_layer, n.k);
      else
	base.predict(n.output_layer, n.k);
      n.output_layer.ld = 0;
    }

    n.prediction = GD::finalize_prediction (*(n.all), n.output_layer.partial_prediction);

    if (shouldOutput) {
      outputStringStream << ' ' << n.output_layer.partial_prediction;
      n.all->print_text(n.all->raw_prediction, outputStringStream.str(), ec.tag);
    }

    if (is_learn && n.all->training && ld->label != FLT_MAX) {
      float gradient = n.all->loss->first_derivative(n.all->sd, 
                                                  n.prediction,
                                                  ld->label);

      if (fabs (gradient) > 0) {
        n.all->loss = n.squared_loss;
        n.all->set_minmax = noop_mm;
        save_min_label = n.all->sd->min_label;
        n.all->sd->min_label = hidden_min_activation;
        save_max_label = n.all->sd->max_label;
        n.all->sd->max_label = hidden_max_activation;

        for (unsigned int i = 0; i < n.k; ++i) {
          if (! dropped_out[i]) {
            float sigmah = 
              n.output_layer.atomics[nn_output_namespace][i].x / dropscale;
            float sigmahprime = dropscale * (1.0f - sigmah * sigmah);
            uint32_t nuindex = n.output_layer.atomics[nn_output_namespace][i].weight_index + (n.k * (uint32_t)n.increment) + ec.ft_offset;
            float nu = n.all->reg.weight_vector[nuindex & n.all->reg.weight_mask];
            float gradhw = 0.5f * nu * gradient * sigmahprime;

            ld->label = GD::finalize_prediction (*(n.all), hidden_units[i] - gradhw);
            if (ld->label != hidden_units[i]) 
              base.learn(ec, i);
          }
        }

        n.all->loss = save_loss;
        n.all->set_minmax = save_set_minmax;
        n.all->sd->min_label = save_min_label;
        n.all->sd->max_label = save_max_label;
      }
    }

    ld->label = save_label;

    if (! converse) {
      save_partial_prediction = n.output_layer.partial_prediction;
      save_final_prediction = n.prediction;
      save_ec_loss = n.output_layer.loss;
    }

    if (n.dropout && ! converse)
      {
        for (unsigned int i = 0; i < n.k; ++i)
          {
            dropped_out[i] = ! dropped_out[i];
          }

        converse = true;
        goto CONVERSE;
      }

    ec.partial_prediction = save_partial_prediction;
    ld->prediction = save_final_prediction;
    ec.loss = save_ec_loss;
  }

  void copy_examples(example& ec1, const example& ec2) {
    if (ec2.ld != NULL) { 
      ec1.ld = calloc_or_die(1, sizeof(label_data));
      VW::copy_example_data(false, &ec1, &ec2, sizeof(label_data), NULL);
    }
  }
  
  void add_size_t(size_t& t1, const size_t& t2) {
    t1 += t2;
  }

  void sync_queries(nn& n) {
    /* get pool sizes */
    size_t* pool_sizes = (size_t*) calloc_or_die(n.total, sizeof(size_t));
    for (size_t i = 0; i <= n.total; i++)
      pool_sizes[i] = 0;
    pool_sizes[n.node] = (n.pool_pos - 1);
    all_reduce<size_t, add_size_t>(pool_sizes, n.total, *n.span_server, n.unique_id, n.total, n.node, *n.socks);

    size_t total_examples, offset = 0;
    for (size_t i = 0; i <= n.total; i++) {
      total_examples += pool_sizes[i];
      if (i < n.node)
	offset += pool_sizes[i];
    }
    /* If all sizes are zero we are done */
    if (total_examples == 0) {
      n.pool_pos = 0;
      return;
    }
    else {
      example* train_arr = (example*) calloc_or_die(total_examples, sizeof(example));
      for (int i = 0; i <= (n.pool_pos-1); i++)
	copy_examples(train_arr[offset+i], *(n.pool[i]));
      all_reduce<example, copy_examples>(train_arr, total_examples, *n.span_server, n.unique_id, n.total, n.node, *n.socks);
      
      /* copy all of the examples in n.pool */
      /* TODO is n.pool big enough */
      for (int i = 0; i <= total_examples; i++) {
	n.pool[i] = &(train_arr[i]);
      }
      n.pool_pos = total_examples+1;
    }
  }
  

  void active_bfgs(nn& n, learner& base) {
    BFGS::bfgs* b = (BFGS::bfgs*) base.learn_fd.data;
    BFGS::reset_state(*(n.all), *b, true);
    b->backstep_on = false;
    for (size_t iters = 0; iters < n.active_passes; iters++) {
      for (int i = 0; i < n.pool_pos; i++)
	passive_predict_or_learn<true>(n, base, *(n.pool[i]));

      bool save_quiet = n.all->quiet;
      n.all->quiet = true;
      //b->regularizers = n.all->reg.weight_vector;
      int status = BFGS::process_pass(*(n.all), *b);
      //n.all->l2_lambda += n.active_reg_base; // scale regularizer linearly.
      n.all->quiet = save_quiet;
      if (status != 0) {
	// then we are done iterating.
	//cerr << "Early termination of bfgs updates"<<endl;
	break;
      }
    }
    BFGS::preconditioner_to_regularizer(*(n.all), *b, 0);
    // cerr << endl;
  }

  bool active_update_model(nn& n, learner& base) {
    if (n.para_active)
      sync_queries(n);

    if (n.pool_pos == 0)
      return 1;

    if (n.active_bfgs) {
      time_t start,end;
      time(&start);
      active_bfgs(n, base);
      time(&end);
      double seconds = difftime(end, start);
      n.train_time += seconds;
      //cerr << "BFGS on "<<num_train<<" examples, subsampling rate "<< (float) n.numqueries/(float) n.numexamples<<endl;
      //cerr << "BFGS on "<<num_train<<" examples in " << seconds << " seconds"<< endl;
    }
    else
      for (int i = 0; i < n.pool_pos; i++) {
	passive_predict_or_learn<true>(n, base, *(n.pool[i]));
      }
    return 0;
  }


  template <bool is_learn>
  void active_predict_or_learn(nn& n, learner& base, example& ec)
  {
    n.numexamples++;
    passive_predict_or_learn<false>(n, base, ec);

    if (is_learn) {
      // Then we decide immediately whether to train on this example or not
      // We add it to n.pool and when n.pool is full we update the model.
      float gradient = fabs(n.all->loss->first_derivative(n.all->sd, ec.partial_prediction, ((label_data*)ec.ld)->label));
      ec.loss = n.all->loss->getLoss(n.all->sd, ec.partial_prediction, ((label_data*)ec.ld)->label);
      float queryp = min<float>(gradient, 0.15);
      queryp = max<float>(queryp, min_prob);
      // cerr << "Processing example queryp: "<<queryp<< " rand: " << r <<" pool_pos: "<< n.pool_pos << " pool_size: "<< n.pool_size<<endl;
      if (frand48() < queryp) {
	// cerr << "Adding example to pool"<<endl;
	// Copy the example and add it to the pool.
	example* ec_cpy = (example*) calloc_or_die(1, sizeof(example));
	ec_cpy->ld = calloc_or_die(1, sizeof(label_data));
	VW::copy_example_data(false, ec_cpy, &ec, sizeof(label_data), NULL);
	//float gradient = fabs(n.all->loss->first_derivative(n.all->sd, ec_cpy->partial_prediction, ((label_data*)ec_cpy->ld)->label));
	//ec_cpy->loss = n.all->loss->getLoss(n.all->sd, ec_cpy->partial_prediction, ((label_data*)ec_cpy->ld)->label);
	
	n.pool[n.pool_pos++] = ec_cpy;
	label_data* ld = (label_data*) ec_cpy->ld;
	ld->weight = 1/queryp;
	n.numqueries++;
      }
      if (n.pool_pos == n.pool_size) {
	time_t end;
	time(&end);
	n.subsample_time += difftime(end, n.subsample_start_time);
	active_update_model(n, base);
	for (int i = 0; i < n.pool_pos; i++) {
	  free (n.pool[i]->ld);
	  free (n.pool[i]);
	}
	n.pool_pos = 0;
	time(&(n.subsample_start_time));
      }
    }
  }

  void finish_example(vw& all, nn&, example& ec)
  {
    int save_raw_prediction = all.raw_prediction;
    all.raw_prediction = -1;
    return_simple_example(all, NULL, ec);
    all.raw_prediction = save_raw_prediction;
  }

  void finish(nn& n)
  {
    if (n.para_active) {
      while (!active_update_model(n, *(n.all->l)));
    }
    if (n.active)
      cerr<<"Number of label queries = "<<n.numqueries<<endl;

    delete n.squared_loss;
    free (n.output_layer.indices.begin);
    free (n.output_layer.atomics[nn_output_namespace].begin);
  }

  learner* setup(vw& all, po::variables_map& vm)
  {
    nn* n = (nn*)calloc_or_die(1,sizeof(nn));
    n->all = &all;

    po::options_description nn_opts("NN options");
    nn_opts.add_options()
      ("pool_greedy", "use greedy selection on mini pools")
      ("para_active", "do parallel active learning")
      ("pool_size", po::value<size_t>(), "size of pools for active learning")
      ("subsample", po::value<size_t>(), "number of items to subsample from the pool")
      ("active_bfgs", po::value<size_t>(), "do batch bfgs optimization on active pools")
      ("inpass", "Train or test sigmoidal feedforward network with input passthrough.")
      ("dropout", "Train or test sigmoidal feedforward network using dropout.")
      ("meanfield", "Train or test sigmoidal feedforward network using mean field.");

    vm = add_options(all, nn_opts);

    //first parse for number of hidden units
    n->k = (uint32_t)vm["nn"].as<size_t>();
    
    std::stringstream ss;
    ss << " --nn " << n->k;
    all.file_options.append(ss.str());

    n->active = all.active_simulation;
    if (n->active) {
      if (vm.count("pool_greedy"))
	n->active_pool_greedy = 1;
      if (vm.count("para_active"))
	n->para_active = 1;
      n->numqueries = 0;
      n->numexamples = 0;
      if (n->para_active)
	n->current_t = 0;
    }

    if(n->para_active) {
      n->span_server = new std::string(all.span_server);
      n->total = all.total;
      n->unique_id = all.unique_id;
      n->node = all.node;
      n->socks = new node_socks();
      n->socks->current_master = all.socks.current_master;

      all.span_server = "";
      all.total = 0;      
    }

    if (vm.count("active_bfgs")) {
      n->active_bfgs = 1;
      n->active_passes = vm["active_bfgs"].as<std::size_t>();
    }

    if (vm.count("pool_size"))
      n->pool_size = vm["pool_size"].as<std::size_t>();
    else
      n->pool_size = 1;

    if (n->para_active)
      n->pool = (example**)calloc_or_die(n->pool_size*n->total, sizeof(example*));
    else
      n->pool = (example**)calloc_or_die(n->pool_size, sizeof(example*));
    n->pool_pos = 0;

    if (vm.count("subsample"))
      n->subsample = vm["subsample"].as<std::size_t>();
    else if (n->para_active)
      n->subsample = ceil(n->pool_size / n->total);
    else
      n->subsample = 1;
    // cerr<<"Subsample = "<<n->subsample<<endl;
      
      

    if ( vm.count("dropout") ) {
      n->dropout = true;
      
      std::stringstream ss;
      ss << " --dropout ";
      all.file_options.append(ss.str());
    }
    
    if ( vm.count("meanfield") ) {
      n->dropout = false;
      if (! all.quiet) 
        std::cerr << "using mean field for neural network " 
                  << (all.training ? "training" : "testing") 
                  << std::endl;
    }

    if (n->dropout) 
      if (! all.quiet)
        std::cerr << "using dropout for neural network "
                  << (all.training ? "training" : "testing") 
                  << std::endl;

    if (vm.count ("inpass")) {
      n->inpass = true;

      std::stringstream ss;
      ss << " --inpass";
      all.file_options.append(ss.str());
    }

    if (n->inpass && ! all.quiet)
      std::cerr << "using input passthrough for neural network "
                << (all.training ? "training" : "testing") 
                << std::endl;

    n->finished_setup = false;
    n->squared_loss = getLossFunction (0, "squared", 0);

    n->xsubi = 0;

    if (vm.count("random_seed"))
      n->xsubi = vm["random_seed"].as<size_t>();

    n->save_xsubi = n->xsubi;
    n->increment = all.l->increment;//Indexing of output layer is odd.

    if (all.l2_lambda != 0)
      n->active_reg_base = all.l2_lambda;
    all.l2_lambda = 0;

    // diagnostics
    n->train_time = 0;
    n->subsample_time = 0;
    time(&(n->subsample_start_time));

    learner* l = new learner(n, all.l, n->k+1);
    if (n->active) {
      l->set_learn<nn, active_predict_or_learn<true> >();
      l->set_predict<nn, active_predict_or_learn<false> >();
    } else {
      l->set_learn<nn, passive_predict_or_learn<true> >();
      l->set_predict<nn, passive_predict_or_learn<false> >();
    }
    l->set_finish<nn, finish>();
    l->set_finish_example<nn, finish_example>();
    l->set_end_pass<nn,end_pass>();
    return l;
  }
}
