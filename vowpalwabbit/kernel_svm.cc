/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
  individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */
#include <float.h>
#ifdef _WIN32
#include <WinSock2.h>
#else
#include <netdb.h>
#endif
#include <string.h>
#include <stdio.h>
#include <assert.h>

#if defined(__SSE2__) && !defined(VW_LDA_NO_SSE)
#include <xmmintrin.h>
#endif

#include "reductions.h"
#include "constant.h"
  // #include "sparse_dense.h"
#include "gd.h"
#include "kernel_svm.h"
#include "cache.h"
#include "simple_label.h"
#include "vw.h"
#include "rand48.h"
#include <map>



using namespace std;
using namespace LEARNER;

namespace KSVM
{

  inline float sign(float w){ if (w < 0.) return -1; else return 1.;}

  struct svm_params{
    size_t current_pass;
    bool active;
    bool active_pool_greedy;
    bool para_active;

    size_t pool_size;
    size_t pool_pos;
    size_t subsample; //NOTE: Eliminating subsample to only support 1/pool_size
    size_t reprocess;

    svm_model* model;
    size_t maxcache;

    svm_example** pool;
    double lambda;

    void* kernel_params;
    size_t kernel_type;

    size_t local_begin, local_end;
    size_t current_t;

    vw* all;
  };

  static int num_kernel_evals = 0;
  static int num_cache_evals = 0;

  svm_example::svm_example(VW::flat_example *fec)
  {
    *(VW::flat_example*)this = *fec;
    memset(fec, 0, sizeof(flat_example));
  }
  
  svm_example::~svm_example()
  {
    krow.delete_v();
    // free flatten example contents
    VW::flat_example *fec = (VW::flat_example*)malloc(sizeof(VW::flat_example));
    *fec = *(VW::flat_example*)this;
    VW::free_flatten_example(fec); // free contents of flat example and frees fec.
  }

  float kernel_function(const VW::flat_example* fec1, const VW::flat_example* fec2, void* params, size_t kernel_type);

  int
  svm_example::compute_kernels(svm_params *params)
  {
    int alloc = 0;
    svm_model *model = params->model;
    size_t n = model->num_support;

    if (krow.size() < n)
      {
	// compute new kernel values and caching them
	num_kernel_evals += krow.size();
	for (size_t i=krow.size(); i<n; i++) 
	  {
	    svm_example *sec = model->support_vec[i];
	    float kv = kernel_function(this, sec, params->kernel_params, params->kernel_type);
	    krow.push_back(kv);
	    alloc += 1;
	  }
      }
    else
      num_cache_evals += n;
    return alloc;
  }

  int
  svm_example::clear_kernels()
  {
    int rowsize = krow.size();
    krow.end = krow.begin;
    krow.resize(0);
    return -rowsize;
  }

  /**
   * This is the LRU caching algorithm. Upon request for svi, we bring
   * it to the front of the cache and rotate everything back by one.
   */
  static int
  make_hot_sv(svm_params *svm, int svi)
  {
    svm_model *model = svm->model;
    int n = model->num_support;
    if ((size_t) svi >= model->num_support)
      cerr << "Internal error at " << __FILE__ << ":" << __LINE__ << endl;
    // rotate params fields
    svm_example *svi_e = model->support_vec[svi];
    int alloc = svi_e->compute_kernels(svm);
    float svi_alpha = model->alpha[svi];
    float svi_delta = model->delta[svi];
    for (int i=svi; i>0; --i)
      {
	model->support_vec[i] = model->support_vec[i-1];
	model->alpha[i] = model->alpha[i-1];
	model->delta[i] = model->delta[i-1];
      }
    model->support_vec[0] = svi_e;
    model->alpha[0] = svi_alpha;
    model->delta[0] = svi_delta;
    // rotate cache
    for (int j=0; j<n; j++)
      {
	svm_example *e = model->support_vec[j];
	int rowsize = e->krow.size();
	if (svi < rowsize)
	  {
	    float kv = e->krow[svi];
	    for (int i=svi; i>0; --i)
	      e->krow[i] = e->krow[i-1];
	    e->krow[0] = kv;
	  }
	else
	  {
	    float kv = svi_e->krow[j];
	    e->krow.push_back(0);
	    alloc += 1;
	    for (int i=e->krow.size()-1; i>0; --i)
	      e->krow[i] = e->krow[i-1];
	    e->krow[0] = kv;
	  }
      }
    return alloc;
  }

  static int
  trim_cache(svm_params *params)
  {
    size_t sz = params->maxcache;
    svm_model *model = params->model;
    int n = model->num_support;
    int alloc = 0;
    for (int i=0; i<n; i++)
      {
        svm_example *e = model->support_vec[i];
        sz -= e->krow.size();
        if (sz < 0)
          alloc += e->clear_kernels();
      }
    return alloc;
  }

  float linear_kernel(const VW::flat_example* fec1, const VW::flat_example* fec2) {
    float dotprod = 0;
    feature* ec2f = fec2->feature_map;
    uint32_t ec2pos = ec2f->weight_index;
    uint32_t idx1 = 0, idx2 = 0;

    int numint = 0;
    for (feature* f = fec1->feature_map; idx1 < fec1->feature_map_len && idx2 < fec2->feature_map_len; f++, idx1++) {
      uint32_t ec1pos = f->weight_index;
      if (ec1pos < ec2pos) continue;

      while (ec1pos > ec2pos && idx2 < fec2->feature_map_len) {
	ec2f++;
	idx2++;
	if (idx2 < fec2->feature_map_len)
	  ec2pos = ec2f->weight_index;
      }
      if (ec1pos == ec2pos) {
	numint++;
	dotprod += f->x*ec2f->x;
	ec2f++;
	idx2++;
	if (idx2 < fec2->feature_map_len)
	  ec2pos = ec2f->weight_index;
      }
    }
    return dotprod;
  }

  float poly_kernel(const VW::flat_example* fec1, const VW::flat_example* fec2, int power) {
    float dotprod = linear_kernel(fec1, fec2);
    return pow(1 + dotprod, power);
  }

  float rbf_kernel(const VW::flat_example* fec1, const VW::flat_example* fec2, double bandwidth) {
    float dotprod = linear_kernel(fec1, fec2);
    return exp(-(fec1->total_sum_feat_sq + fec2->total_sum_feat_sq - 2*dotprod)*bandwidth);
  }

  float kernel_function(const VW::flat_example* fec1, const VW::flat_example* fec2, void* params, size_t kernel_type) {
    switch(kernel_type) {
    case SVM_KER_RBF:
      return rbf_kernel(fec1, fec2, *((double*) params));
    case SVM_KER_LIN:
      return linear_kernel(fec1, fec2);
    }
    return 0;
  }

  float dense_dot(float* v1, v_array<float> v2, size_t n) {
    float dot_prod = 0;
    for(size_t i = 0; i < n; i++) {
      dot_prod += v1[i]*v2[i];
    }
    return dot_prod;
  }

  void predict(svm_params* svm, svm_example** ec_arr, double* scores, size_t n) {
    svm_model* model = svm->model;
    for(size_t i = 0; i < n; i++) {
      ec_arr[i]->compute_kernels(svm);
      scores[i] = dense_dot(ec_arr[i]->krow.begin, model->alpha, model->num_support)/svm->lambda;
    }
  }

  size_t suboptimality(svm_model* model, double* subopt) {
    int max_pos = 0;
    double max_val = 0;
    for(size_t i = 0; i < model->num_support; i++) {
      label_data* ld = (label_data*)(model->support_vec[i]->ld);
      double tmp = model->alpha[i]*ld->label;

      if ((tmp < ld->weight && model->delta[i] < 0) || (tmp > 0 && model->delta[i] > 0))
	subopt[i] = fabs(model->delta[i]);
      else
	subopt[i] = 0;

      if (subopt[i] > max_val) {
	max_val = subopt[i];
	max_pos = i;
      }
    }
    return max_pos;
  }

  int remove(svm_params* svm, int svi) {
    svm_model* model = svm->model;
    if ((size_t) svi >= model->num_support)
      cerr << "Internal error at " << __FILE__ << ":" << __LINE__ << endl;
    svm_example* svi_e = model->support_vec[svi];
    for (size_t i=svi; i<model->num_support-1; ++i)
      {
	model->support_vec[i] = model->support_vec[i+1];
	model->alpha[i] = model->alpha[i+1];
	model->delta[i] = model->delta[i+1];
      }
    delete svi_e;
    model->support_vec.pop();
    model->alpha.pop();
    model->delta.pop();
    model->num_support--;

    int alloc = 0;
    for (size_t j=0; j<model->num_support; j++)
      {
	svm_example *e = model->support_vec[j];
	int rowsize = e->krow.size();
	if (svi < rowsize)
	  {
	    for (int i=svi; i<rowsize-1; i++)
	      e->krow[i] = e->krow[i+1];
	    e->krow.pop();
	    alloc -= 1;
	  }
      }
    return alloc;
  }

  int add(svm_params* svm, svm_example* fec) {
    svm_model* model = svm->model;
    model->num_support++;
    model->support_vec.push_back(fec);
    model->alpha.push_back(0.);
    model->delta.push_back(0.);
    return (model->support_vec.size()-1);
  }

  bool update(svm_params* svm, int pos) {

    //cerr<<"Update\n";
    svm_model* model = svm->model;
    bool overshoot = false;
    svm_example* fec = model->support_vec[pos];
    label_data* ld = (label_data*) fec->ld;
    fec->compute_kernels(svm);
    float *inprods = fec->krow.begin;
    double alphaKi = dense_dot(inprods, model->alpha, model->num_support);
    model->delta[pos] = alphaKi*ld->label/svm->lambda - 1;
    double alpha_old = model->alpha[pos];
    alphaKi -= model->alpha[pos]*inprods[pos];
    model->alpha[pos] = 0.;

    double proj = alphaKi*ld->label;
    double ai = (svm->lambda - proj)/inprods[pos];


    if(ai > ld->weight)
      ai = ld->weight;
    else if(ai < 0)
      ai = 0;

    ai *= ld->label;
    double diff = ai - alpha_old;

    if(fabs(diff) > 1.0e-06)
      overshoot = true;

    if(fabs(diff) > 1.) {
      diff = sign(diff);
      ai = alpha_old + diff;
    }

    for(size_t i = 0;i < model->num_support; i++) {
      label_data* ldi = (label_data*) model->support_vec[i]->ld;
      model->delta[i] += diff*inprods[i]*ldi->label/svm->lambda;
    }

    if(fabs(ai) <= 1.0e-10)
      remove(svm, pos);
    else 
      model->alpha[pos] = ai;

    return overshoot;
  }

  
  void train(svm_params* svm) {
    bool* train_pool = (bool*)calloc_or_die(svm->pool_size, sizeof(bool));
    for (size_t i = 0; i < svm->pool_size; i++) 
      train_pool[i] = false;

    double* scores = (double*)calloc_or_die(svm->pool_pos, sizeof(double));
    // TODO: this can be optimized since we should already have the predictions
    predict(svm, svm->pool, scores, svm->pool_pos);
    if (svm->active) {
      if (svm->active_pool_greedy){
	multimap<double, int> scoremap;
	for (size_t i = 0; i < svm->pool_pos; i++)
	  scoremap.insert(pair<const double, const int>(fabs(scores[i]), i));

	multimap<double, int>::iterator iter = scoremap.begin();
	for (size_t train_size = 1; iter != scoremap.end() && train_size <= svm->subsample; train_size++) {
	  train_pool[iter->second] = 1;
	  iter++;
	}
      } else {
	for (size_t i = 0; i < svm->pool_pos; i++) {
	  double queryp = 2.0/(1.0 + exp(svm->all->active_c0*fabs(scores[i])*pow(svm->pool[i]->example_counter, 0.5)));
	  if (frand48() < queryp) {
	    svm_example* fec = svm->pool[i];
	    label_data* ld = (label_data*)fec->ld;
	    ld->weight *= 1/queryp;
	    train_pool[i] = 1;
	  }
	}
      }
    }

    if (svm->para_active) {
      // TODO: fill me in
    }

    svm_model* model = svm->model;
    for (size_t i = 0; i < svm->pool_size; i++) {
      int model_pos = -1;
      if (svm->active) {
	if (train_pool[i]) {
	  model_pos = add(svm, svm->pool[i]);
	}
      } else
	model_pos = add(svm, svm->pool[i]);

      if (model_pos >= 0) {
	bool overshoot = update(svm, model_pos);

	double* subopt = (double*)calloc_or_die(model->num_support, sizeof(double));
	for(size_t j = 0; j < svm->reprocess; j++) {
	  if (model->num_support == 0) break;
	  int randi = 1;
	  if (randi) {
	    size_t max_pos = suboptimality(model, subopt);
	    if (subopt[max_pos] > 0) {
	      if(!overshoot && max_pos == (size_t) model_pos && max_pos > 0 && j == 0)
		cerr<<"Shouldn't reprocess right after process!!!\n";
	      //cerr<<max_pos<<" "<<subopt[max_pos]<<endl;
	      // cerr<<params->model->support_vec[0]->example_counter<<endl;
	      if(max_pos*model->num_support <= svm->maxcache)
		make_hot_sv(svm, max_pos);
	      update(svm, max_pos);
	    }
	  } else {
	    size_t rand_pos = rand()%model->num_support;
	    update(svm, rand_pos);
	  }
	}
	free(subopt);
      }
    }
    free (scores);
    free (train_pool);
  }

  template <bool is_learn>
  void predict_or_learn(svm_params& svm, learner& base, example& ec)
  {
    VW::flat_example* fec = VW::flatten_example(*(svm.all), &ec);
    if (fec) {
      svm_example* sec = new svm_example(fec);
      VW::free_flatten_example(fec);
      double score = 0;
      predict(&svm, &sec, &score, 1);
      ec.partial_prediction = score;
      float label = ((label_data*)ec.ld)->label;
      ec.loss = max(0., 1-score*label);
      if (is_learn) {
	if (ec.example_counter % 100 == 0)
	  trim_cache(&svm);
	if (ec.example_counter % 1000 == 0) {
	  cerr<<"Number of support vectors = "<<svm.model->num_support<<endl;
	  cerr<<"Number of kernel evaluations = "<<num_kernel_evals<<" "<<"Number of cachqueries = "<<num_cache_evals<<endl;
	}
	svm.pool[svm.pool_pos] = sec;
	svm.pool_pos++;
	if(svm.pool_pos == svm.pool_size) {
	  train(&svm);
	  svm.pool_pos = 0;
	}
      }
    }
  }

  void finish_example(vw& all, svm_params&, example& ec)
  {
    int save_raw_prediction = all.raw_prediction;
    all.raw_prediction = -1;
    return_simple_example(all, NULL, ec);
    all.raw_prediction = save_raw_prediction;
  }

  void free_svm_model(svm_model* model)
  {
    for(size_t i = 0;i < model->num_support; i++) {
      delete model->support_vec[i];
      model->support_vec[i] = 0;
    }

    model->support_vec.delete_v();
    model->alpha.delete_v();
    model->delta.delete_v();
    free(model);
  }


  void finish(svm_params& params) {
    free(params.pool);

    cerr<<"Num support = "<<params.model->num_support<<endl;
    cerr<<"Number of kernel evaluations = "<<num_kernel_evals<<" "<<"Number of cache queries = "<<num_cache_evals<<endl;
    //double maxalpha = fabs(params->model->alpha[0]);
    //size_t maxpos = 0;

    // for(size_t i = 1;i < params->model->num_support; i++)
    //   if(maxalpha < fabs(params->model->alpha[i])) {
    //  maxalpha = fabs(params->model->alpha[i]);
    //  maxpos = i;
    //   }

    //cerr<<maxalpha<<" "<<maxpos<<endl;

    //cerr<<"Done freeing pool\n";

    free_svm_model(params.model);
    cerr<<"Done freeing model\n";
    if(params.kernel_params) free(params.kernel_params);
    cerr<<"Done freeing kernel params\n";
    //free(&params);
    //cerr<<"Done with finish \n";
  }

  learner* setup(vw& all, po::variables_map& vm)
  {
    svm_params* params = (svm_params*) calloc_or_die(1,sizeof(svm_params));
    cerr<<"In setup"<<endl;

    params->model = (svm_model*) calloc_or_die(1,sizeof(svm_model));
    params->model->num_support = 0;
    params->maxcache = 1024*1024*1024;

    po::options_description svm_opts("KSVM options");
    svm_opts.add_options()
      ("reprocess", po::value<size_t>(), "number of reprocess steps for LASVM")
      ("pool_greedy", "use greedy selection on mini pools")
      ("para_active", "do parallel active learning")
      ("pool_size", po::value<size_t>(), "size of pools for active learning")
      ("subsample", po::value<size_t>(), "number of items to subsample from the pool")
      ("kernel", po::value<string>(), "type of kernel (rbf or linear (default))")
      ("bandwidth", po::value<double>(), "bandwidth of rbf kernel")
      ("degree", po::value<int>(), "degree of poly kernel")
      ("lambda", po::value<double>(), "saving regularization for test time");
      
    vm = add_options(all, svm_opts);
    
    params->all = &all;

    if (vm.count("reprocess"))
      params->reprocess = vm["reprocess"].as<std::size_t>();
    else
      params->reprocess = 1;

    params->active = all.active_simulation;
    if (params->active) {
      if (vm.count("pool_greedy"))
	params->active_pool_greedy = 1;
      else
      if (vm.count("para_active"))
	params->para_active = 1;
    }
    
    if (vm.count("pool_size"))
      params->pool_size = vm["pool_size"].as<std::size_t>();
    else
      params->pool_size = 1;

    params->pool = (svm_example**) calloc_or_die(params->pool_size, sizeof(svm_example*));
    params->pool_pos = 0;

    if (vm.count("subsample"))
      params->subsample = vm["subsample"].as<std::size_t>();
    else if(params->para_active)
      params->subsample = ceil(params->pool_size / all.total);
    else
      params->subsample = 1;
    
    params->lambda = all.l2_lambda;
    
    if (vm.count("lambda"))
      params->lambda = vm["lambda"].as<double>();
    cerr<<"Lambda = "<<params->lambda<<endl;
    
    std::string kernel_type;
    if (vm.count("kernel"))
      kernel_type = vm["kernel"].as<std::string>();
    else 
      kernel_type = string("linear");
    cerr << "Kernel = "<< kernel_type<<endl;

    if (kernel_type.compare("rbf") == 0) {
      params->kernel_type = SVM_KER_RBF;
      double bandwidth = 1.;
      if (vm.count("bandwidth"))
	bandwidth = vm["bandwidth"].as<double>();
      cerr<<"bandwidth = "<<bandwidth<<endl;
      params->kernel_params = calloc_or_die(1,sizeof(double*));
      *((double*)params->kernel_params) = bandwidth;
    } else if (kernel_type.compare("poly") == 0) {
      params->kernel_type = SVM_KER_POLY;
      int degree = 2;
      if (vm.count("degree"))
	degree = vm["degree"].as<int>();
      cerr<<"degree = "<<degree<<endl;
      params->kernel_params = calloc_or_die(1,sizeof(double*));
      *((double*)params->kernel_params) = degree;
    } else
      params->kernel_type = SVM_KER_LIN;

    learner* l = new learner(params, all.l); /* TODO: check how many weight vectors we need */
    l->set_learn<svm_params, predict_or_learn<true> >();
    l->set_predict<svm_params, predict_or_learn<false> >();
    l->set_finish<svm_params, finish>();
    l->set_finish_example<svm_params, finish_example>();
    return l;
  }
}
