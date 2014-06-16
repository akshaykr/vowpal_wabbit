/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD
license as described in the file LICENSE.
 */
#ifndef KERNEL_SVM_H
#define KERNEL_SVM_H

#ifdef _WIN32
#include <WinSock2.h>
#else
#include <netdb.h>
#endif

#define SVM_KER_LIN 0
#define SVM_KER_RBF 1
#define SVM_KER_POLY 2

#include "global_data.h"
#include "parse_args.h"

namespace KSVM
{
  struct svm_params;
  struct svm_example : public VW::flat_example {
    v_array<float> krow;

    ~svm_example();
    svm_example(VW::flat_example *fec);
    int compute_kernels(svm_params *params);
    int clear_kernels();
  };

  struct svm_model{
    size_t num_support;
    v_array<svm_example*> support_vec;
    v_array<float> alpha;
    v_array<float> delta;
  };

  void free_svm_model(svm_model*);

  LEARNER::learner* setup(vw& all, po::variables_map& vm);
}

#endif
