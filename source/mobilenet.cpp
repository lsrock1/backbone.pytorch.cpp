#include <mobilenet.h>
#include <cmath>

using namespace std;

namespace model{
namespace mobilenet{
  MobileNetV2Impl::MobileNetV2Impl(int64_t num_classes, double width_mult){
      int64_t input_channel = 32;
      int64_t last_channel = 1280;
      torch::nn::Sequential body;

      vector<tuple<ExpandRatio, Channel, NumBlocks, Stride>> inverted_residual_setting{
        make_tuple(1, 16, 1, 1),
        make_tuple(6, 24, 2, 2),
        make_tuple(6, 32, 3, 2),
        make_tuple(6, 64, 4, 2),
        make_tuple(6, 96, 3, 1),
        make_tuple(6, 160, 3, 2),
        make_tuple(6, 320, 1, 1)
      };

      input_channel = static_cast<int64_t> (input_channel * width_mult);
      last_channel = static_cast<int64_t> (last_channel * max(1.0, width_mult));
      body->push_back(ConvBNReLU(3, input_channel, 3, 2));
      int64_t output_channel;
      int64_t stride;
      for(auto i = inverted_residual_setting.begin(); i != inverted_residual_setting.end(); ++i){
          output_channel = static_cast<int> (get<1>(*i) * width_mult);
          for(int l = 0; l < get<2>(*i); ++l){
            stride = l == 0 ? get<3>(*i) :  1;
            body->push_back(InvertedResidual(input_channel, output_channel, stride, get<0>(*i)));
            input_channel = output_channel;
          }
      }
      body->push_back(ConvBNReLU(input_channel, last_channel, 1));
      features_ = register_module("features", body);
      classifier_ = register_module("classifier", torch::nn::Linear(last_channel, num_classes));
      initialize();
  };

  torch::Tensor MobileNetV2Impl::forward(torch::Tensor x){
    x = features_->forward(x);
    x = x.mean({2, 3});
    x = torch::dropout(x, /*p=*/0.5, is_training());
    return x = classifier_->forward(x);
  }

  void MobileNetV2Impl::initialize(){
    for(auto &param : this->named_parameters()){
      if(param.key().find("conv") != string::npos){
        if(param.key().find("weight") != string::npos){
          torch::nn::init::xavier_uniform_(param.value());
        }
        else if(param.key().find("bias") != string::npos){
          torch::nn::init::zeros_(param.value());
        }
      }
      else if(param.key().find("bn") != string::npos){
        if(param.key().find("weight") != string::npos){
          torch::nn::init::ones_(param.value());
        }
        else if(param.key().find("bias") != string::npos){
          torch::nn::init::zeros_(param.value());
        }
      }
      else if(param.key().find("classifier") != string::npos){
        if(param.key().find("weight") != string::npos){
          torch::nn::init::normal_(param.value(), 0, 0.01);
        }
        else if(param.key().find("bias") != string::npos){
          torch::nn::init::zeros_(param.value());
        }
      }
    }
  }

  InvertedResidualImpl::InvertedResidualImpl(int64_t in_planes, int64_t out_planes, int64_t stride, int64_t expand_ratio){
      int64_t hidden_dim = static_cast<int> (round(in_planes * expand_ratio));
      use_res_connection_ = stride == 1 && in_planes == out_planes;
      torch::nn::Sequential body;
      if(expand_ratio != 1){
        body->push_back(ConvBNReLU(in_planes, hidden_dim, /*kernel_size=*/1));
      }
      body->push_back(ConvBNReLU(hidden_dim, hidden_dim, /*kernel_size=*/3, /*stride=*/stride, /*groups=*/hidden_dim));
      body->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_dim, out_planes, 1).stride(1).padding(0).with_bias(false)));
      body->push_back(torch::nn::BatchNorm(out_planes));
      body_ = register_module("inverted_res", body);
  }

  torch::Tensor InvertedResidualImpl::forward(torch::Tensor x){
    if(use_res_connection_){
        return x + body_->forward(x);
    }  
    else{
        return body_->forward(x);
    }
  }
  
  ConvBNReLUImpl::ConvBNReLUImpl(int64_t in_planes, 
                                int64_t out_planes, 
                                int64_t kernel_size, 
                                int64_t stride, 
                                int64_t groups)
                                : conv_(register_module("conv", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(in_planes, out_planes, kernel_size)
                                                                .stride(stride)
                                                                .groups(groups)
                                                                .padding((kernel_size-1)/2)
                                                                .with_bias(false)))),
                                  bn_(register_module("bn", torch::nn::BatchNorm(out_planes)))
                                {};
  
  torch::Tensor ConvBNReLUImpl::forward(torch::Tensor x){
    return bn_->forward(conv_->forward(x)).relu_().clamp_max_(6);
  }
}//mobilenet
}//model