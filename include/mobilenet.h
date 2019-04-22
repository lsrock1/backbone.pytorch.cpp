#pragma once
#include <torch/torch.h>


namespace model{
namespace mobilenet{
  class MobileNetV2Impl : public torch::nn::Module{
      using ExpandRatio = int64_t;
      using Channel = int64_t;
      using NumBlocks = int64_t;
      using Stride = int64_t;

      public:
        MobileNetV2Impl(int64_t num_classes = 1000, double width_mult = 1.0);
        torch::Tensor forward(torch::Tensor x);
    
      private:
        torch::nn::Sequential features_{nullptr};
        torch::nn::Linear classifier_{nullptr};
        void initialize();
  };

  TORCH_MODULE(MobileNetV2);

  class InvertedResidualImpl : public torch::nn::Module{
      public:
        InvertedResidualImpl(int64_t in_planes, int64_t out_planes, int64_t stride, int64_t expand_ratio);
        torch::Tensor forward(torch::Tensor x);

      private:
        bool use_res_connection_;
        torch::nn::Sequential body_{nullptr};
  };

  TORCH_MODULE(InvertedResidual);

  class ConvBNReLUImpl : public torch::nn::Module{
      public:
        ConvBNReLUImpl(int64_t in_planes, int64_t out_planes, int64_t kernel_size=3, int64_t stride=1, int64_t groups=1);
        torch::Tensor forward(torch::Tensor x);
    
      private:
        torch::nn::Conv2d conv_;
        torch::nn::BatchNorm bn_;
  };
  
  TORCH_MODULE(ConvBNReLU);
}//mobilenet
}//model