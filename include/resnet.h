#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>

using namespace std;

namespace model{
    namespace resnet{
        class ResNetImpl : public torch::nn::Module{
            public:
                ResNetImpl(const char* block, vector<int64_t> layers, int num_classes=1000, bool zero_init_residual=false, int64_t groups=1, int64_t width_per_groups=64);
                torch::Tensor forward(torch::Tensor x);
            
            torch::nn::Sequential MakeLayer(int64_t planes, int64_t blocks, int64_t stride=1);
            void initialize();

            string block_;
            int64_t in_planes_ = 64;
            int64_t groups_;
            int64_t base_width_;
            int64_t expansion_;
            torch::nn::Conv2d conv1_;
            torch::nn::BatchNorm bn1_;
            torch::nn::Sequential layer1_{nullptr}, layer2_{nullptr}, layer3_{nullptr}, layer4_{nullptr};
            torch::nn::Linear fc_;
        };

        TORCH_MODULE(ResNet);

        class BasicBlockImpl : public torch::nn::Module{
            
            public:
                BasicBlockImpl(int64_t in_planes, int64_t out_planes, torch::nn::Sequential downsample, int64_t stride=1, int64_t groups=1, int64_t base_width=64/*c++ frontend only has batch norm*/);
                BasicBlockImpl(int64_t in_planes, int64_t out_planes, int64_t stride=1, int64_t groups=1, int64_t base_width=64/*c++ frontend only has batch norm*/);
                torch::Tensor forward(torch::Tensor x);
                const static int64_t kExpansion = 1;
            
            torch::nn::Conv2d conv1_, conv2_;
            torch::nn::BatchNorm bn1_, bn2_;
            torch::nn::Sequential downsample_{nullptr};

            int64_t width_;
            int64_t stride_;
        };

        TORCH_MODULE(BasicBlock);

        class BottleneckImpl : public torch::nn::Module{
            
            public:
                BottleneckImpl(int64_t in_planes, int64_t out_planes, torch::nn::Sequential downsample, int64_t stride=1, int64_t groups=1, int64_t base_width=64);
                BottleneckImpl(int64_t in_planes, int64_t out_planes, int64_t stride=1, int64_t groups=1, int64_t base_width=64/*c++ frontend only has batch norm*/);
                torch::Tensor forward(torch::Tensor x);
                const static int64_t kExpansion = 4;

            torch::nn::Conv2d conv1_, conv2_, conv3_;
            torch::nn::BatchNorm bn1_, bn2_, bn3_;
            torch::nn::Sequential downsample_{nullptr};

            int64_t stride_;
            int64_t width_;
        };

        TORCH_MODULE(Bottleneck);

        torch::nn::Conv2d Conv3x3(int64_t in_planes, int64_t out_planes, int64_t stride=1, int64_t groups=1);
        torch::nn::Conv2d Conv1x1(int64_t in_planes, int64_t out_planes, int64_t stride=1);
    }
}//namespace model