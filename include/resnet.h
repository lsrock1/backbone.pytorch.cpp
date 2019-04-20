#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>

using namespace std;

namespace model{
    namespace resnet{
        // ResNet ResNet34(/*pretrained*/ int num_classes=1000, bool zero_init_residual=false);
        // ResNet ResNet50(/*pretrained*/ int num_classes=1000, bool zero_init_residual=false);
        // ResNet ResNet101(/*pretrained*/ int num_classes=1000, bool zero_init_residual=false);
        // ResNet ResNet152(/*pretrained*/ int num_classes=1000, bool zero_init_residual=false);
        // ResNet ResNext50_32x4d(/*pretrained*/ int num_classes=1000, bool zero_init_residual=false);
        // ResNet ResNext101_32x8d(/*pretrained*/ int num_classes=1000, bool zero_init_residual=false);
        class ResNetImpl : public torch::nn::Module{
            public:
                ResNetImpl(const char* block, vector<int64_t> layers, int num_classes=1000, bool zero_init_residual=false, int64_t groups=1, int64_t width_per_groups=64);
                ResNetImpl(const char* block, std::initializer_list<int64_t> layers, int num_classes=1000, bool zero_init_residual=false, int64_t groups=1, int64_t width_per_groups=64);
                torch::Tensor forward(torch::Tensor x);
            
            torch::nn::Sequential MakeLayer(int64_t planes, int64_t blocks, int64_t stride=1);
            void initialize(bool zero_init_residual, string block_name);

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
            int64_t stride_;
            int64_t width_;

            public:
                BottleneckImpl(int64_t in_planes, int64_t out_planes, torch::nn::Sequential downsample, int64_t stride=1, int64_t groups=1, int64_t base_width=64);
                BottleneckImpl(int64_t in_planes, int64_t out_planes, int64_t stride=1, int64_t groups=1, int64_t base_width=64/*c++ frontend only has batch norm*/);
                torch::Tensor forward(torch::Tensor x);
                const static int64_t kExpansion = 4;

            torch::nn::Conv2d conv1_, conv2_, conv3_;
            torch::nn::BatchNorm bn1_, bn2_, bn3_;
            torch::nn::Sequential downsample_{nullptr};
        };

        TORCH_MODULE(Bottleneck);

        torch::nn::Conv2d Conv3x3(int64_t in_planes, int64_t out_planes, int64_t stride=1, int64_t groups=1);
        torch::nn::Conv2d Conv1x1(int64_t in_planes, int64_t out_planes, int64_t stride=1);

        class ResNet18Impl : public ResNetImpl{
            public:
                ResNet18Impl(int num_classes=1000, bool zero_init_residual=false): ResNetImpl("BasicBlock", {2, 2, 2, 2}, num_classes, zero_init_residual){};
        };

        TORCH_MODULE(ResNet18);

        class ResNet34Impl : public ResNetImpl{
            public:
                ResNet34Impl(int num_classes=1000, bool zero_init_residual=false): ResNetImpl("BasicBlock", {3, 4, 6, 3}, num_classes, zero_init_residual){};
        };

        TORCH_MODULE(ResNet34);

        class ResNet50Impl : public ResNetImpl{
            public:
                ResNet50Impl(int num_classes=1000, bool zero_init_residual=false): ResNetImpl("Bottleneck", {3, 4, 6, 3}, num_classes, zero_init_residual){};
        };

        TORCH_MODULE(ResNet50);

        class ResNet101Impl : public ResNetImpl{
            public:
                ResNet101Impl(int num_classes=1000, bool zero_init_residual=false): ResNetImpl("Bottleneck", {3, 4, 23, 3}, num_classes, zero_init_residual){};
        };

        TORCH_MODULE(ResNet101);

        class ResNet152Impl : public ResNetImpl{
            public:
                ResNet152Impl(int num_classes=1000, bool zero_init_residual=false): ResNetImpl("Bottleneck", {3, 8, 36, 3}, num_classes, zero_init_residual){};
        };

        TORCH_MODULE(ResNet152);

        class ResNext50_32x4dImpl : public ResNetImpl{
            public:
                ResNext50_32x4dImpl(int num_classes=1000, bool zero_init_residual=false): ResNetImpl("Bottleneck", {3, 4, 6, 3}, num_classes, zero_init_residual, 32, 4){};
        };

        TORCH_MODULE(ResNext50_32x4d);

        class ResNext101_32x8dImpl : public ResNetImpl{
            public:
                ResNext101_32x8dImpl(int num_classes=1000, bool zero_init_residual=false): ResNetImpl("Bottleneck", {3, 4, 23, 3}, num_classes, zero_init_residual, 32, 8){};
        };

        TORCH_MODULE(ResNext101_32x8d);
    }
}//namespace model