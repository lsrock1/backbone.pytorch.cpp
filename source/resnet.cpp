#include <resnet.h>

using namespace model::resnet;

namespace model{
    namespace resnet{
ResNetImpl::ResNetImpl(const char* block,
            vector<int64_t> layers, 
            int num_classes, 
            bool zero_init_residual, 
            int64_t groups, 
            int64_t width_per_group)
            : groups_(groups),
              base_width_(width_per_group),
              block_(block),
              expansion_(block_.compare("Bottleneck") == 0 ? BottleneckImpl::kExpansion : BasicBlockImpl::kExpansion),
              conv1_(register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, in_planes_, 7).stride(2).padding(3).with_bias(false)))),
              bn1_(register_module("bn1", torch::nn::BatchNorm(in_planes_))),
              layer1_(register_module("layer1", MakeLayer(64, layers.at(0)))),
              layer2_(register_module("layer2", MakeLayer(128, layers.at(1), 2))),
              layer3_(register_module("layer3", MakeLayer(256, layers.at(2), 2))),
              layer4_(register_module("layer4", MakeLayer(512, layers.at(3), 2))),
              fc_(register_module("fc", torch::nn::Linear(512 * expansion_, num_classes))){
                  initialize();
              }

torch::Tensor ResNetImpl::forward(torch::Tensor x){
    x = bn1_->forward(conv1_->forward(x)).relu_();
    x = torch::max_pool2d(x, 3, 2, 1);
    x = layer1_->forward(x);
    x = layer2_->forward(x);
    x = layer3_->forward(x);
    x = layer4_->forward(x);
    x = torch::adaptive_avg_pool2d(x, {1, 1});
    x = x.view({x.size(0), -1});
    return fc_->forward(x);
}

void ResNetImpl::initialize(){
    for(auto &module : modules(false)){
        if (module->name().find("Conv2d") != std::string::npos) {
            for (auto& p : module->named_parameters()) {
                if (p.key().find("weight") != std::string::npos) {
                torch::nn::init::xavier_uniform_(p.value());
                } else if (p.key().find("bias") != std::string::npos) {
                torch::nn::init::zeros_(p.value());
                }
            }
        } 
        else if (module->name().find("BatchNorm2d") != std::string::npos) {
            for (auto& p : module->named_parameters()) {
                if (p.key().find("weight") != std::string::npos) {
                torch::nn::init::ones_(p.value());
                }
                if (p.key().find("bias") != std::string::npos) {
                torch::nn::init::zeros_(p.value());
                }
            }
        }
        else if (module->name().find("Linear") != std::string::npos) {
            for (auto& p : module->named_parameters()) {
                if (p.key().find("weight") != std::string::npos) {
                torch::nn::init::normal_(p.value(), 0, 0.01);
                }
                if (p.key().find("bias") != std::string::npos) {
                torch::nn::init::zeros_(p.value());
                }
            }
        }
    }
}

torch::nn::Sequential ResNetImpl::MakeLayer(int64_t planes, int64_t blocks, int64_t stride){
    torch::nn::Sequential downsample{nullptr};
    if(stride != 1 || in_planes_ != planes * expansion_){
        downsample = torch::nn::Sequential(
            Conv1x1(in_planes_, planes * expansion_, stride),
            torch::nn::BatchNorm(planes * expansion_)
        );
    }
    torch::nn::Sequential layers;
    if(block_.compare("Bottleneck") == 0){
        if(downsample){
            layers->push_back(Bottleneck(in_planes_, planes, downsample, stride, groups_, base_width_));
        }
        else{
            layers->push_back(Bottleneck(in_planes_, planes, stride, groups_, base_width_));
        }
    }
    else{
        if(downsample){
            layers->push_back(BasicBlock(in_planes_, planes, downsample, stride, groups_, base_width_));
        }
        else{
            layers->push_back(BasicBlock(in_planes_, planes, stride, groups_, base_width_));
        }
    }
    in_planes_ = planes * expansion_;
    for(int i = 1; i < blocks; ++i){
        if(block_.compare("Bottleneck") == 0){
            layers->push_back(Bottleneck(in_planes_, planes, 1, groups_, base_width_));
        }
        else{
            layers->push_back(BasicBlock(in_planes_, planes, 1, groups_, base_width_));
        }
    }
    return layers;
}

BottleneckImpl::BottleneckImpl(int64_t in_planes, 
                            int64_t out_planes, 
                            torch::nn::Sequential downsample, 
                            int64_t stride, 
                            int64_t groups, 
                            int64_t base_width)
                            : width_(out_planes * (base_width/64.) * groups),
                              conv1_(register_module("conv1", Conv1x1(in_planes, out_planes * (base_width/64.) * groups))),
                              bn1_(register_module("bn1", torch::nn::BatchNorm(out_planes * (base_width/64.) * groups))),
                              conv2_(register_module("conv2", Conv3x3(out_planes * (base_width/64.) * groups, out_planes * (base_width/64.) * groups, stride, groups))),
                              bn2_(register_module("bn2", torch::nn::BatchNorm(out_planes * (base_width/64.) * groups))),
                              conv3_(register_module("conv3", Conv1x1(out_planes * (base_width/64.) * groups, out_planes * BottleneckImpl::kExpansion))),
                              bn3_(register_module("bn3", torch::nn::BatchNorm(out_planes * BottleneckImpl::kExpansion))),
                              downsample_(register_module("downsample", downsample)),
                              stride_(stride){};

BottleneckImpl::BottleneckImpl(int64_t in_planes, 
                            int64_t out_planes, 
                            int64_t stride, 
                            int64_t groups, 
                            int64_t base_width)
                            : width_(out_planes * (base_width/64.) * groups),
                              conv1_(register_module("conv1", Conv1x1(in_planes, out_planes * (base_width/64.) * groups))),
                              bn1_(register_module("bn1", torch::nn::BatchNorm(out_planes * (base_width/64.) * groups))),
                              conv2_(register_module("conv2", Conv3x3(out_planes * (base_width/64.) * groups, out_planes * (base_width/64.) * groups, stride, groups))),
                              bn2_(register_module("bn2", torch::nn::BatchNorm(out_planes * (base_width/64.) * groups))),
                              conv3_(register_module("conv3", Conv1x1(out_planes * (base_width/64.) * groups, out_planes * BottleneckImpl::kExpansion))),
                              bn3_(register_module("bn3", torch::nn::BatchNorm(out_planes * BottleneckImpl::kExpansion))),
                              stride_(stride){};

torch::Tensor BottleneckImpl::forward(torch::Tensor x){
    torch::Tensor identity;
    if(downsample_){
        identity = downsample_->forward(x);
    }
    else{
        identity = x;
    }
    x = bn1_->forward(conv1_->forward(x)).relu_();
    x = bn2_->forward(conv2_->forward(x)).relu_();
    x = bn3_->forward(conv3_->forward(x));
    x += identity;
    return x.relu_();
}

BasicBlockImpl::BasicBlockImpl(int64_t in_planes,
                            int64_t out_planes, 
                            torch::nn::Sequential downsample, 
                            int64_t stride, 
                            int64_t groups, 
                            int64_t base_width/*c++ frontend only has batch norm*/)
                            : conv1_(register_module("conv1", Conv3x3(in_planes, out_planes, stride))),
                              bn1_(register_module("bn1", torch::nn::BatchNorm(out_planes))),
                              conv2_(register_module("conv2", Conv3x3(out_planes, out_planes))),
                              bn2_(register_module("bn2", torch::nn::BatchNorm(out_planes))),
                              downsample_(register_module("downsample", downsample)),
                              stride_(stride)
                              {};

BasicBlockImpl::BasicBlockImpl(int64_t in_planes,
                            int64_t out_planes,
                            int64_t stride, 
                            int64_t groups, 
                            int64_t base_width/*c++ frontend only has batch norm*/)
                            : conv1_(register_module("conv1", Conv3x3(in_planes, out_planes, stride))),
                              bn1_(register_module("bn1", torch::nn::BatchNorm(out_planes))),
                              conv2_(register_module("conv2", Conv3x3(out_planes, out_planes))),
                              bn2_(register_module("bn2", torch::nn::BatchNorm(out_planes))),
                              stride_(stride)
                              {};

torch::Tensor BasicBlockImpl::forward(torch::Tensor x){
    torch::Tensor identity;
    if(downsample_){
        identity = downsample_->forward(x);
    }
    else{
        identity = x;
    }
    x = bn1_->forward(conv1_->forward(x)).relu_();
    x = bn2_->forward(conv2_->forward(x));
    x += identity;
    return x.relu_();
}

torch::nn::Conv2d Conv3x3(int64_t in_planes, int64_t out_planes, int64_t stride, int64_t groups){
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, out_planes, 3)
                                        .stride(stride)
                                        .padding(1)
                                        .groups(groups)
                                        .with_bias(false));
}

torch::nn::Conv2d Conv1x1(int64_t in_planes, int64_t out_planes, int64_t stride){
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, out_planes, 1)
                                        .stride(stride)
                                        .with_bias(false));
}
}
}