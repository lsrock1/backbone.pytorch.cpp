#include <resnet.h>
#include <utils.h>

void train();
//for this train code
//change resnet.cpp
//conv1_ in planes 3 to 1
int main(int argc, char **argv){
    train();
}

void train(){
    const int kNumberOfEpochs = 10;
    const int batch_size = 4;
    //only one gpu
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    // torch::cuda::device_count();
    auto resnet = model::resnet::ResNet18(10);
    resnet->to(device);
    auto dataset = torch::data::datasets::MNIST("/root/mnist")
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());

    auto data_loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
    
    torch::optim::Adam optimizer(resnet->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
        int64_t batch_index = 0;
        for (torch::data::Example<>& batch : *data_loader) {
            
            resnet->zero_grad();
            auto output = resnet(batch.data);
            torch::Tensor target = torch::zeros({batch_size, 10}, torch::kFloat32);
            
            auto loss = torch::nll_loss(output.log_softmax(1), batch.target);
            loss.backward();
            optimizer.step();

            std::printf(
            "\r[%2ld/%2ld][%3ld] D_loss: %.4f ",
            epoch,
            kNumberOfEpochs,
            ++batch_index,
            loss.item<float>());
        }
    }
}