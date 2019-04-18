#include <resnet.h>

using namespace std;

int main(){
    vector<int64_t> vec{3, 4, 6, 3};
    auto t = torch::zeros({1, 3, 50, 50});
    auto resnet = model::resnet::ResNet("Bottleneck", vec);
    cout << resnet(t) << endl;
    for(const auto&p : resnet->named_parameters()){
        cout << p.key() << endl;
    }
    return 0;
}