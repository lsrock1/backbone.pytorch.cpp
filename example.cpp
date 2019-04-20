#include <resnet.h>

using namespace std;
using namespace model::resnet;

int main(){
    auto t = torch::zeros({1, 3, 50, 50});
    auto resnet = ResNet18();
    //cout << resnet(t) << endl;
    for(const auto&p : resnet->named_parameters()){
        cout << p.key() << endl;
    }
    return 0;
}