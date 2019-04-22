#include <resnet.h>
#include <mobilenet.h>

using namespace std;
using namespace model;

int main(){
    auto t = torch::zeros({1, 3, 50, 50});
    auto mn = mobilenet::MobileNetV2();
    //cout << resnet(t) << endl;
    for(const auto&p : mn->named_parameters()){
        cout << p.key() << endl;
    }
    return 0;
}