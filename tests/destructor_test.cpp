#include <iostream>
#include <vector>

struct Hoge {
    struct Handle {
        int a { 0 };
        float b { 0.0f };
    };

    Handle handle_a;
    Handle handle_b;

    ~Hoge() {
        if (handle_a.a) handle_a.a = 0;
        if (handle_b.a) handle_b.b = 0;

        std::cout << "Called" << std::endl;
    }
};

void construct_hoge(Hoge* hoge) {
    hoge->handle_a = { rand(), (float)rand() / RAND_MAX};
    hoge->handle_b = { rand(), (float)rand() / RAND_MAX};
}

int main() {
    std::vector<Hoge*> hoges;
    for (int i=0; i<10; i++) {
        hoges.emplace_back(new Hoge());
        std::cout << "emplace_backed" << std::endl;
        construct_hoge(hoges.back());
        std::cout << "constructed" << std::endl;
    }

    for (auto& hoge : hoges) {
        std::cout << "handle_a: " << hoge->handle_a.a << ' ' << hoge->handle_a.b << std::endl;
        std::cout << "handle_b: " << hoge->handle_b.a << ' ' << hoge->handle_b.b << std::endl;
    }

    return 0;
}