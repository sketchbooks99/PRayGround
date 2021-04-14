#include <iostream>

struct Hoge {
    Hoge(int w, int h, float c) {
        init();
        construct(w, h, c);
    }
    void init() {
        if (data) {
            delete[] data;
        }
    }
    void construct(int w, int h, float c) {
        data = new float[w*h];
        for (int x=0; x<w; x++) {
            for (int y=0; y<h; y++) {
                int idx = x*h+y;
                data[idx] = c;
            }
        }
    }
    float* data { nullptr };
};

int main() {
    Hoge hoge(10, 10, 1.0f);
    hoge.init();
    hoge.construct(10, 10, 0.5f);
    return 0;
}