#include "../exercise.h"
#include <cstring>
// READ: 类模板 <https://zh.cppreference.com/w/cpp/language/class_template>

template<class T>
struct Tensor4D {
    unsigned int shape[4];
    T *data;

    Tensor4D(unsigned int const shape_[4], T const *data_) {
        unsigned int size = 1;
        // TODO: 填入正确的 shape 并计算 size
        for(int i=0;i<4;i++){
            shape[i]=shape_[i];
            size*=shape[i];
        }
        data = new T[size];
        std::memcpy(data, data_, size * sizeof(T));
    }
    ~Tensor4D() {
        delete[] data;
    }

    // 为了保持简单，禁止复制和移动
    Tensor4D(Tensor4D const &) = delete;
    Tensor4D(Tensor4D &&) noexcept = delete;

    // 这个加法需要支持“单向广播”。
    // 具体来说，`others` 可以具有与 `this` 不同的形状，形状不同的维度长度必须为 1。
    // `others` 长度为 1 但 `this` 长度不为 1 的维度将发生广播计算。
    // 例如，`this` 形状为 `[1, 2, 3, 4]`，`others` 形状为 `[1, 2, 1, 4]`，
    // 则 `this` 与 `others` 相加时，3 个形状为 `[1, 2, 1, 4]` 的子张量各自与 `others` 对应项相加。
    Tensor4D &operator+=(Tensor4D const &others) {
        // TODO: 实现单向广播的加法
        unsigned int size = 1;
        for (int i = 0; i < 4; ++i) size *= shape[i];

        for (unsigned int i = 0; i < size; ++i) {
            
            // --- A. 把一维索引 i 还原成四维坐标 (n, c, h, w) ---
            // 这种计算方法叫“取模除法法”，就像把秒转换成“小时:分钟:秒”
            unsigned int idx = i;
            unsigned int coords[4]; // 存放 this 的当前坐标
            
            // 我们从最后一个维度（最内层）开始算
            for (int dim = 3; dim >= 0; --dim) {
                coords[dim] = idx % shape[dim]; // 取余数得到当前维度的坐标
                idx /= shape[dim];              // 除法进入下一层
            }

            // --- B. 找到 others 对应的坐标 ---
            unsigned int others_idx = 0;
            // 这是一个累积乘数，用来把坐标变回一维索引
            unsigned int stride = 1; 

            // 我们从最后一个维度往回算，直接算出 others 的一维索引 j
            for (int dim = 3; dim >= 0; --dim) {
                // 核心广播逻辑：
                // 如果 others 在这个维度长度是 1，那坐标就是 0；
                // 否则，坐标就必须和 this 的坐标（coords[dim]）一样。
                unsigned int coord_in_others = (others.shape[dim] == 1) ? 0 : coords[dim];
                
                // 累加索引： 坐标 * 步长
                others_idx += coord_in_others * stride;
                
                // 更新步长
                stride *= others.shape[dim];
            }

            // --- C. 加法 ---
            data[i] += others.data[others_idx];
        }
        return *this;
    }
};

// ---- 不要修改以下代码 ----
int main(int argc, char **argv) {
    {
        unsigned int shape[]{1, 2, 3, 4};
        // clang-format off
        int data[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        auto t0 = Tensor4D(shape, data);
        auto t1 = Tensor4D(shape, data);
        t0 += t1;
        for (auto i = 0u; i < sizeof(data) / sizeof(*data); ++i) {
            ASSERT(t0.data[i] == data[i] * 2, "Tensor doubled by plus its self.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        float d0[]{
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,

            4, 4, 4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6};
        // clang-format on
        unsigned int s1[]{1, 2, 3, 1};
        // clang-format off
        float d1[]{
            6,
            5,
            4,

            3,
            2,
            1};
        // clang-format on

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == 7.f, "Every element of t0 should be 7 after adding t1 to it.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        double d0[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        unsigned int s1[]{1, 1, 1, 1};
        double d1[]{1};

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == d0[i] + 1, "Every element of t0 should be incremented by 1 after adding t1 to it.");
        }
    }
}
