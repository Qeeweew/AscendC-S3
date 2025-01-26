#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace matmul;

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// x1 (M, K) x2 (K, N) x3 (M, N) y(M, N)
template<typename FLOAT_T, uint32_t TILE_M, uint32_t TILE_K, uint32_t TILE_N>
class KernelMulMatSub {
public:
    __aicore__ inline KernelMulMatSub() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, uint32_t M, uint32_t K, uint32_t N, bool BROAD_CAST_X3)
    {
        this->M = M;
        this->K = K;
        this->N = N;
        this->BROAD_CAST_X3 = BROAD_CAST_X3;
        x1Gm.SetGlobalBuffer((__gm__ FLOAT_T *)x1, M * K * sizeof(FLOAT_T));
        x2Gm.SetGlobalBuffer((__gm__ FLOAT_T *)x2, K * N * sizeof(FLOAT_T));
        x3Gm.SetGlobalBuffer((__gm__ FLOAT_T *)x3, M * N * sizeof(FLOAT_T));
        yGm.SetGlobalBuffer((__gm__ FLOAT_T *)y, M * N * sizeof(FLOAT_T));

        pipe.InitBuffer(inQueueX1, 2, TILE_M * TILE_K * sizeof(FLOAT_T));
        pipe.InitBuffer(inQueueX2, 1, TILE_K * TILE_N * sizeof(FLOAT_T));
        pipe.InitBuffer(inQueueX3, 1, TILE_M * TILE_N * sizeof(FLOAT_T));
        pipe.InitBuffer(outQueueY, 1, TILE_M * TILE_N * sizeof(FLOAT_T));
        if constexpr(std::is_same<FLOAT_T, half>::value) {
            pipe.InitBuffer(calcBuf, TILE_M * TILE_N * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr(std::is_same<FLOAT_T, half>::value) {
            ProcessHalf();
        } else {
            ProcessFloat();
        }
    }

    __aicore__ inline void ProcessFloat()
    {
        int idx = 0;
        for (uint32_t i = 0;i < M;i += TILE_M) {
            uint32_t len_m = MIN(TILE_M, M - i);
            for (uint32_t j = 0;j < N;j += TILE_N, idx++) {
                if (idx % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) continue;
                uint32_t len_n = MIN(TILE_N, N - j);
                auto acc = outQueueY.AllocTensor<FLOAT_T>();
                AscendC::Duplicate(acc, (FLOAT_T) 0.0f, TILE_M * TILE_N);
                for (uint32_t l = 0;l < K;l += TILE_K) {
                    uint32_t len_k = MIN(TILE_K, K - l);
                    CopyInX1(i, l, len_m, len_k);
                    CopyInX2(l, j, len_k, len_n);
                    Compute(acc, len_m, len_k, len_n);
                }
                if (BROAD_CAST_X3) {
                    CopyInX3BroadCast(j, len_n);
                    auto acc0 = inQueueX3.DeQue<FLOAT_T>();
                    for (int j = 0;j < TILE_M;j++) {
                        AscendC::Sub(acc[j * TILE_N], acc[j * TILE_N], acc0, TILE_N);
                    }
                    inQueueX3.FreeTensor(acc0);
                } else {
                    CopyInX3(i, j, len_m, len_n);
                    auto acc0 = inQueueX3.DeQue<FLOAT_T>();
                    AscendC::Sub(acc, acc, acc0, TILE_M * TILE_N);
                    inQueueX3.FreeTensor(acc0);
                }
                outQueueY.EnQue(acc);
                CopyOut(i, j, len_m, len_n);
            }
        }
    }

    __aicore__ inline void ProcessHalf()
    {
        int idx = 0;
        for (uint32_t i = 0;i < M;i += TILE_M) {
            uint32_t len_m = MIN(TILE_M, M - i);
            for (uint32_t j = 0;j < N;j += TILE_N, idx++) {
                if (idx % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) continue;
                uint32_t len_n = MIN(TILE_N, N - j);
                AscendC::LocalTensor<float> acc = calcBuf.Get<float>();
                AscendC::LocalTensor<half> acc_out = outQueueY.AllocTensor<half>();
                AscendC::Duplicate(acc, 0.0f, TILE_M * TILE_N);
                for (uint32_t l = 0;l < K;l += TILE_K) {
                    uint32_t len_k = MIN(TILE_K, K - l);
                    CopyInX1(i, l, len_m, len_k);
                    CopyInX2(l, j, len_k, len_n);
                    Compute(acc, len_m, len_k, len_n);
                }
                AscendC::Cast(acc_out, acc, AscendC::RoundMode::CAST_ROUND, TILE_M * TILE_N);
                if (BROAD_CAST_X3) {
                    CopyInX3BroadCast(j, len_n);
                    auto acc0 = inQueueX3.DeQue<half>();
                    for (int j = 0;j < TILE_M;j++) {
                        AscendC::Sub(acc_out[j * TILE_N], acc_out[j * TILE_N], acc0, TILE_N);
                    }
                    inQueueX3.FreeTensor(acc0);
                } else {
                    CopyInX3(i, j, len_m, len_n);
                    auto acc0 = inQueueX3.DeQue<half>();
                    AscendC::Sub(acc_out, acc_out, acc0, TILE_M * TILE_N);
                    inQueueX3.FreeTensor(acc0);
                }
                outQueueY.EnQue(acc_out);
                CopyOut(i, j, len_m, len_n);
            }
        }
    }

private:
    __aicore__ inline void CopyInX1(uint32_t i, uint32_t j, uint32_t rows, uint32_t cols)
    {
        AscendC::LocalTensor<FLOAT_T> xLocal = inQueueX1.AllocTensor<FLOAT_T>();
        AscendC::DataCopyParams copyParamsX;
        copyParamsX.blockCount = rows;
        copyParamsX.blockLen = cols * sizeof(FLOAT_T);
        copyParamsX.srcStride = (K - cols) * sizeof(FLOAT_T);
        copyParamsX.dstStride = ((TILE_K - cols) / (32 / sizeof(FLOAT_T)));
        AscendC::DataCopyPadParams padParams{false, 0, 0, 0};

        AscendC::DataCopyPad(xLocal, x1Gm[i * K + j], copyParamsX, padParams);
        inQueueX1.EnQue(xLocal);
    }

    __aicore__ inline void CopyInX2(uint32_t i, uint32_t j, uint32_t rows, uint32_t cols)
    {
        AscendC::LocalTensor<FLOAT_T> xLocal = inQueueX2.AllocTensor<FLOAT_T>();
        AscendC::DataCopyParams copyParamsX;
        copyParamsX.blockCount = rows;
        copyParamsX.blockLen = cols * sizeof(FLOAT_T);
        copyParamsX.srcStride = (N - cols) * sizeof(FLOAT_T);
        copyParamsX.dstStride = ((TILE_N - cols) / (32 / sizeof(FLOAT_T)));
        AscendC::DataCopyPadParams padParams{false, 0, 0, 0};

        AscendC::DataCopyPad(xLocal, x2Gm[i * N + j], copyParamsX, padParams);
        inQueueX2.EnQue(xLocal);
    }

    __aicore__ inline void CopyInX3BroadCast(uint32_t j, uint32_t cols)
    {
        AscendC::LocalTensor<FLOAT_T> xLocal = inQueueX3.AllocTensor<FLOAT_T>();
        AscendC::DataCopyParams copyParamsX;
        copyParamsX.blockCount = 1;
        copyParamsX.blockLen = cols * sizeof(FLOAT_T);
        AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, x3Gm[j], copyParamsX, padParams);
        inQueueX3.EnQue(xLocal);
    }

    __aicore__ inline void CopyInX3(uint32_t i, uint32_t j, uint32_t rows, uint32_t cols)
    {
        AscendC::LocalTensor<FLOAT_T> xLocal = inQueueX3.AllocTensor<FLOAT_T>();
        AscendC::DataCopyParams copyParamsX;
        copyParamsX.blockCount = rows;
        copyParamsX.blockLen = cols * sizeof(FLOAT_T);
        copyParamsX.srcStride = (N - cols) * sizeof(FLOAT_T);
        copyParamsX.dstStride = ((TILE_N - cols) / (32 / sizeof(FLOAT_T)));
        AscendC::DataCopyPadParams padParams{false, 0, 0, 0};

        AscendC::DataCopyPad(xLocal, x3Gm[i * N + j], copyParamsX, padParams);
        inQueueX3.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t i, uint32_t j, uint32_t rows, uint32_t cols)
    {
        AscendC::LocalTensor<FLOAT_T> xLocal = outQueueY.DeQue<FLOAT_T>();
        AscendC::DataCopyParams copyParamsX;
        copyParamsX.blockCount = rows;
        copyParamsX.blockLen = cols * sizeof(FLOAT_T);
        copyParamsX.srcStride = ((TILE_N - cols) / (32 / sizeof(FLOAT_T)));
        copyParamsX.dstStride = (N - cols) * sizeof(FLOAT_T);
        AscendC::DataCopyPad(yGm[i * N + j], xLocal, copyParamsX);
        outQueueY.FreeTensor(xLocal);
    }
   
    __aicore__ inline void Compute(const AscendC::LocalTensor<float>& acc, uint32_t m, uint32_t k, uint32_t n)
    {
        AscendC::LocalTensor<FLOAT_T> x1Local = inQueueX1.DeQue<FLOAT_T>();
        AscendC::LocalTensor<FLOAT_T> x2Local = inQueueX2.DeQue<FLOAT_T>();
        for (int l = 0; l < k;l++) {
            for (int i = 0;i < m;i++) {
                AscendC::Axpy(acc[i * TILE_N], x2Local[l * TILE_N], x1Local.GetValue(i * TILE_K + l), TILE_N);
            }
        }
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX1, inQueueX2, inQueueX3;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;

    AscendC::GlobalTensor<FLOAT_T> x1Gm, x2Gm, x3Gm;
    AscendC::GlobalTensor<FLOAT_T> yGm;

    uint32_t M, K, N;
    bool BROAD_CAST_X3;

};

template <typename T>
struct TileSize;

template <>
struct TileSize<float> {
    static constexpr int value = 1024;
};

template <>
struct TileSize<half> {
    static constexpr int value = 1024;
};


__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

// c = a @ b - d 
template <typename FLOAT_T> class MatmulKernel {
public:
    __aicore__ inline MatmulKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d, GM_ADDR workspace, bool use_broadcast,
                                const TCubeTiling &tiling);
    __aicore__ inline void Process0();
    __aicore__ inline void Process1(AscendC::TPipe *pipe);
    
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC, int32_t &offsetD);
    


    __aicore__ inline void CopyInC(uint32_t i, uint32_t j, uint32_t rows, uint32_t cols);
    __aicore__ inline void CopyInD(uint32_t i, uint32_t j, uint32_t rows, uint32_t cols);

    __aicore__ inline void CopyOut(uint32_t i, uint32_t j, uint32_t rows, uint32_t cols);

    __aicore__ inline void ComputeSub();

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, FLOAT_T>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, FLOAT_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>, CFG_MDL>
        matmulObj;

    AscendC::GlobalTensor<FLOAT_T> aGlobal;
    AscendC::GlobalTensor<FLOAT_T> bGlobal;
    AscendC::GlobalTensor<FLOAT_T> cGlobal;
    AscendC::GlobalTensor<float> cF32Global;
    AscendC::GlobalTensor<FLOAT_T> dGlobal;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueC0;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueD;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueC;

    TCubeTiling tiling;

    int singleCoreM;
    int singleCoreN;
    bool use_broadcast;
};

template <typename FLOAT_T>
__aicore__ inline void MatmulKernel<FLOAT_T>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d, GM_ADDR workspace, bool use_broadcast,
                                                                        const TCubeTiling &tiling)
{
    this->use_broadcast = use_broadcast;
    this->tiling = tiling;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ FLOAT_T *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ FLOAT_T *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ FLOAT_T *>(c), tiling.M * tiling.N);
    cF32Global.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace), tiling.M * tiling.N);
    dGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ FLOAT_T *>(d), tiling.M * tiling.N);

    int32_t offsetA = 0;
    int32_t offsetB = 0;
    int32_t offsetC = 0;
    int32_t offsetD = 0;
    CalcOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetD);
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    cF32Global = cF32Global[offsetC];
    dGlobal = dGlobal[offsetD];
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }

}

template <typename FLOAT_T>
__aicore__ inline void MatmulKernel<FLOAT_T>::Process0()
{
    if (singleCoreM <= 0 || singleCoreN <= 0) return;

    // AscendC::TBuf<> tmpMMFormatUb;
    // pipe->InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
    // AscendC::LocalTensor<uint8_t> mmformatUb;
    // mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
    // matmulObj.SetLocalWorkspace(mmformatUb);

    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.IterateAll(cF32Global);
    matmulObj.End();
}

template <typename FLOAT_T>
__aicore__ inline void MatmulKernel<FLOAT_T>::Process1(AscendC::TPipe *pipe)
{
    if (singleCoreM <= 0 || singleCoreN <= 0) return;

    pipe->InitBuffer(inQueueC0, 1, 64 * 64 * sizeof(float));
    pipe->InitBuffer(inQueueD, 1, 64 * 64 * sizeof(FLOAT_T));
    pipe->InitBuffer(outQueueC, 1, 64 * 64 * sizeof(FLOAT_T));
        
    for (int i = 0; i < singleCoreM; i += 64) {
        for (int j = 0; j < singleCoreN; j += 64) {
            CopyInC(i, j, 64, 64);
            CopyInD(i, j, 64, 64);
            ComputeSub();
            CopyOut(i, j, 64, 64);
        }
    }
}

template <typename FLOAT_T>
__aicore__ inline void MatmulKernel<FLOAT_T>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA,
                                                        int32_t &offsetB, int32_t &offsetC, int32_t &offsetD)
{
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;

    if (use_broadcast) {
        offsetD = nCoreIndx * tiling.singleCoreN;
    } else {
        offsetD = offsetC;
    }

    if (mCoreIndx * tiling.singleCoreM >= tiling.M || nCoreIndx * tiling.singleCoreN >= tiling.N) {
        singleCoreM = 0;
        singleCoreN = 0;
        return;
    }

    int tailM = MIN(tiling.M - mCoreIndx * tiling.singleCoreM, tiling.singleCoreM);
    int tailN = MIN(tiling.N - nCoreIndx * tiling.singleCoreN, tiling.singleCoreN); 
    if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN) {
        matmulObj.SetTail(tailM, tailN);
        singleCoreM = tailM;
        singleCoreN = tailN;
    } else {
        singleCoreM = tiling.singleCoreM;
        singleCoreN = tiling.singleCoreN;
    }
}

template <typename FLOAT_T>
__aicore__ inline void MatmulKernel<FLOAT_T>::CopyInD(uint32_t i, uint32_t j, uint32_t rows, uint32_t cols)
{
    AscendC::LocalTensor<FLOAT_T> dLocal = inQueueD.AllocTensor<FLOAT_T>();
    if (use_broadcast) {
        AscendC::DataCopyParams copyParamsX;
        copyParamsX.blockCount = 1;
        copyParamsX.blockLen = cols * sizeof(FLOAT_T);
        AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(dLocal, dGlobal[j], copyParamsX, padParams);
    } else {
        AscendC::DataCopyParams copyParamsX;
        copyParamsX.blockCount = rows;
        copyParamsX.blockLen = cols * sizeof(FLOAT_T);
        copyParamsX.srcStride = (tiling.N - cols) * sizeof(FLOAT_T);
        AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(dLocal, dGlobal[i * tiling.N + j], copyParamsX, padParams);
    }
    inQueueD.EnQue(dLocal);
}

template <typename FLOAT_T>
__aicore__ inline void MatmulKernel<FLOAT_T>::CopyInC(uint32_t i, uint32_t j, uint32_t rows, uint32_t cols)
{
    AscendC::LocalTensor<float> c0Local = inQueueC0.AllocTensor<float>();
    AscendC::DataCopyParams copyParamsX;
    copyParamsX.blockCount = rows;
    copyParamsX.blockLen = cols * sizeof(float);
    copyParamsX.srcStride = (tiling.N - cols) * sizeof(float);
    AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(c0Local, cF32Global[i * tiling.N + j], copyParamsX, padParams);
    // matmulObj.template Iterate<true>();
    // matmulObj.template GetTensorC<true>(c0Local, 0, true);

    inQueueC0.EnQue(c0Local);
}

template <typename FLOAT_T>
__aicore__ inline void MatmulKernel<FLOAT_T>::CopyOut(uint32_t i, uint32_t j, uint32_t rows, uint32_t cols)
{
    AscendC::LocalTensor<FLOAT_T> cLocal = outQueueC.DeQue<FLOAT_T>();
    AscendC::DataCopyParams copyParamsX;
    copyParamsX.blockCount = rows;
    copyParamsX.blockLen = cols * sizeof(FLOAT_T);
    copyParamsX.dstStride = (tiling.N - cols) * sizeof(FLOAT_T);
    AscendC::DataCopyPad(cGlobal[i * tiling.N + j], cLocal, copyParamsX);
    outQueueC.FreeTensor(cLocal);
}

template <typename FLOAT_T>
__aicore__ inline void MatmulKernel<FLOAT_T>::ComputeSub()
{
    AscendC::LocalTensor<float> c0Local = inQueueC0.DeQue<float>();
    AscendC::LocalTensor<FLOAT_T> dLocal = inQueueD.DeQue<FLOAT_T>();
    AscendC::LocalTensor<FLOAT_T> cLocal = outQueueC.AllocTensor<FLOAT_T>();

    if constexpr(std::is_same<half, FLOAT_T>::value) {
        AscendC::Cast(cLocal, c0Local, AscendC::RoundMode::CAST_ROUND, 64 * 64);
        if (use_broadcast) {
            for (int i = 0; i < 64;i++) {
                AscendC::Sub(cLocal[i * 64], cLocal[i * 64], dLocal, 64);
            }
        } else {
            AscendC::Sub(cLocal, cLocal, dLocal, 64 * 64);
        }
    } else {
        if (use_broadcast) {
            for (int i = 0; i < 64;i++) {
                AscendC::Sub(cLocal[i * 64], c0Local[i * 64], dLocal, 64);
            }
        } else {
            AscendC::Sub(cLocal, c0Local, dLocal, 64 * 64);
        }
    }
    outQueueC.EnQue(cLocal);
    inQueueD.FreeTensor(dLocal);
    inQueueC0.FreeTensor(c0Local);
}


extern "C" __global__ __aicore__ void mat_mul_sub(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 设置默认的kernel类型为纯AIV类型
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIV_ONLY);
        constexpr int TILE_N = TileSize<DTYPE_X1>::value;
        KernelMulMatSub<DTYPE_X1, 32 / sizeof(DTYPE_X1), 32 / sizeof(DTYPE_X1), TILE_N> op;
        op.Init(x1, x2, x3, y, tilingData.M, tilingData.K, tilingData.N, tilingData.use_broadcast);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        MatmulKernel<DTYPE_X1> matmulKernel;
        AscendC::TPipe pipe_matmul;
        REGIST_MATMUL_OBJ(&pipe_matmul, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tilingData.cubeTilingData);
        matmulKernel.Init(x1, x2, y, x3, workspace, tilingData.use_broadcast, tilingData.cubeTilingData);
        pipe_matmul.Destroy();
        AscendC::TPipe pipe_sub;
        matmulKernel.Process0();
        matmulKernel.Process1(&pipe_sub);
    }
}