#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"

#define CONTINUS_TILING 0
#define NONCONTINUS_TILING 1

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))
#include<math.h>

// continuous
template<typename FLOAT_T>
class KernelArgMaxWithValue {
public:
    static constexpr int TILE_V = 32 / sizeof(FLOAT_T);
    static constexpr int TILE_X = 4096 / sizeof(FLOAT_T);
    __aicore__ inline KernelArgMaxWithValue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, uint32_t reduce_len, uint32_t sizeV) {
        // AscendC::printf("reduce_len = %d tile_x = %d\n", reduce_len, TILE_X);
        this->reduce_len = reduce_len;
        this->sizeV = sizeV;
        this->blockNum = AscendC::GetBlockNum();

        xGm.SetGlobalBuffer((__gm__ FLOAT_T *)x, reduce_len * sizeV);
        iGm.SetGlobalBuffer((__gm__ int32_t *)indice, sizeV);
        vGm.SetGlobalBuffer((__gm__ FLOAT_T *)values, sizeV);
        pipe.InitBuffer(inQueueX, 2, TILE_X * sizeof(FLOAT_T));
        pipe.InitBuffer(outQueueI, 1, TILE_V * sizeof(int32_t));
        pipe.InitBuffer(outQueueV, 1, TILE_V * sizeof(FLOAT_T));
        pipe.InitBuffer(calcBuf0, 32);
        pipe.InitBuffer(calcBuf1, TILE_X * sizeof(FLOAT_T));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = AscendC::GetBlockIdx() * TILE_V;i < sizeV;i+= blockNum * TILE_V) {
            AscendC::LocalTensor<FLOAT_T> valueLocal = outQueueV.AllocTensor<FLOAT_T>();
            AscendC::LocalTensor<int32_t> indiceLocal = outQueueI.AllocTensor<int32_t>();
            uint32_t len_v = TILE_V < sizeV - i ? TILE_V: sizeV - i;
            for (uint32_t j = 0;j < len_v;j++) {
                for (int k = 0;k < reduce_len;k+=TILE_X) {
                    uint32_t len_x = TILE_X < reduce_len - k ? TILE_X : reduce_len - k;
                    CopyIn(i + j, k, len_x);
                    Reduce(j, valueLocal, indiceLocal, k, len_x);
                }
            }
            outQueueV.EnQue<FLOAT_T>(valueLocal);
            outQueueI.EnQue<int32_t>(indiceLocal);
            CopyOut(i, len_v);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t i, uint32_t j, uint32_t len_x)
    {
        AscendC::LocalTensor<FLOAT_T> xLocal = inQueueX.AllocTensor<FLOAT_T>();
        AscendC::DataCopyExtParams copyParamsX{1, len_x * (uint32_t) sizeof(FLOAT_T), 0, 0, 0};
        AscendC::DataCopyPadExtParams<FLOAT_T> padParams{false, 0, 0, (FLOAT_T)0};

        AscendC::DataCopyPad(xLocal, xGm[i * reduce_len + j], copyParamsX, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Reduce(uint32_t i, const AscendC::LocalTensor<FLOAT_T>& valueLocal,  const AscendC::LocalTensor<int32_t>& indiceLocal, uint32_t k, uint32_t reduce_len)
    {
        AscendC::LocalTensor<FLOAT_T> reduceDst = calcBuf0.Get<FLOAT_T>();
        AscendC::LocalTensor<FLOAT_T> reduceWorkspace = calcBuf1.Get<FLOAT_T>();
        AscendC::LocalTensor<FLOAT_T> xLocal = inQueueX.DeQue<FLOAT_T>();
        AscendC::LocalTensor<FLOAT_T> x0 = calcBuf0.Get<FLOAT_T>();
        AscendC::LocalTensor<FLOAT_T> x1 = calcBuf1.Get<FLOAT_T>();

        int index;
        FLOAT_T val;

        if constexpr(std::is_same<float, FLOAT_T>::value) {
            AscendC::ReduceMax(reduceDst, xLocal, reduceWorkspace, reduce_len, 1);
            float maxIndex = reduceDst.GetValue(1);
            uint32_t realIndex = *reinterpret_cast<uint32_t *>(&maxIndex);
            index = realIndex;
            val = reduceDst.GetValue(0);
        } else if constexpr(std::is_same<half, FLOAT_T>::value) {
            AscendC::ReduceMax(reduceDst, xLocal, reduceWorkspace, reduce_len, 1);
            half maxIndex = reduceDst.GetValue(1);
            uint16_t realIndex = *reinterpret_cast<uint16_t *>(&maxIndex);
            index = realIndex;
            val = reduceDst.GetValue(0);
        } else {
            index = 0;
            val = xLocal.GetValue(0);
            for (int i = 1; i < reduce_len;i++) {
                FLOAT_T ival = xLocal.GetValue(i);
                if (ival > val) {
                    index = i;
                    val = ival;
                }
            }
        }
        inQueueX.FreeTensor(xLocal);

        index += k;

        if (k == 0) {
            indiceLocal.SetValue(i, index);
            valueLocal.SetValue(i, val);
        } else {
            if constexpr(std::is_same<half, FLOAT_T>::value) {
                x0.SetValue(0, val);
                x1.SetValue(0, valueLocal.GetValue(i));
                AscendC::LocalTensor<uint8_t> dstLocal;
                uint64_t mask = 1;
                AscendC::BinaryRepeatParams repeatParams;
                AscendC::Compare(x0, x1, AscendC::CMPMODE::GT, mask, repeatParams);
                AscendC::GetCmpMask(dstLocal);
                if (dstLocal.GetValue(0) & 1) {
                    indiceLocal.SetValue(i, index);
                    valueLocal.SetValue(i, val);
                }
            } else {
                if (val > valueLocal.GetValue(i) ) {
                    indiceLocal.SetValue(i, index);
                    valueLocal.SetValue(i, val);
                }
            }
        }
        // AscendC::printf("%d %f\n", index, val);
    }

    __aicore__ inline void CopyOut(uint32_t i, uint32_t len_v)
    {
        AscendC::LocalTensor<FLOAT_T> vLocal =  outQueueV.DeQue<FLOAT_T>();
        AscendC::LocalTensor<int32_t> iLocal =  outQueueI.DeQue<int32_t>();
        AscendC::DataCopyExtParams copyParamsV{1, len_v * (uint32_t) sizeof(FLOAT_T), 0, 0, 0};
        AscendC::DataCopyExtParams copyParamsI{1, len_v * (uint32_t) sizeof(int32_t), 0, 0, 0};

        AscendC::DataCopyPad(vGm[i], vLocal, copyParamsV);
        AscendC::DataCopyPad(iGm[i], iLocal, copyParamsI);

        outQueueI.FreeTensor(iLocal);
        outQueueV.FreeTensor(vLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueV;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueI;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf1;
    AscendC::GlobalTensor<FLOAT_T> xGm;
    AscendC::GlobalTensor<FLOAT_T> vGm;
    AscendC::GlobalTensor<int32_t> iGm;
    uint32_t reduce_len;
    uint32_t sizeV;
    uint32_t blockNum;
};

// 可以扩展到half，下标处理要改一下(uint16->uint32)
// continuous
class KernelArgMaxWithValueFloat {
public:
    static constexpr int LEN_V = 32;
    static constexpr int TILE_V = 31; // < 32
    static constexpr int TILE_X = 512; // = 8 * 64
    static constexpr uint8_t REPEAT_TIMES_X = TILE_X / 64; 

    __aicore__ inline KernelArgMaxWithValueFloat() {}
    __aicore__ inline void Init(AscendC::TPipe* pipe, GM_ADDR x, GM_ADDR indice, GM_ADDR values, uint32_t reduce_len, uint32_t sizeV_global) {
        // AscendC::printf("reduce_len = %d tile_x = %d\n", reduce_len, TILE_X);
        this->reduce_len = reduce_len;
        
        const uint32_t sizeV_0 = sizeV_global / AscendC::GetBlockNum();
        const uint32_t sizeV_remain = sizeV_global - sizeV_0 * AscendC::GetBlockNum();
        uint32_t offset = sizeV_0 * AscendC::GetBlockIdx();
        if (AscendC::GetBlockIdx() < sizeV_remain) {
            this->sizeV = sizeV_0 + 1;
            offset += AscendC::GetBlockIdx();
        } else {
            this->sizeV = sizeV_0;
            offset += sizeV_remain;
        }

        xGm.SetGlobalBuffer((__gm__ float *)x + offset * reduce_len, reduce_len * sizeV);
        iGm.SetGlobalBuffer((__gm__ int32_t *)indice + offset, sizeV);
        vGm.SetGlobalBuffer((__gm__ float *)values + offset, sizeV);
        pipe->InitBuffer(inQueueX, 1, TILE_X * MIN(sizeV, TILE_V) * sizeof(float));
        pipe->InitBuffer(outQueueVI, 1, 2 * LEN_V * sizeof(float));
        pipe->InitBuffer(calcBuf0, LEN_V * 256);
        pipe->InitBuffer(calcBuf1, 3 * 256);
        pipe->InitBuffer(calcBuf2, 3 * LEN_V * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (int i = 0;i < sizeV;i+=TILE_V) {
            AscendC::LocalTensor<float> valueIndiceLocal = outQueueVI.AllocTensor<float>();
            uint32_t len_v = MIN(sizeV -i, TILE_V);
            uint32_t k = MIN(TILE_X, reduce_len);
            CopyIn(i, 0, len_v, k);
            Reduce0(i, len_v, k, valueIndiceLocal);
            for (;k < reduce_len;k+=TILE_X) {
                CopyIn(i, k, len_v, MIN(reduce_len - k, TILE_X));
                Reduce(i, k, len_v, MIN(reduce_len - k, TILE_X), valueIndiceLocal);
            }
            outQueueVI.EnQue<float>(valueIndiceLocal);
            CopyOut(i, len_v);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t i, uint32_t j, uint32_t len_v, uint32_t len_x)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = len_v;
        copyParamsX.blockLen = len_x * sizeof(float);
        copyParamsX.srcStride = (reduce_len - len_x) * sizeof(float);
        copyParamsX.dstStride = ((TILE_X - len_x) / 8);
        AscendC::DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>((TILE_X - len_x) % 8), -INFINITY};

        AscendC::DataCopyPad(xLocal, xGm[i * reduce_len + j], copyParamsX, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Reduce0(uint32_t i, uint32_t len_v, uint32_t len_x,
        AscendC::LocalTensor<float>& val_ind_cur)
    {
        AscendC::LocalTensor<float> reduceDstAll = calcBuf0.Get<float>();
        AscendC::LocalTensor<float> reduceDst = calcBuf1.Get<float>();

        AscendC::LocalTensor<int32_t> srcOffsetLocal0 = calcBuf2.Get<int32_t>();
        AscendC::LocalTensor<int32_t> srcOffsetLocal1 = srcOffsetLocal0[32];

        // val_ind_cur <->  value[0:32] : indice[0:32]
        AscendC::CreateVecIndex(srcOffsetLocal0, 0, LEN_V);
        AscendC::ShiftLeft(srcOffsetLocal1, srcOffsetLocal0, 3, LEN_V);
        AscendC::Adds(srcOffsetLocal1[LEN_V], srcOffsetLocal1, 4, LEN_V);
        
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();

        if (ALIGN_UP(len_x, 8) != ALIGN_UP(len_x, 64)) {
            AscendC::Duplicate(xLocal[ALIGN_UP(len_x, 8)], -INFINITY, ALIGN_UP(len_x, 64) - ALIGN_UP(len_x, 8), len_v, 1, 8 * REPEAT_TIMES_X);
        }

        AscendC::WholeReduceMax(reduceDstAll, xLocal, 64, len_v * REPEAT_TIMES_X, 4, 1, 8);

        const uint64_t mask = 0x0101010101010101 >> (64 - ALIGN_UP(len_x, 64) / 8);
        AscendC::WholeReduceMax(reduceDst, reduceDstAll, &mask, len_v, 1, 1, 8);

        // 0 8 16 ...
        AscendC::Gather(val_ind_cur, reduceDst, srcOffsetLocal1.ReinterpretCast<uint32_t>(), 0, 2 * LEN_V);

        AscendC::LocalTensor<int32_t> indiceCur = val_ind_cur.ReinterpretCast<int32_t>()[LEN_V];

        // 0 256 512 ...
        AscendC::ShiftLeft(srcOffsetLocal1, srcOffsetLocal0, 6, LEN_V);
        AscendC::Add(srcOffsetLocal1, srcOffsetLocal1, indiceCur, LEN_V);
        AscendC::Adds(srcOffsetLocal1, srcOffsetLocal1, 1, LEN_V);
        AscendC::ShiftLeft(srcOffsetLocal1, srcOffsetLocal1, 2, LEN_V);

        AscendC::Gather(srcOffsetLocal1, reduceDstAll.ReinterpretCast<int32_t>(), srcOffsetLocal1.ReinterpretCast<uint32_t>(), 0, len_v);
        AscendC::ShiftLeft(indiceCur, indiceCur, 3, len_v);
        AscendC::Add(indiceCur, indiceCur, srcOffsetLocal1, len_v);

        inQueueX.FreeTensor(xLocal);
    }

    // k > 0
    __aicore__ inline void Reduce(uint32_t i, uint32_t k, uint32_t len_v, uint32_t len_x,
        AscendC::LocalTensor<float>& valueIndiceLocal)
    {
        AscendC::LocalTensor<uint32_t> cmpResult = calcBuf1.Get<uint32_t>()[64];
        AscendC::LocalTensor<float> val_ind_cur = calcBuf1.Get<float>()[128];
        Reduce0(i, len_v, len_x, val_ind_cur);

        AscendC::LocalTensor<int32_t> indiceCur = val_ind_cur.ReinterpretCast<int32_t>()[LEN_V];
        AscendC::Adds(indiceCur, indiceCur, (int32_t) k, LEN_V, 1, {1, 1, 8, 8});
        AscendC::Compare(cmpResult, val_ind_cur, valueIndiceLocal,  AscendC::CMPMODE::GT, LEN_V, 1, { 1, 1, 1, 8, 8, 8 });
        AscendC::Max(valueIndiceLocal, val_ind_cur, valueIndiceLocal, LEN_V, 1, { 1, 1, 1, 8, 8, 8 });
        AscendC::Select(valueIndiceLocal[LEN_V], cmpResult,
                        val_ind_cur[LEN_V], valueIndiceLocal[LEN_V],
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, LEN_V, 1, { 1, 1, 1, 8, 8, 8 });
    }

    __aicore__ inline void CopyOut(uint32_t i, uint32_t len_v)
    {
        AscendC::LocalTensor<float> valueIndiceLocal =  outQueueVI.DeQue<float>();

        AscendC::DataCopyExtParams copyParamsVI{1, len_v * (uint32_t) sizeof(float), 0, 0, 0};

        AscendC::DataCopyPad(vGm[i], valueIndiceLocal, copyParamsVI);
        AscendC::DataCopyPad(iGm[i], valueIndiceLocal[LEN_V].ReinterpretCast<int32_t>(), copyParamsVI);
        outQueueVI.FreeTensor(valueIndiceLocal);
    }

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueVI;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf2;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<int32_t> iGm;
    uint32_t reduce_len;
    uint32_t sizeV;
};

// non-continuous

template<typename FLOAT_T>
class KernelArgMaxWithValue1 {
    static constexpr int TILE_V = 256 / sizeof(FLOAT_T);
public:
    __aicore__ inline KernelArgMaxWithValue1() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, uint32_t reduce_len, uint32_t prefix_size, uint32_t stride) {
        this->reduce_len = reduce_len;
        this->prefix_size = prefix_size;
        this->stride = stride;
        uint32_t sizeV = prefix_size * stride;
        this->blockNum = AscendC::GetBlockNum();

        xGm.SetGlobalBuffer((__gm__ FLOAT_T *)x, reduce_len * sizeV);
        iGm.SetGlobalBuffer((__gm__ int32_t *)indice, sizeV);
        vGm.SetGlobalBuffer((__gm__ FLOAT_T *)values, sizeV);
        pipe.InitBuffer(inQueueX, 2, TILE_V * sizeof(FLOAT_T));
        pipe.InitBuffer(outQueueI, 2, TILE_V * sizeof(int32_t));
        pipe.InitBuffer(outQueueV, 2, TILE_V * sizeof(FLOAT_T));
        pipe.InitBuffer(calcBuf, TILE_V * sizeof(FLOAT_T));
    }

    __aicore__ inline void Process() {
        uint32_t offset_x = 0;
        uint32_t offset_y = 0;
        for (int i = 0;i < prefix_size;i++) {
            for (int j = AscendC::GetBlockIdx() * TILE_V;j < stride;j+=TILE_V * blockNum) {
                const int len = stride - j > TILE_V ? TILE_V : stride - j;
                AscendC::LocalTensor<FLOAT_T> valueLocal = outQueueV.AllocTensor<FLOAT_T>();
                AscendC::LocalTensor<int32_t> indiceLocal = outQueueI.AllocTensor<int32_t>();
                AscendC::Duplicate(indiceLocal, (int32_t) 0, TILE_V);
                for (int k = 0;k < reduce_len;k++) {
                    CopyIn(offset_x + k * stride + j, len);
                    if constexpr(std::is_same<float, FLOAT_T>::value || std::is_same<half, FLOAT_T>::value) {
                        CompareMax1(k, len, valueLocal, indiceLocal);
                    } else {
                        CompareMax0(k, len, valueLocal, indiceLocal);
                    }
                }
                outQueueV.EnQue<FLOAT_T>(valueLocal);
                outQueueI.EnQue<int32_t>(indiceLocal);
                CopyOut(offset_y + j, len);
            }
            offset_y += stride;
            offset_x += reduce_len * stride;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t i, uint32_t len)
    {
        AscendC::LocalTensor<FLOAT_T> xLocal = inQueueX.AllocTensor<FLOAT_T>();
        AscendC::DataCopyExtParams copyParamsX{1, len * (uint32_t) sizeof(FLOAT_T), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyPadExtParams<FLOAT_T> padParams{false, 0, 0, (FLOAT_T)0};

        AscendC::DataCopyPad(xLocal, xGm[i], copyParamsX, padParams);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void CompareMax0(int32_t i, uint32_t len, const AscendC::LocalTensor<FLOAT_T>& valueLocal,  const AscendC::LocalTensor<int32_t>& indiceLocal)
    {
        AscendC::LocalTensor<FLOAT_T> xLocal = inQueueX.DeQue<FLOAT_T>();
        if (i == 0) {
            AscendC::DataCopy(valueLocal, xLocal, TILE_V);
        }
        for (int j = 0;j < len;j++) {
            FLOAT_T jval = xLocal.GetValue(j);
            if (jval > valueLocal.GetValue(j)) {
                indiceLocal.SetValue(j, i);
                valueLocal.SetValue(j, jval);
            }
        }
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CompareMax1(int32_t indice, uint32_t len, const AscendC::LocalTensor<FLOAT_T>& valueLocal,  const AscendC::LocalTensor<int32_t>& indiceLocal)
    {
        AscendC::LocalTensor<FLOAT_T> xLocal = inQueueX.DeQue<FLOAT_T>();
        AscendC::LocalTensor<uint8_t> mask = calcBuf.Get<uint8_t>();
        if (indice == 0) {
            AscendC::DataCopy(valueLocal, xLocal, TILE_V);
        } else {
            AscendC::Compare(mask, valueLocal, xLocal, AscendC::CMPMODE::GE, TILE_V);
        }
        inQueueX.FreeTensor(xLocal);
        AscendC::Select(valueLocal, mask, valueLocal, xLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
        float* indice_ = (float*) &indice;
        AscendC::Select(indiceLocal.ReinterpretCast<float>(), mask, indiceLocal.ReinterpretCast<float>(), *indice_, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
    }

    __aicore__ inline void CopyOut(uint32_t i, uint32_t len)
    {
        AscendC::LocalTensor<FLOAT_T> vLocal =  outQueueV.DeQue<FLOAT_T>();
        AscendC::LocalTensor<int32_t> iLocal =  outQueueI.DeQue<int32_t>();
        AscendC::DataCopyExtParams copyParamsV{1, len * (uint32_t) sizeof(FLOAT_T), 0, 0, 0};
        AscendC::DataCopyExtParams copyParamsI{1, len * (uint32_t) sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(vGm[i], vLocal, copyParamsV);
        AscendC::DataCopyPad(iGm[i], iLocal, copyParamsI);
        outQueueI.FreeTensor(iLocal);
        outQueueV.FreeTensor(vLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueV;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueI;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::GlobalTensor<FLOAT_T> xGm;
    AscendC::GlobalTensor<FLOAT_T> vGm;
    AscendC::GlobalTensor<int32_t> iGm;
    uint32_t reduce_len;
    uint32_t prefix_size;
    uint32_t stride;
    uint32_t blockNum;
};

extern "C" __global__ __aicore__ void arg_max_with_value(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 设置默认的kernel类型为纯AIV类型

    if (TILING_KEY_IS(0)) {
        if constexpr(std::is_same<DTYPE_X, float>::value) {
            AscendC::TPipe pipe;
            KernelArgMaxWithValueFloat op;
            op.Init(&pipe, x, indice, values, tiling_data.reduce_len, tiling_data.prefix_size);
            op.Process();
        } else {
            KernelArgMaxWithValue<DTYPE_X> op;
            op.Init(x, indice, values, tiling_data.reduce_len, tiling_data.prefix_size);
            op.Process();
        }
    } else if (TILING_KEY_IS(1)) {
        KernelArgMaxWithValue1<DTYPE_X> op;
        op.Init(x, indice, values, tiling_data.reduce_len, tiling_data.prefix_size, tiling_data.stride);
        op.Process();
    }
}