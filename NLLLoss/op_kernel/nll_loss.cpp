#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

#define MIN(a, b) ((a < b) ? a : b)

enum ReduceType {
    Mean,
    Sum,
};
template<ReduceType REDUCTION>
class KernelNLLLoss {
public:
    static constexpr int TILE_Y = 32;// <= 64
    __aicore__ inline KernelNLLLoss() {}
    __aicore__ inline void Init(AscendC::TPipe* pipe, GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR y_work, uint32_t classNum, uint32_t batchPerBlock, uint32_t batchRemain)
    {
        this->classNum = classNum;

        uint32_t start_batch;
        if (AscendC::GetBlockIdx() < batchRemain) {
            start_batch = AscendC::GetBlockIdx() * batchPerBlock + AscendC::GetBlockIdx();
            this->batchSize = batchPerBlock + 1;
        } else {
            start_batch = AscendC::GetBlockIdx() * batchPerBlock + batchRemain;
            this->batchSize = batchPerBlock;
        }
        xGm.SetGlobalBuffer((__gm__ float *)x + start_batch * classNum, batchSize * classNum);
        targetGm.SetGlobalBuffer((__gm__ int32_t *)target + start_batch, batchSize);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, classNum);
        yGm.SetGlobalBuffer((__gm__ float *)y, 1);
        yWorkGm.SetGlobalBuffer((__gm__ float *)y_work, 32 * 20);

        pipe->InitBuffer(inQueueX, 2, TILE_Y * 32);
        pipe->InitBuffer(inQueueX0, 1, 32 * 20);

        pipe->InitBuffer(inQueueTarget, 1, batchSize * sizeof(int32_t));
        pipe->InitBuffer(inQueueWeight, 1, classNum * sizeof(float));
        pipe->InitBuffer(calcBuf0, TILE_Y * sizeof(uint32_t));
        pipe->InitBuffer(calcBufX, TILE_Y * sizeof(float));
        pipe->InitBuffer(calcBufW, TILE_Y * sizeof(float));
        // 2 * 8 datablocks
        pipe->InitBuffer(calcBufSum, 2 * 256);
        AscendC::LocalTensor<int32_t> targetLocal = inQueueTarget.AllocTensor<int32_t>();
        AscendC::LocalTensor<float> weightLocal = inQueueWeight.AllocTensor<float>();

        AscendC::DataCopyExtParams copyParamsTarget{1, static_cast<uint32_t>(batchSize * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(targetLocal, targetGm, copyParamsTarget, AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        inQueueTarget.EnQue(targetLocal);
        
        AscendC::DataCopyExtParams copyParamsWeight{1, static_cast<uint32_t>(classNum * sizeof(float)), 0, 0, 0}; 
        AscendC::DataCopyPad(weightLocal, weightGm, copyParamsWeight, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        inQueueWeight.EnQue(weightLocal);
    }

    __aicore__ inline void Process()
    {

        AscendC::LocalTensor<float> v_sum = calcBufSum.Get<float>();
        AscendC::Duplicate(v_sum, 0.0f, 2 * 64);
        // init v_i
        AscendC::LocalTensor<int32_t> v_i = calcBuf0.Get<int32_t>();
        AscendC::CreateVecIndex(v_i, 0, TILE_Y);
        AscendC::ShiftLeft(v_i, v_i, 5, TILE_Y);

        AscendC::LocalTensor<int32_t> targetLocal = inQueueTarget.DeQue<int32_t>();
        uint32_t i = MIN(TILE_Y, batchSize);
        CopyIn(0, i, targetLocal);
        AscendC::LocalTensor<float> weightLocal = inQueueWeight.DeQue<float>();
        Compute(0, i, targetLocal, weightLocal, v_sum, v_sum[64]);
        for (;i < batchSize;i += TILE_Y) {
            CopyIn(i, MIN(TILE_Y, batchSize - i), targetLocal[i]);
            Compute(i, MIN(TILE_Y, batchSize - i), targetLocal[i], weightLocal, v_sum, v_sum[64]);
        }

        AscendC::LocalTensor<float> y_work = inQueueX0.AllocTensor<float>();
        AscendC::WholeReduceSum<float>(y_work, v_sum, TILE_Y, 2, 1, 1, 8);

        inQueueTarget.FreeTensor(targetLocal);
        inQueueWeight.FreeTensor(weightLocal);

        const uint32_t blockNum = AscendC::GetBlockNum();
        const float magic_number = 114.514;
        
        if (blockNum > 1) {
            y_work.SetValue(2, magic_number);

            // 32B对齐
            uint32_t work_index = AscendC::GetBlockIdx() * 8;
            AscendC::DataCopy(yWorkGm[work_index], y_work, 8);
            if (AscendC::GetBlockIdx() == 0) {
                AscendC::Duplicate(y_work[8], 0.0f, 160 - 8);
                while (true) {
                    AscendC::DataCopy(y_work[8], yWorkGm[8], 8 * (blockNum - 1));
                    bool ok = true;
                    for (int j = 1; j < blockNum; j++) {
                        if (y_work.GetValue(j * 8 + 2) != magic_number) {
                            ok = false;
                        }
                    }
                    if (ok) break;
                }                
                // 20 -> 12
                AscendC::Add(y_work, y_work, y_work[12 * 8], 8 * 8);
                AscendC::Add(y_work, y_work, y_work[6 * 8], 8 * 6);
                AscendC::Add(y_work, y_work, y_work[3 * 8], 8 * 3);
                AscendC::Add(y_work, y_work, y_work[2 * 8], 8);
                AscendC::Add(y_work, y_work, y_work[1 * 8], 8);
            }
        }
        
        if (AscendC::GetBlockIdx() == 0) {
            float sum = y_work.GetValue(0);
            float sum_weight = y_work.GetValue(1);
            if constexpr(REDUCTION == Mean) {
                sum = sum / sum_weight;
            }
            yGm.SetValue(0, -sum);
        }

        inQueueX0.FreeTensor(y_work);
    }

private:

    __aicore__ inline void CopyIn(uint32_t i, uint32_t len, const AscendC::LocalTensor<int32_t>& targetLocal)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();        
        uint32_t j = 0;
        AscendC::DataCopyParams copyParams{(uint16_t)2, (uint16_t)sizeof(float), (uint16_t)0, (uint16_t)0};
        AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
        for (;j + 1 < len;j+=2) {
            uint32_t index0 = targetLocal.GetValue(j);
            uint32_t index1 = targetLocal.GetValue(j + 1);
            copyParams.srcStride = (classNum + index1 - index0 - 1) * sizeof(float);
            AscendC::DataCopyPad(xLocal[j * (32 / sizeof(float))], xGm[(i + j) * classNum + index0], copyParams, padParams);
        }
        if (j != len) {
            uint32_t index = targetLocal.GetValue(j);
            copyParams.blockCount = 1;
            AscendC::DataCopyPad(xLocal[j * (32 / sizeof(float))], xGm[(i + j) * classNum + index], copyParams, padParams);
         }
        inQueueX.EnQue<float>(xLocal);
    }

    __aicore__ inline void Compute(uint32_t i, uint32_t len, const AscendC::LocalTensor<int32_t>& targetLocal, 
                                    const AscendC::LocalTensor<float>& weightLocal, const AscendC::LocalTensor<float>& sum, const AscendC::LocalTensor<float>& sum_weight) {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<uint32_t> v_i0 = calcBuf0.Get<uint32_t>();
        AscendC::LocalTensor<float> v_w = calcBufW.Get<float>();
        AscendC::LocalTensor<float> v_x = calcBufX.Get<float>();
 
        AscendC::LocalTensor<uint32_t> v_i1 = targetLocal.ReinterpretCast<uint32_t>();
        AscendC::ShiftLeft(v_i1, v_i1, (uint32_t)2, len);
        AscendC::Gather(v_w, weightLocal, v_i1, 0, len);
        AscendC::Gather(v_x, xLocal, v_i0, 0, len);
        AscendC::MulAddDst(sum, v_x, v_w, len);
        if constexpr(REDUCTION == Mean) {
            AscendC::Add(sum_weight, sum_weight, v_w, len);
        }
        inQueueX.FreeTensor(xLocal);
    }

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX0;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueTarget, inQueueWeight;
    // AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufW;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufX;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufSum;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<int32_t> targetGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> yWorkGm;

    uint32_t batchSize;
    uint32_t classNum;
};

extern "C" __global__ __aicore__ void nll_loss(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(0)) {
        KernelNLLLoss<Mean> op;
        op.Init(&pipe, x, target, weight, y, workspace, tiling_data.classNum, tiling_data.batchPerBlock, tiling_data.batchRemain);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelNLLLoss<Sum> op;
        op.Init(&pipe, x, target, weight, y, workspace, tiling_data.classNum, tiling_data.batchPerBlock, tiling_data.batchRemain);
        op.Process();
    }

}