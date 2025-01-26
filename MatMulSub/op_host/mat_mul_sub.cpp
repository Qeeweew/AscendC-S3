
#include "mat_mul_sub_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  auto shape_a = context->GetInputTensor(0)->GetOriginShape();
  auto shape_b = context->GetInputTensor(1)->GetOriginShape();
  auto shape_x3 = context->GetInputTensor(2)->GetOriginShape();
  int32_t M = shape_a.GetDim(0);
  int32_t N = shape_b.GetDim(1);
  int32_t K = shape_a.GetDim(1);

  // int32_t baseN = 64;
  // int32_t baseM = 64;

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  MultiCoreMatmulTiling  cubeTiling(ascendcPlatform);

  cubeTiling.SetDim(40);
  if (context->GetInputTensor(0)->GetDataType() == ge::DataType::DT_FLOAT) {
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    if (M % 128 != 0 || N % 128 != 0) {
      context->SetTilingKey(0);
    } else {
      context->SetTilingKey(1);
      cubeTiling.SetFixSplit(128, 128, -1);
    }
  } else {
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    if (M % 128 != 0 || N % 128 != 0) {
      context->SetTilingKey(0);
    } else {
      context->SetTilingKey(1);
      cubeTiling.SetFixSplit(128, 128, -1);
    }
  }
  cubeTiling.SetShape(M, N, K);
  cubeTiling.SetOrgShape(M, N, K);
  // cubeTiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
  cubeTiling.SetBias(false);
  cubeTiling.SetTraverse(MatrixTraverse::FIRSTM);  // 设置遍历方式
  cubeTiling.SetBufferSpace(-1, -1, -1);

  MatMulSubTilingData tiling;
  if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
    context->SetTilingKey(0);
  }

  if (shape_x3.GetDimNum() == 2) {
    tiling.set_use_broadcast(false);
  } else {
    tiling.set_use_broadcast(true);
  }
  tiling.set_M(M);
  tiling.set_K(K);
  tiling.set_N(N);

  uint32_t coreNum = ascendcPlatform.GetCoreNum();
  context->SetBlockDim(20);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  size_t userWorkspaceSize = M * N * sizeof(float);
  size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class MatMulSub : public OpDef {
public:
    explicit MatMulSub(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x3")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(MatMulSub);
}
