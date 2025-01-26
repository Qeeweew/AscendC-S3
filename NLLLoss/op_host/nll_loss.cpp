
#include "nll_loss_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  NLLLossTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  uint32_t batch_size, class_num;
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  const char* reduction = attrs->GetStr(0);
  if (strcmp(reduction, "mean") == 0) {
      context->SetTilingKey(0);
  } else {
      context->SetTilingKey(1);
  }
  size_t usrSize = 20 * 64; // 设置用户需要使用的workspace
  // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
  auto ascendcPlatform = platform_ascendc:: PlatformAscendC(context->GetPlatformInfo());
  uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
  currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。

  constexpr int BLOCK_DIM = 20;

  const int dim_num = x1_shape->GetStorageShape().GetDimNum();
  if (dim_num == 1) {
    batch_size = 1;
    class_num = x1_shape->GetStorageShape().GetDim(0);
  } else if (dim_num == 2) {
    batch_size = x1_shape->GetStorageShape().GetDim(0);
    class_num = x1_shape->GetStorageShape().GetDim(1);
  } else {
    return ge::GRAPH_FAILED;
  }
//   printf("%d %d\n", batch_size, class_num);
  // tiling.set_batchSize(batch_size);
  const int block_dim = 20 < batch_size ? 20 : batch_size;
  tiling.set_classNum(class_num);
  tiling.set_batchPerBlock(batch_size / block_dim);
  tiling.set_batchRemain(batch_size % block_dim);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());

  context->SetBlockDim(block_dim);
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
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
class NLLLoss : public OpDef {
public:
    explicit NLLLoss(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("target")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");
        this->Attr("ignore_index").AttrType(OPTIONAL).Int(-100);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(NLLLoss);
}
