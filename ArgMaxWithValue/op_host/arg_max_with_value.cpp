
#include "arg_max_with_value_tiling.h"
#include "register/op_def_registry.h"

#define CONTINUOUS_TILING 0
#define NONCONTINUOUS_TILING 1

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  ArgMaxWithValueTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  const int* dimension_  = attrs->GetAttrPointer<int>(0);
  const int dimension = *dimension_;
  // printf("%d\n", dimension);

  // int data_sz = 1;
  int stride = 1;
  int prefix_size = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
    int dim_i = x1_shape->GetStorageShape().GetDim(i);
    // data_sz *= dim_i;
    if (i > dimension) stride *= dim_i;
    if (i < dimension) prefix_size *= dim_i;
  }
  const int reduce_len = x1_shape->GetStorageShape().GetDim(dimension);

  tiling.set_stride(stride);
  tiling.set_reduce_len(reduce_len);
  tiling.set_prefix_size(prefix_size);

  const int block_dim = 20;
  context->SetBlockDim(block_dim);
  if (stride == 1) {
    context->SetTilingKey(CONTINUOUS_TILING);
  } else {
    context->SetTilingKey(NONCONTINUOUS_TILING);
  }
//   printf("%s: %d %d %d %d\n", __func__, data_sz, stride, prefix_size, reduce_len);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // const gert::Shape* x1_shape = context->GetInputShape(0);
    // gert::Shape* y_shape = context->GetOutputShape(0);
    // *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class ArgMaxWithValue : public OpDef {
public:
    explicit ArgMaxWithValue(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("indice")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dimension").Int();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(ArgMaxWithValue);
}
