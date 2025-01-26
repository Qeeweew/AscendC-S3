
#include "register/tilingdata_base.h"

#define CONTINUS_TILING 0
#define NONCONTINUS_TILING 1

namespace optiling {
BEGIN_TILING_DATA_DEF(ArgMaxWithValueTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, stride);
  TILING_DATA_FIELD_DEF(uint32_t, prefix_size);
  TILING_DATA_FIELD_DEF(uint32_t, reduce_len);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ArgMaxWithValue, ArgMaxWithValueTilingData)
}
