# 昇腾AI原生创新算子挑战赛（S3赛季）【算子性能挑战命题】

## 算子详情

| 序号 | Op               | Classify          | Name      | Type   | TypeRangeAll          | Attr_Default_value | Format类型 | 参考算子                          | 参考资料                                                                 |
|------|------------------|-------------------|-----------|--------|-----------------------|--------------------|------------|-----------------------------------|--------------------------------------------------------------------------|
| 1    | ArgMaxWithValue  | INPUT             | x         | tensor | fp32,fp16,int32,uint8 |                    | ND         | `tf.argmax` `tf.reduce_max`       | [tf.argmax](https://tensorflow.google.cn/api_docs/python/tf/math/argmax?hl=en) |
|      |                  | OUTPUT            | indice    | tensor | int32                 |                    | ND         |                                   | [tf.reduce_max](https://tensorflow.google.cn/api_docs/python/tf/math/reduce_max?hl=en) |                                                                         |
|      |                  | OUTPUT            | values    | tensor | fp32,fp16,int32,uint8 |                    | ND         |                                   | [mindspore.ops.ArgMaxWithValue](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ArgMaxWithValue.html) |
|      |                  | REQUIRED_ATTR     | dimension | int    |                       |                    |            |                                   | |
|      |                  | ATTR              | keep_dims | bool   |                       | FALSE              |            |                                   |                                                                          |
| 2    | MatMulSub        | INPUT             | x1        | tensor | fp32,fp16             |                    | ND         | `np.matmul` `np.sub`              | y = matmul(x1, x2) - x3                                                  |
|      |                  | INPUT             | x2        | tensor | fp32,fp16             |                    | ND         |                                   |                                                                          |
|      |                  | INPUT             | x3        | tensor | fp32,fp16             |                    | ND         |                                   |                                                                          |
|      |                  | OUTPUT            | y         | tensor | fp32,fp16             |                    | ND         |                                   |                                                                          |
| 3    | NLLLoss          | INPUT             | x         | tensor | fp32                  |                    | ND         | `torch.nn.functional.nll_loss`    | [torch.nn.functional.nll_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.nll_loss.html#torch.nn.functional.nll_loss) |
|      |                  | INPUT             | target    | tensor | int32                 |                    | ND         |                                   |                                                                          |
|      |                  | INPUT             | weight    | tensor | fp32                  |                    | ND         |                                   |                                                                          |
|      |                  | OUTPUT            | y         | tensor | fp32                  |                    | ND         |                                   |                                                                          |
|      |                  | ATTR              | reduction | string |                       | mean               |            |                                   |                                                                          |
|      |                  | ATTR              | ignore_index | int  |                       | -100               |            |                                   |                                                                          |

## 本仓库算子实现的功能问题

**ArgMaxWithValue**：
1. 暂时未知
   
**MatMulSub**：
1. 只考虑2维矩阵
2. y为(M,N)的情况下，x3只能是 (N)或（M，N）
   
**NLLLoss**：
1. reduction只能是mean和sum之一
2. ignore_index没有考虑
