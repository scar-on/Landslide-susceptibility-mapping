# LSM 生成流程优化记录

## 背景

原始 `generate_LSM.py` 会先调用 `pixel_to_image` 为整幅研究区生成所有像素窗口，再把所有预测结果拼成一个大数组后一次性写出 GeoTIFF。对大区域或较大的 `window_size` 来说，这会快速放大内存占用；同时推理前的数据准备和推理过程基本串行，GPU/CPU 容易互相等待。

## 优化 1：修复推理相关 bug

提交：`Fix CNN LSM inference bugs.`

- 修复 `Modified_SPPLayer` 未调用父类初始化的问题，并把 SPP 层注册到 `LSM_cnn` 中，避免每次 forward 临时创建层。
- `generate_LSM.py` 不再硬编码 `LSM_cnn(9)`，而是根据特征栅格数量创建模型。
- 加入 `--model_path` 参数，并使用 `map_location` 加载模型，使无 CUDA 环境也能加载权重。
- 推理输出改为 softmax 后的正类概率，而不是原始 logit。

## 优化 2：流式分块推理和写出

提交：`Stream LSM inference by raster block.`

- 新增按块读取 feature raster 的流程，只保留当前块及其窗口 padding 上下文。
- 每个块内部用 `sliding_window_view` 构造批次窗口，避免一次性生成全图所有像素窗口。
- 预测完成后立即按块写入输出 GeoTIFF，不再缓存整幅图的预测概率数组。
- 新增 `--output_path` 参数，默认仍输出到 `Result/lsm_test.tif`。

## 优化 3：并行预取栅格块

提交：`Prefetch LSM raster blocks in parallel.`

- 新增 `--num_workers` 参数，默认使用 2 个 worker 预取和归一化后续栅格块。
- worker 每次独立打开 GDAL dataset，避免在线程之间共享 GDAL dataset 对象。
- 推理当前块时后台准备后续块，减少 I/O 和 CPU 预处理等待时间。
- 预测块可以乱序完成，但写入时使用块自身的 `x_off/y_off`，不需要把结果重新拼接到内存中。

## 推荐运行方式

```bash
python generate_LSM.py \
  --feature_path origin_data/feature/ \
  --label_path origin_data/label/label1.tif \
  --model_path Result/best.pth \
  --output_path Result/lsm_test.tif \
  --window_size 15 \
  --slide_window 512 \
  --batch_size 128 \
  --num_workers 2
```

如果显存不足，优先降低 `--batch_size`；如果内存不足，优先降低 `--slide_window` 或 `--num_workers`。如果磁盘 I/O 较快但 CPU 预处理慢，可以适当增加 `--num_workers`。
