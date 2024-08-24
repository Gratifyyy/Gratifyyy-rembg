import { consoleDebug } from './utils'
import { ImageTensor } from './ImageTensor'
import { Model } from './Model'
import { Onnx } from './Onnx'
import { Assets } from './Assets'
import type { OnnxOptions } from './Onnx'
import type { ImageSource } from './ImageTensor'

export interface RemoveBackgroundOptions extends OnnxOptions {
  model?: BufferSource | string | URL
  resolution?: number
  output?: 'foreground' | 'mask' | 'background'
}

export async function removeBackground(
  imageSource: ImageSource,
  options: RemoveBackgroundOptions = {}
): Promise<Blob> {
  const {
    debug,
    resolution,
    model: modelSource = await Assets.getObjectUrl('u2netp.onnx')
  } = options

  debug && consoleDebug('Loading onnx runtime...')
  await Onnx.init(options)

  const imageTensor = await ImageTensor.from(imageSource)

  // 如果未指定 resolution，则使用原始图像分辨率
  const targetResolution = resolution || imageTensor.dims[1] // 使用原始宽度作为基准

  // 如果指定了 resolution 且不同于原始尺寸，则进行调整
  let resized =
    targetResolution !== imageTensor.dims[1]
      ? imageTensor.resize(targetResolution, targetResolution)
      : imageTensor

  debug && consoleDebug('Loading model...')
  const model = await Model.from(modelSource)
  await model.load()

  debug && consoleDebug('Processing...')
  const result = await model.run([resized.toBchwImageTensor().toTensor()])
  model.release()

  debug && consoleDebug('Completion', result)

  // 处理输出
  const stride = targetResolution * targetResolution
  for (let i = 0; i < 4 * stride; i += 4) {
    const idx = i / 4
    const alpha = result.data[idx]
    resized.data[i + 3] =
      (options.output === 'background' ? 1.0 - alpha : alpha) * 255
  }

  // 如果缩放了图片，需重新缩放回原尺寸
  if (targetResolution !== imageTensor.dims[1]) {
    resized = resized.resize(imageTensor.dims[1], imageTensor.dims[0])
  }

  return await resized.toBlob()
}
