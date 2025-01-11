import Foundation
import CoreGraphics

enum PollingMethod {
    case max
    case average
    case l2
}

struct PoolingLayer {
    var poolingMethod: PollingMethod
    var poolingHeight: Int
    var poolingWidth: Int
}

struct CNNLayer {
    var filter: [[Int]]
    var poolingLayers: [PoolingLayer]
}

class CNNModule {
    var layersCNN: [CNNLayer]
    var heightImageInputCNN: Int
    var widthImageInputCNN: Int
    var outputSizeCNN: Int

    init<T: BinaryInteger & FixedWidthInteger>(imageSample: [[T]], layersCNN: [CNNLayer]) {
        self.layersCNN = layersCNN
        self.heightImageInputCNN = imageSample.count
        self.widthImageInputCNN = imageSample.first?.count ?? 0
        self.outputSizeCNN = 0
        self.outputSizeCNN = self.fpCNN(image: imageSample, layersCNN: layersCNN).count
    }

    func fpCNN<T: BinaryInteger & FixedWidthInteger>(image: [[T]], layersCNN: [CNNLayer]) -> [Double] {
        var imageProcessed: [[Int64]]
        var outputInFlatDouble: [Double] = []
        for layer in layersCNN {
            //print("Using filter: \(layer.filter)")
            imageProcessed = convolution(image, with: layer.filter)
            for poolingLayer in layer.poolingLayers {
                imageProcessed = pooling(imageProcessed, with: poolingLayer.poolingMethod, windowHeight: poolingLayer.poolingHeight, windowWidth: poolingLayer.poolingWidth)
            }
            outputInFlatDouble.append(contentsOf: flatToDouble(imageProcessed))
        }
        return outputInFlatDouble
    }

    func flatToDouble<T: BinaryInteger & FixedWidthInteger>(_ image: [[T]]) -> [Double] {
        return image.flatMap { row in
            row.map { Double($0) }
        }
    }

    //将像素值缩小到整个值域, 避免像素值过大导致溢出
    func safeClip2Dim<T_origin: BinaryInteger, T: BinaryInteger & FixedWidthInteger>(_ origin: [[T_origin]]) -> [[T]] {
        let height = origin.count
        let width = origin.first?.count ?? 0
        var normalized = Array(repeating: Array(repeating: T(0), count: width), count: height)
        if let maxValue = origin.flatMap({$0}).max() {
            if maxValue <= T.max {
                normalized = origin.map { row in 
                    row.map { T($0) }
                }
            } else {
                normalized = origin.map { row in 
                    row.map { print(T(round(Double($0) / Double(maxValue) * Double(T.max)))); return T(round(Double($0) / Double(maxValue) * Double(T.max)))}
                }
            }
        }
        return normalized
    }

    //将像素值均匀放大到整个值域, 避免像素值过小导致显示不明显
    func normalization2Dim<T_origin: BinaryInteger, T: BinaryInteger & FixedWidthInteger>(_ origin: [[T_origin]]) -> [[T]] {
        let height = origin.count
        let width = origin.first?.count ?? 0
        var normalized = Array(repeating: Array(repeating: T(0), count: width), count: height)
        if let maxValue = origin.flatMap({$0}).max() {
            if maxValue >= T.max {
                normalized = safeClip2Dim(origin)
            } else {
                normalized = origin.map { row in 
                    row.map { T(min(Double(T.max), max(0, round(Double($0) / Double(maxValue) * Double(T.max))))) }
                }
            }
        }
        return normalized
    }
    
    //MARK: 卷积函数
    func convolution<T: BinaryInteger & FixedWidthInteger>(_ image: [[T]], with filter: [[Int]]) -> [[Int64]]{
        let filterHeight = filter.count
        let filterWidth = filter.first?.count ?? 0
        let peddingNumberWidth = Int(floor(Double(filterWidth) / 2 + 0.1))
        let peddingNumberHeight = Int(floor(Double(filterHeight) / 2 + 0.1))
        //增加Padding行列的拓展图像
        var convolutionImage: [[Int64]] = image.map { row in 
            row.map { Int64($0) }
        }
        let imageHeight = convolutionImage.count
        let imageWidth = convolutionImage.first?.count ?? 0
        var outputImageInt64 = Array(repeating: Array(repeating: Int64(0), count: imageWidth), count: imageHeight)
        //与输入图像尺寸相同的图像用以返回
        //var outputImage = Array(repeating: Array(repeating: T(0), count: imageWidth), count: imageHeight)
        //添加Padding行   
        for _ in 0..<peddingNumberHeight {
            convolutionImage.insert(convolutionImage[0], at: 0)
            convolutionImage.append(convolutionImage.last ?? convolutionImage[0])
        }
        //添加Padding列
        for rowIndex in 0..<convolutionImage.count {
            for _ in 0..<peddingNumberWidth {
                convolutionImage[rowIndex].insert(convolutionImage[rowIndex][0], at: 0)
                convolutionImage[rowIndex].append(convolutionImage[rowIndex].last ?? convolutionImage[rowIndex][0])
            }
        }
        //将四个角补充的元素取平均值均匀处理
        let rangeLeft = stride(from: peddingNumberWidth - 1, through: 0, by: -1)
        let rangeRight = (imageWidth - peddingNumberWidth)..<(imageWidth)
        let rangeTop = stride(from: peddingNumberHeight - 1, through: 0, by: -1)
        let rangeBottom = (imageHeight - peddingNumberHeight)..<imageHeight
        //处理左上角元素
        for rowIndex in rangeTop {
            for colIndex in rangeLeft {
                convolutionImage[rowIndex][colIndex] = Int64(round(
                    (Double(convolutionImage[rowIndex][colIndex + 1]) +
                    Double(convolutionImage[rowIndex + 1][colIndex])) / 2))
            }
        }
        //处理左下角元素
        for rowIndex in rangeBottom {
            for colIndex in rangeLeft {
                convolutionImage[rowIndex][colIndex] = Int64(round(
                    (Double(convolutionImage[rowIndex][colIndex + 1]) + 
                    Double(convolutionImage[rowIndex - 1][colIndex])) / 2))
            }
        }
        //处理右上角元素
        for rowIndex in rangeTop {
            for colIndex in rangeRight {
                convolutionImage[rowIndex][colIndex] = Int64(round(
                    (Double(convolutionImage[rowIndex][colIndex - 1]) + 
                    Double(convolutionImage[rowIndex + 1][colIndex])) / 2))
            }
        }
        //处理右下角元素
        for rowIndex in rangeBottom {
            for colIndex in rangeRight {
                convolutionImage[rowIndex][colIndex] = Int64(round(
                    (Double(convolutionImage[rowIndex][colIndex - 1]) + 
                    Double(convolutionImage[rowIndex - 1][colIndex])) / 2))
            }
        }
        // for i in 0..<30 {
        //     print(String(format:"%@", convolutionImage[i].map { String(format: "%.02X", $0) }.joined(separator: " ")))
        // }
        //卷积
        for i in peddingNumberHeight..<(peddingNumberHeight + imageHeight) {
            for j in peddingNumberWidth..<(peddingNumberWidth + imageWidth) {
                var pixelConvolutionalValue = Int64(0)
                for k in 0..<filterHeight {
                    for l in 0..<filterWidth {
                        pixelConvolutionalValue += Int64(filter[k][l]) * convolutionImage[i-peddingNumberHeight+k][j-peddingNumberWidth+l]
                    }
                }
                outputImageInt64[i-1][j-1] = pixelConvolutionalValue
            }
        }
        // let splitOutputImageRG16 = splitIntoRG16(imageInput: outputImageInt64)
        // let rChannel: [[UInt16]] = normalization2Dim(changeToUnsigned(splitOutputImageRG16.R))
        // let gChannel: [[UInt16]] = normalization2Dim(changeToUnsigned(splitOutputImageRG16.G))
        // let bChannel: [[UInt16]] = normalization2Dim(changeToUnsigned(splitOutputImageRG16.B))
        // do {
        //     let imageRecipeRGB161616 = ImageRecipe(width: 28, height: 28, pixelFormat: .RGB161616, provider: try .makeProviderRGB(fromR: rChannel, fromG: gChannel, fromB: bChannel, fromA: nil))
        //     let cgImageRGB161616 = try CGImage.createCGImage(imageRecipe: imageRecipeRGB161616)
        //     try cgImageRGB161616.saveCGImage(useName: "testConvolution", toPath: "./test", as: .png)
        // }catch{print("Error: \(error)")}
        return outputImageInt64
    }

    //MARK: 池化函数:修剪与池化窗口倍数不匹配的像素
    func pooling<T: BinaryInteger & FixedWidthInteger>(_ image: [[T]], with method: PollingMethod, windowHeight heightWindow: Int, windowWidth widthWindow: Int) -> [[T]] {
        let heightInput = image.count
        let widthInput = image.first?.count ?? 0
        let heightOutput = heightInput / heightWindow
        let widthOutput = widthInput / widthWindow
        var imageOutput = Array(repeating: Array(repeating: T(0), count: widthOutput), count: heightOutput)
        for indexRowOutput in 0..<heightOutput {
            for indexColOutput in 0..<widthOutput {
                var poolingValuesTemp: [T] = []
                for indexRowWindow in 0..<heightWindow {
                    for indexColWindow in 0..<widthWindow {
                        poolingValuesTemp.append(image[indexRowOutput*heightWindow+indexRowWindow][indexColOutput*widthWindow+indexColWindow])
                    }
                }
                imageOutput[indexRowOutput][indexColOutput] = poolingAction(poolingValuesTemp, with: method)
            }
        }
        // let splitOutputImageRG16 = splitIntoRG16(imageInput: imageOutput)
        // let rChannel: [[UInt16]] = normalization2Dim(changeToUnsigned(splitOutputImageRG16.R))
        // let gChannel: [[UInt16]] = normalization2Dim(changeToUnsigned(splitOutputImageRG16.G))
        // let bChannel: [[UInt16]] = normalization2Dim(changeToUnsigned(splitOutputImageRG16.B))
        // do {
        //     let imageRecipeRGB161616 = ImageRecipe(width: widthOutput, height: heightOutput, pixelFormat: .RGB161616, provider: try .makeProviderRGB(fromR: rChannel, fromG: gChannel, fromB: bChannel, fromA: nil))
        //     let cgImageRGB161616 = try CGImage.createCGImage(imageRecipe: imageRecipeRGB161616)
        //     try cgImageRGB161616.saveCGImage(useName: "testPooling", toPath: "./test", as: .png)
        // }catch{print("Error: \(error)")}
        return imageOutput
    }

    private func poolingAction<T: BinaryInteger & FixedWidthInteger>(_ values: [T], with method: PollingMethod) -> T {
        var result: T
        switch method {
            case .average:
                var valueSum: Int64 = 0
                valueSum = Int64(values.reduce(0, {$0 + $1}))
                result = T(valueSum / Int64(values.count))
            case .l2:
                var squareSum: UInt64 = 0
                squareSum = UInt64(values.reduce(0, {$0 + $1 * $1}))
                result = T(sqrt(Double(squareSum)))
            case .max:
                result = values.max() ?? T(0)
        }
        return result
    }
}