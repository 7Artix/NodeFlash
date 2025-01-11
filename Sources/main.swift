import Foundation
import CoreGraphics

let pathTrainingImages = "./mnist_database/train-images.idx3-ubyte"
let pathTrainingLabels = "./mnist_database/train-labels.idx1-ubyte"
let pathTestImages = "./mnist_database/t10k-images.idx3-ubyte"
let pathTestLabels = "./mnist_database/t10k-labels.idx1-ubyte"
let trainingData = Database(imagesPath: pathTrainingImages, labelsPath: pathTrainingLabels)
let testData = Database(imagesPath: pathTestImages, labelsPath: pathTestLabels)

let mnistTraining = MNIST(database: trainingData)
let mnistTest = MNIST(database: testData)

let imageTestSave = mnistTraining.getImage(index: 18).image
let labelTestSave = mnistTraining.getImage(index: 18).labelIndex

let batchSize = 50
let epochSize = 1200

//let trainingImages = mnistTraining.getImagesBatchForNetwork(fromIndex: 0, batchSize: batchSize)
let testImages = mnistTest.getImagesBatchForNN(fromIndex: 0, batchSize: 50)
let testImagesAll = mnistTest.getImagesBatchForNN(fromIndex: 0, batchSize: 10000)

//MARK: 测试保存照片
// do {
//     for i in 0..<20 {
//         let trainingImages = mnistTraining.getImagesBatchForNetwork(fromIndex: i * batchSize, batchSize: batchSize)
//         let image1Dim = trainingImages.images[0]
//         var image2Dim: [[UInt8]] = []
//         for i in 0..<28 {
//             let row = Array(image1Dim[i * 28..<(i + 1) * 28]).map { UInt8($0) }
//             image2Dim.append(row)
//         }
//         let label = trainingImages.labels[0]
//         let labelIndex: Int = Int(label.firstIndex(of: Double(1.0))!)
//         let imageRecipeGray8 = ImageRecipe(width: 28, height: 28, pixelFormat: .GRAY8, provider: try .makeProviderGrayscale(from: image2Dim))
//         let cgImageGray8 = try CGImage.createCGImage(imageRecipe: imageRecipeGray8)
//         try cgImageGray8.saveCGImage(useName: "image_\(i)_label_\(labelIndex)", toPath: "./test", as: .png)
//     }
// }

let labelsMeaning = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]

// MARK: NN Training
var networkStructure: [[NodeStructure]] = []
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 512))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 256))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 128))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 64))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 10))
let outputlayer = OutputLayer(outputSize: 10, normalizationFunction: Softmax(), lossFunction: CrossEntropy())
let networkConfig = NNConfig(inputSize: 784, structure: networkStructure, outputLayer: outputlayer)
let trainingConfig = TrainingConfig(batchSize: batchSize, epochSize: epochSize, learningRateBase: 0.05, learningRateScheduler: ExponentialDecay(), negativeAttempts: 5)
let network = NN(networkConfig: networkConfig, trainingConfig: trainingConfig)
network.setLabelsMeaning(use: labelsMeaning)
network.setGradientThreshold(threshold: 4.0)
var trainingImages: [(images: [[Double]], labels: [[Double]])] = []
for i in 0..<epochSize {
    print(String(format: "\rPreparing training data: %.1f%%", Double(i+1) * 100.0 / Double(epochSize)), terminator: "")
    fflush(stdout)
    trainingImages.append(mnistTraining.getImagesBatchForNN(fromIndex: i * batchSize, batchSize: batchSize))
}
print("\n")
//单步前向传播时间
// let timeSingleStart = Date()
// network.fp(input: trainingImages[0].images[0], labels: trainingImages[0].labels[0])
// let timeSingleEnd = Date()
// print("Single step time: \(timeSingleEnd.timeIntervalSince(timeSingleStart))")
network.descentEpoch(imagesTraining: trainingImages, imagesTest: testImages)
network.printAccuracy(ImgaesTest: testImagesAll.images, LabelsTest: testImagesAll.labels)


//MARK: CNN Training
// var layersPoolings: [PoolingLayer] = []
// layersPoolings.append(PoolingLayer(poolingMethod: .max, poolingHeight: 2, poolingWidth: 2))
// layersPoolings.append(PoolingLayer(poolingMethod: .average, poolingHeight: 2, poolingWidth: 2))

// var layersCNN: [CNNLayer] = []

// // layersCNN.append(CNNLayer(filter: filterHorizontal, poolingLayers: layersPoolings))
// // layersCNN.append(CNNLayer(filter: filterVertical, poolingLayers: layersPoolings))
// layersCNN.append(CNNLayer(filter: filterSlash, poolingLayers: layersPoolings))
// layersCNN.append(CNNLayer(filter: filterBackslash, poolingLayers: layersPoolings))
// layersCNN.append(CNNLayer(filter: filterCross, poolingLayers: layersPoolings))
// layersCNN.append(CNNLayer(filter: filterVerticalRight, poolingLayers: layersPoolings))
// layersCNN.append(CNNLayer(filter: filterVerticalLeft, poolingLayers: layersPoolings))
// layersCNN.append(CNNLayer(filter: filterLeftTop, poolingLayers: layersPoolings))
// layersCNN.append(CNNLayer(filter: filterRightTop, poolingLayers: layersPoolings))
// layersCNN.append(CNNLayer(filter: filterLeftBottom, poolingLayers: layersPoolings))
// layersCNN.append(CNNLayer(filter: filterRightBottom, poolingLayers: layersPoolings))
// layersCNN.append(CNNLayer(filter: filterLess, poolingLayers: layersPoolings))
// layersCNN.append(CNNLayer(filter: filterAngle, poolingLayers: layersPoolings))

// let imageSample: [[UInt8]] = mnistTraining.getImage(index: 18).image
// let moduleCNN = CNNModule(imageSample: imageSample, layersCNN: layersCNN)

//MARK: 测试保存卷积池化照片
// for (i, layer) in moduleCNN.layersCNN.enumerated() {
//     print("Layer:")
//     let imageCon = moduleCNN.convolution(imageTestSave, with: layer.filter)
//     var imagePooling = imageCon
//     for poolingLayer in layer.poolingLayers {
//         imagePooling = moduleCNN.pooling(imagePooling, with: poolingLayer.poolingMethod, windowHeight: poolingLayer.poolingHeight, windowWidth: poolingLayer.poolingWidth)
//     }
//     let splitOutputImageRG16 = splitIntoRG16(imageInput: imageCon)
//     let rChannel: [[UInt16]] = moduleCNN.normalization2Dim(changeToUnsigned(splitOutputImageRG16.R))
//     let gChannel: [[UInt16]] = moduleCNN.normalization2Dim(changeToUnsigned(splitOutputImageRG16.G))
//     let bChannel: [[UInt16]] = moduleCNN.normalization2Dim(changeToUnsigned(splitOutputImageRG16.B))
//     do {
//         let imageRecipeRGB161616 = ImageRecipe(width: 28, height: 28, pixelFormat: .RGB161616, provider: try .makeProviderRGB(fromR: rChannel, fromG: gChannel, fromB: bChannel, fromA: nil))
//         let cgImageRGB161616 = try CGImage.createCGImage(imageRecipe: imageRecipeRGB161616)
//         try cgImageRGB161616.saveCGImage(useName: "testC_\(i)", toPath: "./test", as: .png)
//     }catch{print("Error: \(error)")}
//     let splitOutputImageRG16P = splitIntoRG16(imageInput: imagePooling)
//     let rChannelP: [[UInt16]] = moduleCNN.normalization2Dim(changeToUnsigned(splitOutputImageRG16P.R))
//     let gChannelP: [[UInt16]] = moduleCNN.normalization2Dim(changeToUnsigned(splitOutputImageRG16P.G))
//     let bChannelP: [[UInt16]] = moduleCNN.normalization2Dim(changeToUnsigned(splitOutputImageRG16P.B))
//     do {
//         let imageRecipeRGB161616 = ImageRecipe(width: 7, height: 7, pixelFormat: .RGB161616, provider: try .makeProviderRGB(fromR: rChannelP, fromG: gChannelP, fromB: bChannelP, fromA: nil))
//         let cgImageRGB161616 = try CGImage.createCGImage(imageRecipe: imageRecipeRGB161616)
//         try cgImageRGB161616.saveCGImage(useName: "testP_\(i)", toPath: "./test", as: .png)
//     }catch{print("Error: \(error)")}
// }

// let epochSize = 1200
// var networkStructureCNN: [[NodeStructure]] = []
// networkStructureCNN.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 500))
// networkStructureCNN.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 250))
// networkStructureCNN.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 100))
// networkStructureCNN.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 50))
// networkStructureCNN.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 10))
// let outputlayerCNN = OutputLayer(outputSize: 10, normalizationFunction: Softmax(), lossFunction: CrossEntropy())
// let networkConfigCNN = NNConfig(inputSize: 1, structure: networkStructureCNN, outputLayer: outputlayerCNN)
// let trainingConfigCNN = TrainingConfig(batchSize: batchSize, epochSize: epochSize, learningRateBase: 0.1, learningRateScheduler: ExponentialDecay(), negativeAttempts: 5)
// let networkCNN = CNN(networkConfig: networkConfigCNN, trainingConfig: trainingConfigCNN, moduleCNN: moduleCNN)
// networkCNN.setLabelsMeaning(use: labelsMeaning)
// networkCNN.setGradientThreshold(threshold: 2.0)
// print("\nLayers: \(networkCNN.layerStructure)")
// let testImagesCNN = mnistTest.getImagesBatchForCNN(fromIndex: 0, batchSize: 5)
// let testImagesCNNAll = mnistTest.getImagesBatchForCNN(fromIndex: 0, batchSize: 10000)
// let testImagesCNNFromTraining = mnistTraining.getImagesBatchForCNN(fromIndex: 0, batchSize: 10000)
// var trainingImagesCNN: [(images: [[[UInt8]]], labels: [[Double]])] = []
// for i in 0..<epochSize {
//     print(String(format: "\rPreparing training data: %.1f%%", Double(i+1) * 100.0 / Double(epochSize)), terminator: "")
//     fflush(stdout)
//     trainingImagesCNN.append(mnistTraining.getImagesBatchForCNN(fromIndex: i * batchSize, batchSize: batchSize))
// }
// print("\n")

// networkCNN.descentEpochCNN(imagesTraining: trainingImagesCNN, imagesTest: testImagesCNN)
// print("Test Samples Accuracy:")
// networkCNN.printAccuracyCNN(ImgaesTest: testImagesCNNAll.images, LabelsTest: testImagesCNNAll.labels)
