import Foundation

class CNN: NN {
    let moduleCNN: CNNModule
    init(networkConfig: NNConfig, trainingConfig: TrainingConfig, moduleCNN: CNNModule) {
        self.moduleCNN = moduleCNN
        let networkConfigCNN = NNConfig(inputSize: moduleCNN.outputSizeCNN, structure: networkConfig.structure, outputLayer: networkConfig.outputLayer)
        super.init(networkConfig: networkConfigCNN, trainingConfig: trainingConfig)
    }

    func fpCNN<T: BinaryInteger & FixedWidthInteger>(input: [[T]], labels: [Double]) {
        let inputNN = moduleCNN.fpCNN(image: input, layersCNN: moduleCNN.layersCNN)
        super.fp(input: inputNN, labels: labels)
    }

    func descentbatchesCNN<T: BinaryInteger & FixedWidthInteger>(inputs: [[[T]]], labels: [[Double]]) {
        if inputs.count != self.trainingConfig.batchSize || labels.count != self.trainingConfig.batchSize {
            fatalError("Error: Gradient descent. Input Data doesn't match batch size")
        }
        let startTime = Date()
        let batchSizeInDouble = Double(self.trainingConfig.batchSize)
        self.dParametersBatch = []
        for indexBatch in 0..<self.trainingConfig.batchSize {
            print(String(format: "\rTraining... Batch Progress: %.1f%%", (Double(indexBatch+1) / Double(self.trainingConfig.batchSize) * 100.0)), terminator: "")
            fflush(stdout)
            self.fpCNN(input: inputs[indexBatch], labels: labels[indexBatch])
            self.bp()
            self.dParametersBatch.append(NN.NNParameter(weights: self.dW, biases: self.dB))
            super.resetGradients()
        }
        //初始化平均梯度结构体
        var dParametersBatchesMean = NNParameter(
            weights: self.dW.map { layer in
                layer.map { Array(repeating: Double(0.0), count: $0.count) }
            },
            biases: self.dB.map { layer in
                Array(repeating: Double(0.0), count: layer.count)
            }
        )
        print("\n", terminator: "")
        for indexBatch in 0..<self.trainingConfig.batchSize {
            print(String(format: "\rUpdating Parameters... Batch Progress: %.1f%%", (Double(indexBatch+1) / Double(self.trainingConfig.batchSize) * 100.0)), terminator: "")
            fflush(stdout)
            for indexLayer in 0..<self.dParametersBatch[indexBatch].weights.count {
                for indexNode in 0..<self.dParametersBatch[indexBatch].weights[indexLayer].count {
                    for indexWeight in 0..<self.dParametersBatch[indexBatch].weights[indexLayer][indexNode].count {
                        dParametersBatchesMean.weights[indexLayer][indexNode][indexWeight] += self.dParametersBatch[indexBatch].weights[indexLayer][indexNode][indexWeight]
                    }
                    dParametersBatchesMean.biases[indexLayer][indexNode] += self.dParametersBatch[indexBatch].biases[indexLayer][indexNode]
                }
            }
        }
        for indexLayer in 0..<dParametersBatchesMean.weights.count {
            for indexNode in 0..<dParametersBatchesMean.weights[indexLayer].count {
                for indexWeight in 0..<dParametersBatchesMean.weights[indexLayer][indexNode].count {
                    dParametersBatchesMean.weights[indexLayer][indexNode][indexWeight] /= batchSizeInDouble
                }
                dParametersBatchesMean.biases[indexLayer][indexNode] /= batchSizeInDouble
            }
        }
        self.updateParameters(withGradients: self.gradientsClip(dParameters: dParametersBatchesMean))
        print(String(format: "\nSingle batch training complete! Time: %.2f seconds", Date().timeIntervalSince(startTime)))
    }

    func descentEpochCNN<T: BinaryInteger & FixedWidthInteger>(imagesTraining: [(images: [[[T]]], labels: [[Double]])], imagesTest: (images: [[[T]]], labels: [[Double]])) {
        if imagesTraining.count != self.trainingConfig.epochSize {
            fatalError("Error: Gradient descent. Input Data doesn't match epoch size")
        }
        let startTime = Date()
        for indexBatch in 0..<self.trainingConfig.epochSize {
            print(String(format: "Epoch training... Batch Progress: %.1f%%", (Double(indexBatch+1) / Double(self.trainingConfig.epochSize) * 100.0)))
            print("Learning Rate: \(self.learningRate)")
            fflush(stdout)
            self.learningRate = self.trainingConfig.learningRateScheduler.updateLearningRate(baseLearningRate: self.trainingConfig.learningRateBase, epochIndex: indexBatch+1, epochSize: self.trainingConfig.epochSize)
            self.descentbatchesCNN(inputs: imagesTraining[indexBatch].images, labels: imagesTraining[indexBatch].labels)
            self.printAccuracyCNN(ImgaesTest: imagesTest.images, LabelsTest: imagesTest.labels)
            print(String(format: "Time spent: %.2f seconds\n", Date().timeIntervalSince(startTime)))
        }
        print(String(format: "\nEpoch training complete! Time: %.2f seconds", Date().timeIntervalSince(startTime)))
    }

    func getAccuracyCNN<T: BinaryInteger & FixedWidthInteger>(ImgaesTest: [[[T]]], LabelsTest: [[Double]]) -> (accuracy: Double, loss: Double) {
        var countCorrect = 0
        var sumLoss: Double = 0.0
        for index in 0..<ImgaesTest.count {
            print("\rTesting... Progress: \(index+1)/\(ImgaesTest.count)", terminator: "")
            fflush(stdout)
            self.fpCNN(input: ImgaesTest[index], labels: LabelsTest[index])
            sumLoss += self.getLoss()
            if self.getAnswerCheck() {
                countCorrect += 1
            }
            //self.printResultsInDetail()
        }
        print("\nTesting Done!")
        return (Double(countCorrect) / Double(ImgaesTest.count), sumLoss / Double(ImgaesTest.count))
    }

    func printAccuracyCNN<T: BinaryInteger & FixedWidthInteger>(ImgaesTest: [[[T]]], LabelsTest: [[Double]]) {
        let (accuracy, loss) = self.getAccuracyCNN(ImgaesTest: ImgaesTest, LabelsTest: LabelsTest)
        print(String(format: "Accuracy: %.2f%%  Loss: %.2f", accuracy * 100.0, loss))
    }
}