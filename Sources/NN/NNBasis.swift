import Foundation

//MARK: 学习率调度器
protocol LearningRateScheduler {
    func updateLearningRate(baseLearningRate: Double, epochIndex: Int, epochSize: Int) -> Double
}
//指数衰减
class ExponentialDecay: LearningRateScheduler {
    func updateLearningRate(baseLearningRate: Double, epochIndex: Int, epochSize: Int) -> Double {
        return baseLearningRate * pow(0.01, Double(epochIndex) / Double(epochSize))
    }
}
//余弦退火
class CosineAnnealing: LearningRateScheduler {
    func updateLearningRate(baseLearningRate: Double, epochIndex: Int, epochSize: Int) -> Double {
        return baseLearningRate * 0.5 * (1 + cos(.pi * Double(epochIndex) / Double(epochSize)))
    }
}
//线性衰减
class LinearDecay: LearningRateScheduler {
    func updateLearningRate(baseLearningRate: Double, epochIndex: Int, epochSize: Int) -> Double {
        return baseLearningRate * (1 - Double(epochIndex) / Double(epochSize))
    }
}


//MARK: 激活函数
protocol ActivationFunction {
    func forward(_ x: Double) -> Double
    func backward(_ x: Double) -> Double
}
//ReLU
class ReLU: ActivationFunction {
    func forward(_ x: Double) -> Double {
        return max(0, x)
    }
    func backward(_ x: Double) -> Double {
        return x > 0 ? 1.0 : 0.0
    }
}
//Sigmoid
class Sigmoid: ActivationFunction {
    func forward(_ x: Double) -> Double {
        return 1 / (1 + exp(-x))
    }
    func backward(_ x: Double) -> Double {
        let y = forward(x)
        return y * (1 - y)
    }
}

//MARK: 权重初始化函数
//Glorot初始化 for weight
func glorotInitializer(inputSize: Int?, outputSize: Int?) -> Double {
    let limit = sqrt(6.0 / Double((inputSize ?? 1) + (outputSize ?? 1)))
    return Double.random(in: -limit...limit)
}
//HE初始化 for weight
func heInitializer(inputSize: Int?, _: Int?) -> Double {
    let stdDev = sqrt(2.0 / Double((inputSize ?? 1)))
    return generateNormal(mean: 0.0, stdDev: stdDev)
}
//常值初始化 for weight
func constantInitializer(_: Int?, _: Int?) -> Double {
    return 0.5
}

//MARK: 损失函数
protocol LossFunction {
    //init()
    func forward(predictions: [Double], labels: [Double]) -> Double
    func backward(indexNode: Int) -> Double
    func reset()
}
// 交叉熵损失函数
class CrossEntropy: LossFunction {
    var predictions: [Double]?
    var labels: [Double]?
    func forward(predictions: [Double], labels: [Double]) -> Double {
        guard predictions.count == labels.count else {
            fatalError("Error: CrossEntropy. Predictions doesn't match the labels in count")
        }
        self.predictions = predictions
        self.labels = labels
        //设定最小值防止log(0)的情况
        let epsilon = 1e-15
        let outputForward = zip(predictions, labels).map { prediction, label in 0.0 - label * log(max(prediction, epsilon)) }
        return outputForward.reduce(0.0 , +)
    }
    func backward(indexNode: Int) -> Double {
        guard let predictions = self.predictions, let labels = self.labels else {
            fatalError("Error: CrossEntropy. Make sure forward() is called before backward")
        }
        return 0.0 - (labels[indexNode] / predictions[indexNode])
    }
    func reset() {
        self.predictions = nil
        self.labels = nil
    }
}

//MARK: 归一化函数
protocol NormalizationFunction {
    func forward(inputAll: [Double], indexNode: Int) -> Double
    func backward(dInputAll: [Double], indexNode: Int) -> Double
    func reset()
}
//softmax
class Softmax: NormalizationFunction {
    var outputForward: [Double]?
    func forward(inputAll: [Double], indexNode index: Int) -> Double {
        let maxInput = inputAll.max() ?? 0.0
        //减去最大值, 防止由于指数过大导致溢出, 或指数过小导致精度问题
        let expValues = inputAll.map { exp($0 - maxInput) }
        let sumExp = expValues.reduce(0.0, +)
        //记录前向传播输出值
        let epsilon = 1e-15
        self.outputForward = expValues.map { max(($0 / sumExp), epsilon) }
        return self.outputForward![index]
    }
    //softmax求导为全梯度
    func backward(dInputAll: [Double], indexNode index: Int) -> Double {
        guard let outputForward = self.outputForward else {
            fatalError("Error: softmax. Make sure forward() is called before backward()")
        }
        if dInputAll.count != outputForward.count {
            fatalError("Error: softmax. Input gradient dosen't match the output layer")
        }
        var dValue: Double = 0.0
        for i in 0..<dInputAll.count {
            if i == index {
                dValue += dInputAll[i] * (outputForward[i] * (1 - outputForward[i]))
            }else {
                dValue += 0.0 - (dInputAll[i] * (outputForward[i] * outputForward[index]))
            }
        }
        return dValue
    }
    //对所有输出层节点梯度求值后, 将前向输出数组置nil
    func reset() {
        self.outputForward = nil
    }
}

//MARK: 生成函数
//正态分布生成
func generateNormal(mean: Double, stdDev: Double) -> Double {
    let u1 = Double.random(in: 0...1)
    let u2 = Double.random(in: 0...1)
    let z = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
    return mean + z * stdDev
}
