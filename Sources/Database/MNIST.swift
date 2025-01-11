import Foundation

class MNIST {
    let numberOfItems: Int
    let numberOfRows: Int
    let numberOfCols: Int
    var database: Database

    init(database: Database) {
        self.database = database
        let magicNumberImages = self.database.imagesData[0..<4].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        if magicNumberImages != 0x00000803 {
            fatalError("Error: MNIST. Data is broken")
        }
        let magicNumberLabels = self.database.labelsData[0..<4].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        if magicNumberLabels != 0x00000801 {
            fatalError("Error: MNIST. Data is broken")
        }
        let numberOfItemsImages = self.database.imagesData[4..<8].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        let numberOfItemsLabels = self.database.labelsData[4..<8].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        if numberOfItemsImages != numberOfItemsLabels {
            fatalError("Error: MNIST. Data is broken")
        }
        self.numberOfItems = Int(numberOfItemsImages)
        self.numberOfRows = Int(self.database.imagesData[8..<12].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian })
        self.numberOfCols = Int(self.database.imagesData[12..<16].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian })
    }

    func getImage(index: Int) -> (image: [[UInt8]], labelIndex: Int) {
        var image: [[UInt8]] = []
        for indexRow in 0..<self.numberOfRows {
            let indexRowStart = indexRow * self.numberOfCols + 16 + (index * self.numberOfRows * self.numberOfCols)
            let indexRowEnd = indexRowStart + self.numberOfCols
            let arrayRow = Array(database.imagesData[indexRowStart..<indexRowEnd]).map { UInt8($0) }
            image.append(arrayRow)
        }
        let label: Int = Int(database.labelsData[index + 8])
        return (image, label)
    }

    func getImageForNetwork(index: Int) -> (image: [Double], labels: [Double]) {
        let image = self.getImage(index: index).0.flatMap { row in
            row.map { Double($0) }
        }
        let label = self.getImage(index: index).1
        var oneHotLabel = Array(repeating: Double(0.0), count: 10)
        oneHotLabel[label] = 1.0
        return (image, oneHotLabel)
    }

    func getImageForNetwork2Dim(index: Int) -> (image: [[Double]], labels: [Double]) {
        let image = self.getImage(index: index).0.map { row in
            row.map { Double($0) }
        }
        let label = self.getImage(index: index).1
        var oneHotLabel = Array(repeating: Double(0.0), count: 10)
        oneHotLabel[label] = 1.0
        return (image, oneHotLabel)
    }

    func getImagesBatch(fromIndex index: Int, batchSize: Int) -> (images: [[[UInt8]]], labelsIndex: [Int]) {
        var images: [[[UInt8]]] = []
        var labels: [Int] = []
        for indexBatch in 0..<batchSize {
            images.append(self.getImage(index: index + indexBatch).0)
            labels.append(self.getImage(index: index + indexBatch).1)
        }
        return (images, labels)
    }

    func getImagesBatchForNN(fromIndex index: Int, batchSize: Int) -> (images: [[Double]], labels: [[Double]]) {
        var imagesInOneDimDouble: [[Double]] = []
        var labelsInOneDimDouble: [[Double]] = []
        for indexBatch in 0..<batchSize {
            let image = self.getImage(index: index + indexBatch).0.flatMap { row in
                row.map { Double($0) }
            }
            imagesInOneDimDouble.append(image)
            let label = self.getImage(index: index + indexBatch).1
            var oneHotLabel = Array(repeating: Double(0.0), count: 10)
            oneHotLabel[label] = 1.0
            labelsInOneDimDouble.append(oneHotLabel)
        }
        return (imagesInOneDimDouble, labelsInOneDimDouble)
    }

    func getImagesBatchForCNN(fromIndex index: Int, batchSize: Int) -> (images: [[[UInt8]]], labels: [[Double]]) {
        var imagesInTwoDim: [[[UInt8]]] = []
        var labelsInTwoDim: [[Double]] = []
        for indexBatch in 0..<batchSize {
            let image = self.getImage(index: index + indexBatch).0
            imagesInTwoDim.append(image)
            let label = self.getImage(index: index + indexBatch).1
            var oneHotLabel = Array(repeating: Double(0.0), count: 10)
            oneHotLabel[label] = 1.0
            labelsInTwoDim.append(oneHotLabel)
        }
        return (imagesInTwoDim, labelsInTwoDim)
    }
}