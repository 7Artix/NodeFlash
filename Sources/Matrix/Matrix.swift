struct Matrix<T: BinaryFloatingPoint> {
    let height: Int
    var elements: [T]

    var width: Int {
        return elements.count / height
    }

    var elements1Dim: [T] {
        return elements
    }

    var count: Int {
        return elements.count
    }

    init(height: Int, width: Int, initialValue: T = 0) {
        self.height = height
        self.elements = Array(repeating: initialValue, count: width * height)
    }

    init<U: BinaryFloatingPoint>(elements: [U], height: Int) {
        self.height = height
        self.elements = elements.map{ T($0) }
        assert(elements.count % height == 0, "Element count must be a multiple of height")
    }

    init<U: BinaryFloatingPoint>(elements: [[U]]) {
        guard let firstRow = elements.first else {
            fatalError("The input array is empty.")
        }
        let rowCount = elements.count
        for row in elements {
            precondition(row.count == firstRow.count, "All rows must have the same number of columns.")
        }
        self.height = rowCount
        self.elements = elements.flatMap { row in
            row.map { T($0) }
        }
    }

    func element(row: Int, col: Int) -> T {
        precondition(row >= 0 && row < height, "Row index out of range")
        precondition(col >= 0 && col < width, "Column index out of range")
        return elements[row * width + col]
    }

    mutating func setElement(_ value: T, row: Int, col: Int) {
        precondition(row >= 0 && row < height, "Row index out of range")
        precondition(col >= 0 && col < width, "Column index out of range")
        elements[row * width + col] = value
    }

    mutating func reset<U: BinaryFloatingPoint>(elements: [U]) {
        assert(elements.count == self.count, "New elements count must match the current matrix size.")
        self.elements = elements.map { T($0) }
    }

    mutating func reset<U: BinaryFloatingPoint>(elements: [[U]]) {
        guard elements.count == self.height else {
            fatalError("New matrix height (\(elements.count)) does not match the current height (\(self.height)).")
        }
        guard let firstRow = elements.first, firstRow.count == self.width else {
            fatalError("New matrix width (\(elements.first?.count ?? 0)) does not match the current width (\(self.width)).")
        }
        for row in elements {
            precondition(row.count == self.width, "All rows must have the same number of columns.")
        }
        self.elements = elements.flatMap { row in
            row.map { T($0) }
        }
    }
}

extension Matrix: CustomStringConvertible {
    var description: String {
        var result = ""
        for row in 0..<height {
            let startIndex = row * width
            let endIndex = startIndex + width
            let rowElements = elements[startIndex..<endIndex]
            result += rowElements.map { "\($0)" }.joined(separator: "\t") + "\n"
        }
        return result
    }
}