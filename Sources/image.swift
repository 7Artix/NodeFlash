import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

//Supported Pixel Formats: 
//https://developer.apple.com/library/archive/documentation/GraphicsImaging/Conceptual/drawingwithquartz2d/dq_context/dq_context.html#//apple_ref/doc/uid/TP30001066-CH203

//MARK: 支持的像素格式
enum SupportedPixelFormat {
    case GRAY8
    case GRAY16
    case RGB888
    case RGB161616
    case RGBA8888
    case RGBA16161616
}

//MARK: 图片成分结构体
struct ImageRecipe {
    let width: Int
    let height: Int
    let pixelFormat: SupportedPixelFormat
    let colorSpace: CGColorSpace
    let bitsPerPixel: Int
    let bitsPerComponent: Int
    let bytesPerRow: Int
    let bitmapInfo: CGBitmapInfo
    let provider: CGDataProvider

    init(width: Int, height: Int, pixelFormat: SupportedPixelFormat, provider: CGDataProvider) {
        self.width = width
        self.height = height
        self.provider = provider
        self.pixelFormat = pixelFormat
        switch pixelFormat {
            case .GRAY8:
                self.colorSpace = CGColorSpaceCreateDeviceGray()
                self.bitsPerPixel = 8
                self.bitsPerComponent = 8
                self.bytesPerRow = width
                self.bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
            case .GRAY16:
                self.colorSpace = CGColorSpaceCreateDeviceGray()
                self.bitsPerPixel = 16
                self.bitsPerComponent = 16
                self.bytesPerRow = width * 2
                self.bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
            case .RGB888:
                self.colorSpace = CGColorSpaceCreateDeviceRGB()
                self.bitsPerPixel = 32
                self.bitsPerComponent = 8
                self.bytesPerRow = width * 4
                self.bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
            case .RGB161616:
                self.colorSpace = CGColorSpaceCreateDeviceRGB()
                self.bitsPerPixel = 64
                self.bitsPerComponent = 16
                self.bytesPerRow = width * 8
                self.bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
            case .RGBA8888:
                self.colorSpace = CGColorSpaceCreateDeviceRGB()
                self.bitsPerPixel = 32
                self.bitsPerComponent = 8
                self.bytesPerRow = width * 4
                self.bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
            case .RGBA16161616:
                self.colorSpace = CGColorSpaceCreateDeviceRGB()
                self.bitsPerPixel = 64
                self.bitsPerComponent = 16
                self.bytesPerRow = width * 8
                self.bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        }
    }
}

//MARK: 拓展CGImage 保存&创建
extension CGImage {
    func saveCGImage(useName name: String, toPath path: String, as type: UTType) throws {
        guard let fileExtension = type.preferredFilenameExtension else {
            throw NSError(domain: "ImageSaveError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Unsupported file type: \(type)"])
        }
        // 将文件路径转换为 URL
        let fileURL = URL(fileURLWithPath: path).appendingPathComponent("\(name).\(fileExtension)")

        // 创建目标文件的 `CGImageDestination`
        guard let destination = CGImageDestinationCreateWithURL(fileURL as CFURL, type.identifier as CFString, 1, nil) else {
            throw NSError(domain: "ImageSaveError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Failed to create image destination."])
        }
        
        // 将 CGImage 添加到目标
        CGImageDestinationAddImage(destination, self, nil)
        
        // 完成保存
        if !CGImageDestinationFinalize(destination) {
            throw NSError(domain: "ImageSaveError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Failed to finalize image destination."])
        }
        
        print("Image saved successfully at \(path) as \(name).\(fileExtension)")
    }
    static func createCGImage(imageRecipe: ImageRecipe, shouldInterpolate: Bool = false, intent: CGColorRenderingIntent = .defaultIntent) throws -> CGImage {
        guard let cgImage = CGImage(
            width: imageRecipe.width, 
            height: imageRecipe.height, 
            bitsPerComponent: imageRecipe.bitsPerComponent, 
            bitsPerPixel: imageRecipe.bitsPerPixel, 
            bytesPerRow: imageRecipe.bytesPerRow, 
            space: imageRecipe.colorSpace, 
            bitmapInfo: imageRecipe.bitmapInfo, 
            provider: imageRecipe.provider, 
            decode: nil, 
            shouldInterpolate: shouldInterpolate, 
            intent: intent)
        else {
            throw NSError(domain: "ImageDataError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGImage."])
        }
        return cgImage
    }
}

//MARK: Provider处理像素数据
extension CGDataProvider {
    static func makeProviderGrayscale(from pixelData: [[UInt8]]) throws -> CGDataProvider {
        let dataWithOneDim = pixelData.flatMap{$0}
        let cfdata = Data(dataWithOneDim) as CFData
        guard let provider = CGDataProvider(data: cfdata) else {
            throw NSError(domain: "DataProviderError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGDataProvider."])
        }
        return provider
    }
    static func makeProviderGrayscale(from pixelData: [[UInt16]]) throws -> CGDataProvider {
        let dataWithOneDim = pixelData.flatMap { row in
            row.flatMap { value in 
            [UInt8(value >> 8 & 0xFF), UInt8(value & 0xFF)]}
        }
        let cfdata = Data(dataWithOneDim) as CFData
        guard let provider = CGDataProvider(data: cfdata) else {
            throw NSError(domain: "DataProviderError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGDataProvider."])
        }
        return provider
    }
    static func makeProviderRGB(fromR pixelDataR:[[Any]], fromG pixelDataG:[[Any]], fromB pixelDataB: [[Any]], fromA pixelDataA: [[Any]]?) throws -> CGDataProvider{        
        var dataWithOneDim: [UInt8] = []
        if type(of: pixelDataR) != type(of: pixelDataG) || type(of: pixelDataR) != type(of: pixelDataB) {
            throw NSError(domain: "DataProviderError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Unmatched pixel data: RGB channels type"])
        }
        if pixelDataR.count != pixelDataG.count || pixelDataG.count != pixelDataB.count {
            throw NSError(domain: "DataProviderError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Unmatched pixel data: RGB channels number of pixels"])
        }
        guard let pixelCountWidthR = pixelDataR.first?.count, let pixelCountWidthG = pixelDataG.first?.count, let pixelCountWidthB = pixelDataB.first?.count else {
            throw NSError(domain: "DataProviderError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Unmatched pixel data: original data error"])
        }
        if pixelCountWidthR != pixelCountWidthG || pixelCountWidthG != pixelCountWidthB {
            throw NSError(domain: "DataProviderError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Unmatched pixel data: RGB channels number of pixels"])
        }
        if let pixelDataAlpha = pixelDataA {
            if pixelDataAlpha.count != pixelDataR.count {
                throw NSError(domain: "DataProviderError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Unmatched pixel data: Alpha channels number of pixels"])
            }
            guard let pixelCountA = pixelDataAlpha.first?.count else {
                throw NSError(domain: "DataProviderError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Unmatched pixel data: original data error"])
            }
            if pixelCountA != pixelCountWidthR {
                throw NSError(domain: "DataProviderError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Unmatched pixel data: Alpha channels number of pixels"])
            }
        }
        if let pixelDataRed = pixelDataR as? [[UInt8]], let pixelDataGreen = pixelDataG as? [[UInt8]], let pixelDataBlue = pixelDataB as? [[UInt8]]{
            if let pixelDataAlpha = pixelDataA as? [[UInt8]] {
                var redPremultiplied: UInt8
                var greenPremultiplied: UInt8
                var bluePremultiplied: UInt8
                for y in 0..<pixelDataR.count {
                    for x in 0..<pixelCountWidthR {
                        redPremultiplied = UInt8(Double(pixelDataRed[y][x]) * Double(pixelDataAlpha[y][x]) / 255.0)
                        greenPremultiplied = UInt8(Double(pixelDataGreen[y][x]) * Double(pixelDataAlpha[y][x]) / 255.0)
                        bluePremultiplied = UInt8(Double(pixelDataBlue[y][x]) * Double(pixelDataAlpha[y][x]) / 255.0)
                        dataWithOneDim.append(redPremultiplied)
                        dataWithOneDim.append(greenPremultiplied)
                        dataWithOneDim.append(bluePremultiplied)
                        dataWithOneDim.append(pixelDataAlpha[y][x])
                    }
                }
            } else {
                for y in 0..<pixelDataR.count {
                    for x in 0..<pixelCountWidthR {
                        dataWithOneDim.append(pixelDataRed[y][x])
                        dataWithOneDim.append(pixelDataGreen[y][x])
                        dataWithOneDim.append(pixelDataBlue[y][x])
                        dataWithOneDim.append(UInt8(0xFF))
                    }
                }
            }
        } else if let pixelDataRed = pixelDataR as? [[UInt16]], let pixelDataGreen = pixelDataG as? [[UInt16]], let pixelDataBlue = pixelDataB as? [[UInt16]]{
            if let pixelDataAlpha = pixelDataA as? [[UInt16]] {
                var redPremultiplied: UInt16
                var greenPremultiplied: UInt16
                var bluePremultiplied: UInt16
                for y in 0..<pixelDataR.count {
                    for x in 0..<pixelCountWidthR {
                        redPremultiplied = UInt16(Double(pixelDataRed[y][x]) * Double(pixelDataAlpha[y][x]) / 65535.0)
                        greenPremultiplied = UInt16(Double(pixelDataGreen[y][x]) * Double(pixelDataAlpha[y][x]) / 65535.0)
                        bluePremultiplied = UInt16(Double(pixelDataBlue[y][x]) * Double(pixelDataAlpha[y][x]) / 65535.0)
                        dataWithOneDim.append(UInt8(redPremultiplied >> 8 & 0xFF))
                        dataWithOneDim.append(UInt8(redPremultiplied & 0xFF))
                        dataWithOneDim.append(UInt8(greenPremultiplied >> 8 & 0xFF))
                        dataWithOneDim.append(UInt8(greenPremultiplied & 0xFF))
                        dataWithOneDim.append(UInt8(bluePremultiplied >> 8 & 0xFF))
                        dataWithOneDim.append(UInt8(bluePremultiplied & 0xFF))
                        dataWithOneDim.append(UInt8(pixelDataAlpha[y][x] >> 8 & 0xFF))
                        dataWithOneDim.append(UInt8(pixelDataAlpha[y][x] & 0xFF))
                    }
                }
            } else {
                for y in 0..<pixelDataR.count {
                    for x in 0..<pixelCountWidthR {
                        dataWithOneDim.append(UInt8(pixelDataRed[y][x] >> 8 & 0xFF))
                        dataWithOneDim.append(UInt8(pixelDataRed[y][x] & 0xFF))
                        dataWithOneDim.append(UInt8(pixelDataGreen[y][x] >> 8 & 0xFF))
                        dataWithOneDim.append(UInt8(pixelDataGreen[y][x] & 0xFF))
                        dataWithOneDim.append(UInt8(pixelDataBlue[y][x] >> 8 & 0xFF))
                        dataWithOneDim.append(UInt8(pixelDataBlue[y][x] & 0xFF))
                        dataWithOneDim.append(UInt8(0xFF))
                        dataWithOneDim.append(UInt8(0xFF))
                    }
                }
            }
        } else {
            throw NSError(domain: "DataProviderError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Unmatched pixel data: unsupported original data"])
        }
        let cfdata = Data(dataWithOneDim) as CFData
        guard let provider = CGDataProvider(data: cfdata) else {
            throw NSError(domain: "DataProviderError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGDataProvider."])
        }
        // for value in dataWithOneDim {
        //     print("\(String(format: "%02X", value))",terminator: " ")
        // }
        // print(dataWithOneDim.count)
        return provider
    }
}

func splitIntoRG16<T: BinaryInteger & FixedWidthInteger>(imageInput: [[T]]) -> (R: [[Int16]], G: [[Int16]], B: [[Int16]]) {
    var R: [[Int16]] = []
    var G: [[Int16]] = []
    var B: [[Int16]] = []
    for row in imageInput {
        var rowR: [Int16] = []
        var rowG: [Int16] = []
        var rowB: [Int16] = []
        for pixel in row {
            rowB.append(0)
            pixel < 0 ? rowR.append(Int16(pixel)) : rowR.append(0)
            pixel > 0 ? rowG.append(-(Int16(pixel))) : rowG.append(0)
        }
        R.append(rowR)
        G.append(rowG)
        B.append(rowB)
    }
    return (R, G, B)
}

func flatTo1DimFrom2Dim<T: BinaryInteger & FixedWidthInteger>(_ image: [[T]]) -> [T] {
    return image.flatMap { row in
        row.map { $0 }
    }
}

func changeToUnsigned<T: BinaryInteger & FixedWidthInteger>(_ image: [[T]]) -> [[T.Magnitude]] {
    return image.map { row in
        row.map { pixel in
            pixel.magnitude
        }
    }
}