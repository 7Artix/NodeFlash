import Foundation

class Database {
    enum DatabaseType {
        case Images
        case Labels
    }

    var imagesPath: String
    var labelsPath: String

    var imagesData: Data
    var labelsData: Data

    init(imagesPath: String, labelsPath: String) {
        self.imagesPath = imagesPath
        self.labelsPath = labelsPath
        do {
            self.imagesData = try Data(contentsOf: URL(fileURLWithPath: imagesPath))
            self.labelsData = try Data(contentsOf: URL(fileURLWithPath: labelsPath))
        } catch {
            fatalError("Error: Load failed.")
        }
    }

    func setPath(for target: DatabaseType, with path: String) {
        switch target {
        case .Images:
            self.imagesPath = path
        case .Labels:
            self.labelsPath = path
        }
    }

    func reloadDatabase() {
        do {
            self.imagesData = try Data(contentsOf: URL(fileURLWithPath: self.imagesPath))
            self.labelsData = try Data(contentsOf: URL(fileURLWithPath: self.labelsPath))
        } catch {
            fatalError("Error: Load failed.")
        }
    }
}
