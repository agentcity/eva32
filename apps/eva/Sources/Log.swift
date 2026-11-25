import Foundation

/// Журнал приложения: stderr + `~/Library/Logs/EVA/eva.log`.
///
/// EVA.app запускается из статус-бара, без терминала, поэтому файл — единственное
/// место, где видно, что происходило: старт, запрос к `/api/chat`, куски SSE,
/// озвучка, ошибки разрешений. Писать можно из любого потока.
final class LogSink: @unchecked Sendable {
    private let lock = NSLock()
    private let handle: FileHandle?

    init() {
        let fm = FileManager.default
        guard let library = fm.urls(for: .libraryDirectory, in: .userDomainMask).first else {
            handle = nil
            return
        }
        let directory = library.appendingPathComponent("Logs/EVA", isDirectory: true)
        try? fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("eva.log")
        if !fm.fileExists(atPath: url.path) {
            fm.createFile(atPath: url.path, contents: nil)
        }
        handle = try? FileHandle(forWritingTo: url)
        handle?.seekToEndOfFile()
    }

    func write(_ text: String) {
        lock.lock()
        defer { lock.unlock() }
        let data = Data(text.utf8)
        FileHandle.standardError.write(data)
        handle?.write(data)
    }
}

enum EvaLog {
    static let sink = LogSink()

    static func write(_ message: String) {
        // Время — местное: журнал читает человек, а не парсер.
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        formatter.timeZone = TimeZone.current
        sink.write("[\(formatter.string(from: Date()))] \(message)\n")
    }
}

/// Короткая запись в журнал (см. `EvaLog`).
func log(_ message: String) {
    EvaLog.write(message)
}
