import Foundation

/// Какая тишина идёт после фразы: обычная или «абзац закончился».
enum ReadingBreak: Equatable, Sendable {
    case phrase
    case paragraph
}

/// Паузы между фразами озвучки (задача 0230): ребёнку нужно медленнее.
///
/// База — 0,35 с между фразами и 0,9 с после абзаца. В сказке (ответ плагина,
/// картинки или `finish_reason=plugin`) на настройке «медленная» паузы
/// умножаются на 1,8: ≈0,63 с и ≈1,62 с. Обычный чат читается по базе при
/// любой настройке — она относится только к сказкам.
///
/// Расчёт — чистые `nonisolated` функции: их проверяет `--selftest-pauses`
/// без звука, сети и реального ожидания.
enum ReadingPause {
    /// Тишина после обычной фразы.
    static let phrase: TimeInterval = 0.35
    /// Тишина после абзаца: заметно длиннее, чтобы ребёнок успел перевести дух.
    static let paragraph: TimeInterval = 0.9
    /// Во сколько раз паузы длиннее в сказке на «медленной» настройке.
    static let taleFactor: Double = 1.8

    /// Конец абзаца по тексту: в хвосте пустая строка (`\n\n`, в том числе
    /// `\r\n\r\n`). `PhraseSplitter` обрезает переводы строк, поэтому отдаёт
    /// абзац отдельным флагом; текстовая проверка нужна `--selftest-pauses`
    /// и вызовам без флага.
    nonisolated static func isParagraph(_ text: String) -> Bool {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        let tail = normalized.reversed().prefix { $0.isWhitespace }
        return tail.filter { $0 == "\n" }.count >= 2
    }

    /// Вид паузы по тексту фразы.
    nonisolated static func kind(for text: String) -> ReadingBreak {
        isParagraph(text) ? .paragraph : .phrase
    }

    /// Длительность паузы: база по виду, множитель — только для сказки на
    /// «медленной» настройке. «Обычная» сказка читается как чат (×1,0).
    nonisolated static func duration(
        _ kind: ReadingBreak,
        tale: Bool,
        slowTale: Bool
    ) -> TimeInterval {
        let base = kind == .paragraph ? paragraph : phrase
        return base * (tale && slowTale ? taleFactor : 1)
    }

    /// То же по флагу от `PhraseSplitter`.
    nonisolated static func duration(
        paragraph: Bool,
        tale: Bool,
        slowTale: Bool
    ) -> TimeInterval {
        duration(paragraph ? .paragraph : .phrase, tale: tale, slowTale: slowTale)
    }

    /// То же по тексту фразы.
    nonisolated static func duration(
        text: String,
        tale: Bool,
        slowTale: Bool
    ) -> TimeInterval {
        duration(kind(for: text), tale: tale, slowTale: slowTale)
    }
}

/// Ожидание между фразами голоса. Одна пауза за раз: новая отменяет прежнюю,
/// `cancel()` (стоп, новый вопрос) гасит её мгновенно. `onDone` зовётся после
/// сна на главном акторе, только если паузу не отменили и не заменили.
@MainActor
final class PauseGate {
    private var task: Task<Void, Never>?

    /// Пауза идёт: очередь ещё не опустела, голос не смолк.
    var isWaiting: Bool { task != nil }

    /// Запустить паузу `seconds`; `seconds <= 0` — сразу `onDone`.
    func schedule(seconds: TimeInterval, onDone: @escaping @MainActor () -> Void) {
        cancel()
        guard seconds > 0 else {
            onDone()
            return
        }
        let nanoseconds = UInt64((seconds * 1_000_000_000).rounded())
        task = Task { [weak self] in
            try? await Task.sleep(nanoseconds: nanoseconds)
            guard let self, !Task.isCancelled, self.task != nil else { return }
            self.task = nil
            onDone()
        }
    }

    /// Отменить паузу немедленно: `onDone` не сработает.
    func cancel() {
        task?.cancel()
        task = nil
    }
}
