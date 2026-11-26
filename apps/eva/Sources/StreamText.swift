import Foundation

/// Убирает `<think>…</think>` из потока, не теряя куски тегов на границах
/// чанков: модель выдаёт по нескольку символов, и тег может прийти по частям.
///
/// `visible` — то, что показываем и озвучиваем; `thinking` — рассуждение
/// (в окне оно свёрнуто, в озвучку не попадает никогда).
struct ThinkFilter {
    private(set) var visible = ""
    private(set) var thinking = ""

    private let openTag = "<think>"
    private let closeTag = "</think>"
    private var pending = ""
    private var inside = false

    /// Скормить очередной кусок, вернуть прирост видимого текста.
    mutating func feed(_ chunk: String) -> String {
        let before = visible.count
        pending += chunk
        drain()
        guard visible.count > before else { return "" }
        return String(visible.dropFirst(before))
    }

    /// Закрыть поток: остаток буфера уходит туда, где он сейчас уместен.
    mutating func flush() {
        if inside {
            thinking += pending
        } else {
            visible += pending
        }
        pending = ""
    }

    private mutating func drain() {
        while true {
            if inside {
                if let range = pending.range(of: closeTag) {
                    thinking += String(pending[..<range.lowerBound])
                    pending = String(pending[range.upperBound...])
                    inside = false
                    continue
                }
                let keep = Self.suffixPrefixLength(pending, of: closeTag)
                if pending.count > keep {
                    thinking += String(pending.dropLast(keep))
                    pending = String(pending.suffix(keep))
                }
                return
            }
            if let range = pending.range(of: openTag) {
                visible += String(pending[..<range.lowerBound])
                pending = String(pending[range.upperBound...])
                inside = true
                continue
            }
            let keep = Self.suffixPrefixLength(pending, of: openTag)
            if pending.count > keep {
                visible += String(pending.dropLast(keep))
                pending = String(pending.suffix(keep))
            }
            return
        }
    }

    /// Длина самого длинного суффикса `text`, который является началом `tag`
    /// (короче самого тега — иначе это уже совпадение, его обрабатывает range).
    private static func suffixPrefixLength(_ text: String, of tag: String) -> Int {
        let limit = min(tag.count - 1, text.count)
        guard limit > 0 else { return 0 }
        var k = limit
        while k > 0 {
            if text.hasSuffix(tag.prefix(k)) { return k }
            k -= 1
        }
        return 0
    }
}

/// Фраза для `/api/tts` и пауза, которая должна прозвучать после неё
/// (задача 0230). `paragraph` — фраза закрывала абзац: сам текст уже обрезан,
/// поэтому пустую строку в нём не найти, флаг помнит нарезка.
struct SpokenPhrase: Equatable, Sendable {
    let text: String
    let paragraph: Bool
}

/// Режет текст ответа на фразы, чтобы отправлять их в `/api/tts` по мере
/// прихода, а не ждать конца ответа. Длинные фразы — по словам, у сервера
/// лимит 600 символов на запрос.
struct PhraseSplitter {
    /// Фраза, которую нельзя отдать сразу: в конце текущего куска текста был
    /// один перевод строки, а следующий кусок может добавить второй (абзац).
    /// Держим её до следующего `feedPhrases`/`flushPhrases`.
    private struct DeferredPhrase {
        let text: String
        /// Сколько переводов строк уже увидели после фразы.
        let newlines: Int
    }

    private(set) var maxChars = 560
    private var buffer = ""
    private var deferred: DeferredPhrase?

    /// Сколько символов ещё не стало фразой (лежит в буфере). Нужно позиции
    /// картинки (задача 0227): событие `image` может прийти в середине фразы,
    /// и до её конца речь ещё не дойдёт — картинку нельзя показывать раньше.
    /// Придержанная фраза тоже в счёт: она уже не в буфере, но ещё не ушла
    /// в `/api/tts`.
    var pendingChars: Int {
        let waiting = deferred?.text.count ?? 0
        return waiting + buffer.trimmingCharacters(in: .whitespacesAndNewlines).count
    }

    /// Только текст фраз — прежний контракт для `--selftest-text` и мест,
    /// которым паузы не нужны.
    mutating func feed(_ text: String) -> [String] {
        feedPhrases(text).map(\.text)
    }

    /// Фразы вместе с признаком «за фразой был абзац» (задача 0230).
    mutating func feedPhrases(_ text: String) -> [SpokenPhrase] {
        buffer += text
        var phrases = takeDeferred()
        phrases += takeComplete()
        return phrases
    }

    mutating func flush() -> [String] {
        flushPhrases().map(\.text)
    }

    mutating func flushPhrases() -> [SpokenPhrase] {
        var phrases = takeDeferred()
        // Хвост ответа мог кончиться пустой строкой — это тоже абзац.
        let paragraph = ReadingPause.isParagraph(buffer)
        let rest = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        buffer = ""
        guard !rest.isEmpty else { return phrases }
        let chunks = Self.chunks(rest, max: maxChars).filter(Self.isSpeakable)
        for (index, text) in chunks.enumerated() {
            phrases.append(SpokenPhrase(
                text: text,
                paragraph: paragraph && index == chunks.count - 1
            ))
        }
        return phrases
    }

    /// Отдать придержанную фразу: следующий кусок текста показал, был ли за
    /// ней абзац (второй перевод строки) или обычный конец фразы.
    private mutating func takeDeferred() -> [SpokenPhrase] {
        guard let waiting = deferred else { return [] }
        deferred = nil
        let newlines = waiting.newlines + Self.leadingNewlines(buffer)
        return [SpokenPhrase(
            text: waiting.text,
            paragraph: newlines >= 2
        )]
    }

    private mutating func takeComplete() -> [SpokenPhrase] {
        var phrases: [SpokenPhrase] = []
        while let cut = Self.boundary(buffer) {
            let segment = buffer[..<cut]
            let rest = buffer[cut...]
            let newlines = Self.newlines(segment: segment, rest: rest)
            let paragraph = newlines >= 2
            let phrase = String(segment).trimmingCharacters(in: .whitespacesAndNewlines)
            let restText = String(rest)
            buffer = restText
            guard !phrase.isEmpty else { continue }
            let chunks = Self.chunks(phrase, max: maxChars)
            // Конец куска текста: следующий delta может начаться с ещё одного
            // перевода строки и сделать эту фразу концом абзаца. Последнюю
            // фразу куска придерживаем до следующего delta или `flush`.
            if !paragraph, restText.allSatisfy({ $0.isWhitespace }) {
                buffer = ""
                for (index, text) in chunks.enumerated() {
                    if index == chunks.count - 1 {
                        deferred = DeferredPhrase(text: text, newlines: newlines)
                    } else {
                        phrases.append(SpokenPhrase(text: text, paragraph: false))
                    }
                }
                break
            }
            for (index, text) in chunks.enumerated() {
                phrases.append(SpokenPhrase(
                    text: text,
                    paragraph: paragraph && index == chunks.count - 1
                ))
            }
        }
        // Текст без точек (список, код) иначе молчал бы до конца ответа.
        if buffer.count > maxChars * 2 {
            let paragraph = ReadingPause.isParagraph(buffer)
            let text = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
            buffer = ""
            let chunks = Self.chunks(text, max: maxChars)
            for (index, chunk) in chunks.enumerated() {
                phrases.append(SpokenPhrase(
                    text: chunk,
                    paragraph: paragraph && index == chunks.count - 1
                ))
            }
        }
        return phrases.filter { Self.isSpeakable($0.text) }
    }

    /// Сколько переводов строк на границе разреза: в хвостовых пробелах
    /// вырезанного куска и ведущих пробелах остатка. Разрез может пройти по
    /// любому из них, а абзац — это минимум две пустые строки подряд.
    private static func newlines(segment: Substring, rest: Substring) -> Int {
        var count = 0
        for character in segment.reversed() {
            guard character.isWhitespace else { break }
            if character == "\n" { count += 1 }
        }
        return count + leadingNewlines(rest)
    }

    /// Переводы строк в ведущих пробелах текста.
    private static func leadingNewlines<S: StringProtocol>(_ text: S) -> Int {
        var count = 0
        for character in text {
            guard character.isWhitespace else { break }
            if character == "\n" { count += 1 }
        }
        return count
    }

    /// Конец первого предложения: терминатор, за которым пробел/перевод строки.
    private static func boundary(_ text: String) -> String.Index? {
        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]
            if character == "\n" {
                return text.index(after: index)
            }
            if ".!?…".contains(character) {
                let next = text.index(after: index)
                if next < text.endIndex, text[next] == " " || text[next] == "\n" {
                    return text.index(after: next)
                }
            }
            index = text.index(after: index)
        }
        return nil
    }

    private static func chunks(_ text: String, max: Int) -> [String] {
        guard text.count > max else { return [text] }
        var out: [String] = []
        var current = ""
        for word in text.split(separator: " ", omittingEmptySubsequences: true) {
            if current.isEmpty {
                current = String(word)
            } else if current.count + 1 + word.count <= max {
                current += " " + word
            } else {
                out.append(current)
                current = String(word)
            }
            while current.count > max {
                out.append(String(current.prefix(max)))
                current = String(current.dropFirst(max))
            }
        }
        if !current.isEmpty { out.append(current) }
        return out
    }

    private static func isSpeakable(_ text: String) -> Bool {
        text.contains { $0.isLetter || $0.isNumber }
    }
}
