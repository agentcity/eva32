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

/// Режет текст ответа на фразы, чтобы отправлять их в `/api/tts` по мере
/// прихода, а не ждать конца ответа. Длинные фразы — по словам, у сервера
/// лимит 600 символов на запрос.
struct PhraseSplitter {
    private(set) var maxChars = 560
    private var buffer = ""

    /// Сколько символов ещё не стало фразой (лежит в буфере). Нужно позиции
    /// картинки (задача 0227): событие `image` может прийти в середине фразы,
    /// и до её конца речь ещё не дойдёт — картинку нельзя показывать раньше.
    var pendingChars: Int {
        buffer.trimmingCharacters(in: .whitespacesAndNewlines).count
    }

    mutating func feed(_ text: String) -> [String] {
        buffer += text
        return takeComplete()
    }

    mutating func flush() -> [String] {
        let rest = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        buffer = ""
        guard !rest.isEmpty else { return [] }
        return Self.chunks(rest, max: maxChars).filter(Self.isSpeakable)
    }

    private mutating func takeComplete() -> [String] {
        var phrases: [String] = []
        while let cut = Self.boundary(buffer) {
            let phrase = String(buffer[..<cut]).trimmingCharacters(in: .whitespacesAndNewlines)
            buffer = String(buffer[cut...])
            if !phrase.isEmpty { phrases += Self.chunks(phrase, max: maxChars) }
        }
        // Текст без точек (список, код) иначе молчал бы до конца ответа.
        if buffer.count > maxChars * 2 {
            let text = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
            buffer = ""
            phrases += Self.chunks(text, max: maxChars)
        }
        return phrases.filter(Self.isSpeakable)
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
