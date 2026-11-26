import Foundation

/// Аргументы запуска. Нужны для проверки без мышки и для скриптов:
///
/// * `--ask "текст"` — сразу отправить вопрос в чат и показать окно;
/// * `--no-voice` — не озвучивать ответ (проверка без звука);
/// * `--server URL`, `--model backend|id` — временно переопределить настройки;
/// * `--selftest-chat "текст"` — прогнать поток `/api/chat` без UI;
/// * `--selftest-tts "текст"` — получить WAV из `/api/tts`;
/// * `--selftest-backends` — распечатать реестр моделей сервера;
/// * `--selftest-speech` — права и офлайн-доступность Apple Speech;
/// * `--selftest-text` — скрытие `<think>` и нарезка фраз для озвучки;
/// * `--selftest-wake` — триггер «Привет, Ева» и кольцевой буфер (без микрофона);
/// * `--selftest-followup` — пауза тишины follow-up и правило «только вместе с
///   ожиданием „Привет, Ева“» (без микрофона);
/// * `--selftest-stop` — матчер «Стоп, Ева», эхо-защита, план слушателей и
///   настройка (без микрофона);
/// * `--selftest-wake-audio N` — N секунд держать микрофон и кольцевой буфер
///   без распознавания: замер CPU и проверка остановки `AVAudioEngine`.
/// * `--selftest-compact` — компактный круг: геометрия, живое окно
///   (CGWindowList), `visibleFrame`, позиция в UserDefaults (задача 0188).
enum Launch {
    static let arguments = CommandLine.arguments

    static func value(_ flag: String) -> String? {
        guard let index = arguments.firstIndex(of: flag), index + 1 < arguments.count else {
            return nil
        }
        return arguments[index + 1]
    }

    static let ask = value("--ask")
    /// `--no-voice` — не озвучивать (сценарии проверки, чтобы не шуметь).
    static let noVoice = arguments.contains("--no-voice")
    static let serverOverride = value("--server")
    static let modelOverride = value("--model")

    /// Режим самопроверки; `nil` — обычный запуск приложения.
    static var selftest: SelfTest.Mode? {
        if let question = value("--selftest-chat") { return .chat(question) }
        if let text = value("--selftest-tts") { return .tts(text) }
        if arguments.contains("--selftest-backends") { return .backends }
        if arguments.contains("--selftest-speech") { return .speech }
        if arguments.contains("--selftest-text") { return .text }
        if arguments.contains("--selftest-wake") { return .wake }
        if arguments.contains("--selftest-followup") { return .followUp }
        if arguments.contains("--selftest-stop") { return .stop }
        if arguments.contains("--selftest-compact") { return .compact }
        if let raw = value("--selftest-wake-audio") {
            return .wakeAudio(Double(raw) ?? 10)
        }
        if arguments.contains("--version") { return .version }
        return nil
    }
}
