import Foundation

/// Связывает поток ответа с окном и озвучкой: копит видимый текст, скрывает
/// размышления, режет ответ на фразы для `/api/tts`.
///
/// `runID` защищает от запоздавших событий: после «Стоп» или нового вопроса
/// старый поток может прислать ещё несколько `delta`, их надо выбросить.
@MainActor
final class AnswerController {
    private let chat = ChatSession()
    private let voice = VoicePlayer()
    private var filter = ThinkFilter()
    private var splitter = PhraseSplitter()
    private var runID = 0
    private var speak = false
    private var hadError = false
    private var stats = ""
    private var startedAt = Date()
    private var deltaCount = 0

    var onStatus: ((String, Bool) -> Void)?
    var onAnswer: ((String, String) -> Void)?
    var onSpeakingChanged: ((Bool) -> Void)?
    /// Громкость озвучки 0…1 — для большого круга.
    var onLevel: ((Float) -> Void)?
    /// Поток ответа закончился (успешно или с ошибкой): круг перестаёт «думать».
    var onFinished: (() -> Void)?
    /// Озвучка доиграла всё, что было в очереди: в follow-up можно открывать
    /// микрофон (задача 0190). Раньше этого делать нельзя — перебьём голос.
    var onVoiceDrained: (() -> Void)?

    /// Нечего играть и нечего синтезировать прямо сейчас.
    var voiceIsIdle: Bool { voice.isDrained }

    /// Ответ этого запуска закончился ошибкой (сеть, HTTP, модель). Follow-up
    /// после ошибки не открываем: иначе красная вспышка и текст ошибки будут
    /// мгновенно перекрыты «слушаю…».
    var didFail: Bool { hadError }

    init() {
        voice.onIssue = { [weak self] message in
            self?.onStatus?(message, true)
        }
        voice.onSpeakingChanged = { [weak self] speaking in
            self?.onSpeakingChanged?(speaking)
        }
        voice.onLevel = { [weak self] level in
            self?.onLevel?(level)
        }
        voice.onDrained = { [weak self] in
            self?.onVoiceDrained?()
        }
    }

    func ask(_ question: String, config: ChatConfig, speak enabled: Bool, voiceID: String) {
        runID += 1
        let run = runID
        chat.cancel()
        voice.stop()
        voice.configure(server: config.serverURL, voice: voiceID)
        filter = ThinkFilter()
        splitter = PhraseSplitter()
        speak = enabled
        hadError = false
        stats = ""
        startedAt = Date()
        deltaCount = 0
        onAnswer?("", "")
        onStatus?("думаю…", false)

        chat.send(question: question, config: config) { [weak self] event in
            guard let self, self.runID == run else { return }
            switch event {
            case .started(let label, let cold):
                var text = "отвечает: \(label)"
                if cold == true { text += " · холодный старт" }
                self.onStatus?(text, false)
                log("sse start: \(label) cold=\(cold.map(String.init) ?? "—")")
            case .delta(let chunk):
                self.deltaCount += 1
                if self.deltaCount == 1 {
                    log("sse: первый delta через "
                        + String(format: "%.2f", Date().timeIntervalSince(self.startedAt))
                        + " с")
                }
                let visible = self.filter.feed(chunk)
                self.publish(visible: self.filter.visible, thinking: self.filter.thinking)
                if self.speak, !visible.isEmpty {
                    self.voice.enqueue(self.splitter.feed(visible))
                }
            case .stats(let line):
                self.stats = line
                log("sse done: \(line)")
            case .failed(let message):
                self.hadError = true
                self.onStatus?("ошибка: \(message)", true)
                log("sse ошибка: \(message)")
            case .finished:
                self.filter.flush()
                self.publish(visible: self.filter.visible, thinking: self.filter.thinking)
                if self.speak {
                    self.voice.enqueue(self.splitter.flush())
                }
                log("sse finished: кусков \(self.deltaCount), видимых символов "
                    + "\(self.filter.visible.count), размышления "
                    + "\(self.filter.thinking.count)")
                log("ответ: \(self.filter.visible.prefix(400))")
                self.onFinished?()
                self.finishStatus()
            }
        }
    }

    func stop() {
        runID += 1
        chat.cancel()
        voice.stop()
        onStatus?("остановлено", false)
    }

    /// Только звук: выключили галочку «Озвучивать» — генерацию не трогаем.
    func stopVoice() {
        voice.stop()
    }

    private func publish(visible: String, thinking: String) {
        onAnswer?(visible, thinking)
    }

    private func finishStatus() {
        guard !hadError else { return }
        let seconds = Date().timeIntervalSince(startedAt)
        var text = "готово · " + String(format: "%.1f с", seconds)
        if !stats.isEmpty { text += " · " + stats }
        if filter.visible.isEmpty {
            onStatus?("модель промолчала — пустой ответ", true)
            return
        }
        onStatus?(text, false)
    }
}
