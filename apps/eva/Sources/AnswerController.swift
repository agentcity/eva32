import AppKit
import Foundation

/// Связывает поток ответа с окном и озвучкой: копит видимый текст, скрывает
/// размышления, режет ответ на фразы для `/api/tts`, принимает картинки сказки
/// и признак `silent` (ответ уже озвучен записью на сервере).
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
    /// Ответ уже озвучен записью на сервере (0223): в `/api/tts` не отправляем
    /// ни одной фразы, иначе сказку читают два голоса (задача 0224).
    private var silent = false
    private var hadError = false
    private var stats = ""
    private var startedAt = Date()
    private var deltaCount = 0
    private var imageCount = 0
    /// Базовый адрес сервера этого ответа: по нему собираются URL картинок.
    private var baseURL: URL?

    // Порядок показа картинок. Загрузка идёт параллельно, поэтому быстрая
    // картинка могла бы обогнать медленную и остаться на экране «последней».
    // События нумеруются, а показываются строго по порядку: `imageResolved` —
    // какие номера уже готовы (или не загрузились), `imageReady` — декодирован-
    // ные, `imageOrderShown` — до какого номера очередь дошла.
    private var imageResolved: Set<Int> = []
    private var imageReady: [Int: (image: NSImage, alt: String)] = [:]
    private var imageOrderShown = 0

    var onStatus: ((String, Bool) -> Void)?
    var onAnswer: ((String, String) -> Void)?
    var onSpeakingChanged: ((Bool) -> Void)?
    /// Громкость озвучки 0…1 — для большого круга.
    var onLevel: ((Float) -> Void)?
    /// Картинка сказки готова к показу (скачана и декодирована). Второй
    /// аргумент — подпись для всплывающей подсказки. Показ картинок не меняет
    /// состояние круга: это не озвучка.
    var onImage: ((NSImage, String) -> Void)?
    /// Поток ответа закончился (успешно или с ошибкой): круг перестаёт «думать».
    var onFinished: (() -> Void)?
    /// Озвучка доиграла всё, что было в очереди: в follow-up можно открывать
    /// микрофон (задача 0190). Раньше этого делать нельзя — перебьём голос.
    var onVoiceDrained: (() -> Void)?

    /// Нечего играть и нечего синтезировать прямо сейчас.
    var voiceIsIdle: Bool { voice.isDrained }

    /// Ответ озвучен записью на сервере: приложение не должно синтезировать
    /// текст. Нужно `AppDelegate`, чтобы круг не ждал несуществующий звук.
    var isSilentAnswer: Bool { silent }

    /// Текст фразы, которая звучит из колонок прямо сейчас (nil — тишина).
    /// Нужен стоп-слушателю для эхо-защиты: свою же реплику со «стоп» он
    /// должен игнорировать, а команду человека — нет (задача 0194).
    var currentVoiceText: String? { voice.currentText }

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

    /// Отправлять ли фразу в `/api/tts`. Вынесено из `ask`, чтобы правило
    /// «silent → ни звука, текст показываем» проверялось в `--selftest-images`
    /// без сервера и без звука.
    nonisolated static func shouldSpeak(voiceOn: Bool, silent: Bool) -> Bool {
        voiceOn && !silent
    }

    /// URL картинки, которую можно показать: настройка включена и адрес
    /// годный. `nil` — событие `image` пропускаем, ответ продолжается.
    /// Вынесено из потока по той же причине, что и `shouldSpeak`.
    nonisolated static func imageURL(
        setting: Bool,
        url: String,
        base: URL?
    ) -> URL? {
        guard setting else { return nil }
        return TaleImageRef(url: url, alt: "").resolved(base: base)
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
        silent = false
        baseURL = config.serverURL
        imageCount = 0
        imageResolved.removeAll()
        imageReady.removeAll()
        imageOrderShown = 0
        if enabled {
            voice.beginStream()
        }
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
            case .silent(let flag):
                guard flag, !self.silent else { break }
                self.silent = true
                // Требование задачи: в логе должно быть видно, что голос не
                // включился, а не что он «не смог».
                log("озвучка: пропущена (silent) — ответ уже озвучен записью на сервере")
            case .delta(let chunk):
                self.deltaCount += 1
                if self.deltaCount == 1 {
                    log("sse: первый delta через "
                        + String(format: "%.2f", Date().timeIntervalSince(self.startedAt))
                        + " с")
                }
                let visible = self.filter.feed(chunk)
                self.publish(visible: self.filter.visible, thinking: self.filter.thinking)
                if Self.shouldSpeak(voiceOn: self.speak, silent: self.silent),
                   !visible.isEmpty
                {
                    self.voice.enqueue(self.splitter.feed(visible))
                }
            case .image(let url, let alt):
                self.imageCount += 1
                let order = self.imageCount
                log("sse image: \(url) alt=\(alt.prefix(80))")
                guard let target = Self.imageURL(
                    setting: Settings.shared.showTaleImages,
                    url: url,
                    base: self.baseURL
                ) else {
                    log("картинка: пропущена — "
                        + (Settings.shared.showTaleImages
                            ? "непонятный URL \(url)"
                            : "настройка «Показывать картинки сказок» выключена"))
                    // Место в очереди всё равно закрываем: иначе картинки,
                    // пришедшие после пропущенной, не покажутся вовсе.
                    self.resolveImage(order: order)
                    break
                }
                self.loadImage(target, alt: alt, run: run, order: order)
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
                    if self.silent {
                        log("озвучка: пропущена (silent) — в /api/tts не отправлено "
                            + "ни одной фразы")
                    } else {
                        self.voice.enqueue(self.splitter.flush())
                    }
                    // Только теперь очередь может объявить «досказали»:
                    // иначе follow-up открывал микрофон посреди длинной
                    // сказки и последняя фраза обрывалась. При silent очередь
                    // пуста сразу, и follow-up открывается на этом вызове.
                    self.voice.endStream()
                }
                log("sse finished: кусков \(self.deltaCount), видимых символов "
                    + "\(self.filter.visible.count), размышления "
                    + "\(self.filter.thinking.count), картинок \(self.imageCount)"
                    + (self.silent ? ", озвучка пропущена (silent)" : ""))
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

    // MARK: - Картинки сказки

    /// Скачать картинку в фоне и поставить её в очередь показа. Ошибка
    /// загрузки — просто нет картинки: текст ответа и озвучка идут дальше.
    /// `run` отсекает результат, пришедший после «Стоп» или нового вопроса.
    private func loadImage(_ url: URL, alt: String, run: Int, order: Int) {
        Task { [weak self] in
            let data = await TaleImageLoader.load(url)
            guard let self, self.runID == run else { return }
            // Настройку могли снять, пока картинка качалась.
            if let data, Settings.shared.showTaleImages {
                if let image = NSImage(data: data) {
                    self.imageReady[order] = (image, alt)
                } else {
                    log("картинка: не декодировалась \(url.lastPathComponent)")
                }
            } else if data != nil {
                log("картинка: не показываю — настройка выключена во время загрузки")
            }
            self.resolveImage(order: order)
        }
    }

    /// Отметить номер картинки готовым и показать всё, что уже можно показать
    /// по порядку (пропуски из-за ошибок тоже считаются готовыми).
    private func resolveImage(order: Int) {
        imageResolved.insert(order)
        while imageResolved.contains(imageOrderShown + 1) {
            imageOrderShown += 1
            guard let ready = imageReady.removeValue(forKey: imageOrderShown) else {
                continue
            }
            log("картинка: показываю \(ready.alt.isEmpty ? "иллюстрацию" : ready.alt) "
                + "(№\(imageOrderShown))")
            onImage?(ready.image, ready.alt)
        }
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
