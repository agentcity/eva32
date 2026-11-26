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
    /// Ответ читается как сказка: паузы длиннее (задача 0230). Признаки —
    /// картинки плагина, длина ответа или `finish_reason=plugin`.
    private var tale = false
    /// Базовый адрес сервера этого ответа: по нему собираются URL картинок.
    private var baseURL: URL?

    // Показ картинок. События `image` приходят почти сразу вместе с текстом,
    // поэтому показывать в момент прихода нельзя: видна только последняя.
    // `pacer` помнит позицию каждой картинки (после скольких сказанных
    // символов её показывать), `imageResolved` — какие номера уже готовы (или
    // не загрузились), `imageReady` — декодированные. Показ строго по порядку
    // событий, даже если файлы скачались вразнобой.
    private var pacer = TaleImagePacer(mode: .timer)
    private var imageResolved: Set<Int> = []
    private var imageReady: [Int: (image: NSImage, alt: String)] = [:]
    /// Таймер оценочного темпа: работает, когда озвучки нет (выключена,
    /// `silent` или TTS недоступен). В режиме озвучки прогресс приходит из
    /// `VoicePlayer.onProgress`.
    private var imageTimer: Timer?
    private var imageTimerStart = Date()
    private var imageTimerBase = 0.0
    /// Поток ответа уже закрыт: новые картинки после этого не придут, можно
    /// погасить таймер, когда показана последняя.
    private var streamEnded = false

    var onStatus: ((String, Bool) -> Void)?
    var onAnswer: ((String, String) -> Void)?
    var onSpeakingChanged: ((Bool) -> Void)?
    /// Громкость озвучки 0…1 — для большого круга.
    var onLevel: ((Float) -> Void)?
    /// Картинка сказки готова к показу (скачана и декодирована). Второй
    /// аргумент — подпись для всплывающей подсказки. Показ картинок не меняет
    /// состояние круга: это не озвучка.
    var onImage: ((NSImage, String) -> Void)?
    /// Убрать иллюстрацию: «Стоп» или новый вопрос. Очередь и таймеры к этому
    /// моменту уже погашены.
    var onImagesCleared: (() -> Void)?
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
            guard let self else { return }
            self.voiceUnavailable()
            self.onStatus?(message, true)
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
        // Прогресс озвучки: сколько символов уже сказано. По нему показываем
        // очередную картинку — а не по моменту прихода события `image`.
        voice.onProgress = { [weak self] spoken, _ in
            self?.voiceDidSpeak(spoken)
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

    /// Ответ считать сказкой (задача 0230): длинная проза плагина или его
    /// явные признаки — картинки и `finish_reason=plugin`. Порог 400 символов
    /// выбран между обычным ответом (десятки–сотни символов) и сказкой
    /// (900+), чтобы короткий чат читался с короткими паузами. Вынесено
    /// `nonisolated`, чтобы правило проверял `--selftest-pauses`.
    nonisolated static func isTaleAnswer(
        images: Int,
        chars: Int,
        plugin: Bool
    ) -> Bool {
        plugin || images > 0 || chars >= 400
    }

    func ask(_ question: String, config: ChatConfig, speak enabled: Bool, voiceID: String) {
        runID += 1
        let run = runID
        chat.cancel()
        // Очередь и таймеры прошлого ответа гасим до `voice.stop()`: его
        // `onProgress(0, 0)` не должен показать картинку на прощание.
        resetImagePacing(mode: enabled ? .speech : .timer)
        onImagesCleared?()
        voice.stop()
        voice.configure(server: config.serverURL, voice: voiceID)
        filter = ThinkFilter()
        splitter = PhraseSplitter()
        speak = enabled
        silent = false
        tale = false
        baseURL = config.serverURL
        startedAt = Date()
        if enabled {
            voice.beginStream()
        } else {
            // Озвучки нет вовсе: картинки меняются по оценке 15 симв/с от
            // начала ответа (требование 0227).
            beginTimerPacing(from: startedAt, base: 0)
        }
        hadError = false
        stats = ""
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
                // «Тихий» ответ озвучен записью на сервере (`afplay`): своей
                // озвучки нет, поэтому картинки ведёт таймер от начала
                // внешней записи. Сервер запускает `afplay` внутри этого же
                // запроса, поэтому его начало — это `startedAt` ответа.
                self.beginTimerPacing(from: self.startedAt, base: 0)
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
                if !self.tale,
                   Self.isTaleAnswer(
                       images: self.imageCount,
                       chars: self.filter.visible.count,
                       plugin: false
                   )
                {
                    self.enableTaleMode("длинный ответ")
                }
                if Self.shouldSpeak(voiceOn: self.speak, silent: self.silent),
                   !visible.isEmpty
                {
                    self.voice.enqueue(self.splitter.feedPhrases(visible))
                }
            case .image(let url, let alt):
                self.imageCount += 1
                // Картинки присылает плагин сказок — это самый ранний и
                // надёжный признак, что ответ читается медленнее (0230).
                self.enableTaleMode("картинки сказки")
                let order = self.imageCount
                // Позиция в ответе: сколько символов уже показано к этому
                // событию (для таймера) и сколько символов реально ушло в
                // озвучку (для прогресса `VoicePlayer`, пробелы не в счёт).
                // Сервер режет `delta` ровно по `after_chars`, поэтому к
                // моменту `image` весь текст до неё уже учтён.
                let afterVisible = self.filter.visible.count
                let afterVoice = self.voice.enqueuedChars + self.splitter.pendingChars
                self.pacer.enqueue(TaleImagePacer.Item(
                    order: order,
                    afterVisible: afterVisible,
                    afterVoice: afterVoice
                ))
                log("sse image: \(url) alt=\(alt.prefix(80)) "
                    + "(после \(afterVisible) симв., озвучка \(afterVoice))")
                self.advanceImages()
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
            case .pluginAnswer:
                // Сервер пометил источник в `done`: ответ собрал плагин,
                // даже если картинок в нём нет (0230).
                self.enableTaleMode("источник plugin")
            case .stats(let line):
                self.stats = line
                log("sse done: \(line)")
            case .failed(let message):
                self.hadError = true
                self.onStatus?("ошибка: \(message)", true)
                log("sse ошибка: \(message)")
            case .finished:
                self.streamEnded = true
                self.filter.flush()
                self.publish(visible: self.filter.visible, thinking: self.filter.thinking)
                if self.speak {
                    if self.silent {
                        log("озвучка: пропущена (silent) — в /api/tts не отправлено "
                            + "ни одной фразы")
                    } else {
                        self.voice.enqueue(self.splitter.flushPhrases())
                    }
                    // Только теперь очередь может объявить «досказали»:
                    // иначе follow-up открывал микрофон посреди длинной
                    // сказки и последняя фраза обрывалась. При silent очередь
                    // пуста сразу, и follow-up открывается на этом вызове.
                    self.voice.endStream()
                }
                // Все image-события уже пришли: если в режиме таймера
                // показывать больше нечего, часы можно погасить. Последняя
                // картинка на экране остаётся до нового вопроса.
                if self.pacer.next == nil {
                    self.stopImageTimer()
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
        resetImagePacing(mode: .timer)
        onImagesCleared?()
        tale = false
        voice.stop()
        onStatus?("остановлено", false)
    }

    /// Включить медленное чтение сказки: `VoicePlayer` будет ставить длинные
    /// паузы. Настройка «обычная/медленная» читается им в момент паузы,
    /// поэтому применяется без перезапуска.
    private func enableTaleMode(_ reason: String) {
        guard !tale else { return }
        tale = true
        voice.setTaleMode(true)
        log("озвучка: режим сказки — паузы длиннее (\(reason))")
    }

    /// Только звук: выключили галочку «Озвучивать» — генерацию не трогаем.
    /// Картинки при этом продолжают меняться: прогресса речи больше нет,
    /// поэтому оставшуюся очередь ведёт таймер от текущей позиции.
    func stopVoice() {
        if speak, !silent {
            speak = false
            beginTimerPacing(from: Date(), base: pacer.spoken)
        }
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

    /// Отметить номер картинки готовым (или пропущенным) и показать всё, что
    /// уже можно показать: позиция достигнута и файл готов. Порядок — по
    /// номерам событий, поэтому медленная загрузка не переставляет картинки.
    private func resolveImage(order: Int) {
        imageResolved.insert(order)
        advanceImages()
    }

    /// Показать очередные картинки, до которых дошёл прогресс чтения.
    private func advanceImages() {
        // Настройку могли выключить, пока картинка ждала своей позиции:
        // очередь больше не показываем (окно уже очищено настройками).
        guard Settings.shared.showTaleImages else {
            if pacer.next != nil {
                while pacer.next != nil { pacer.markShown() }
                imageResolved.removeAll()
                imageReady.removeAll()
                log("картинка: настройка выключена — очередь показа очищена")
            }
            return
        }
        while let item = pacer.next,
              pacer.isReached(item),
              imageResolved.contains(item.order)
        {
            pacer.markShown()
            guard let ready = imageReady.removeValue(forKey: item.order) else {
                // Пропуск (настройка выключена, битый URL, ошибка загрузки):
                // место в очереди закрыто, следующая картинка не ждёт.
                continue
            }
            log("картинка: показываю \(ready.alt.isEmpty ? "иллюстрацию" : ready.alt) "
                + "(№\(item.order), после \(item.afterVisible) симв.)")
            onImage?(ready.image, ready.alt)
        }
        if streamEnded, pacer.next == nil {
            stopImageTimer()
        }
    }

    // MARK: - Темп картинок

    /// Сбросить очередь и часы показа. Новый вопрос и «Стоп» начинают с
    /// чистой страницы; последняя картинка к этому моменту уже убрана окном.
    private func resetImagePacing(mode: TaleImagePacer.Mode) {
        stopImageTimer()
        pacer.reset(mode: mode)
        imageResolved.removeAll()
        imageReady.removeAll()
        imageCount = 0
        streamEnded = false
    }

    /// Перейти на оценочный темп: 15 символов в секунду от `origin`.
    /// `base` — уже «пройденные» символы (при переключении посреди ответа,
    /// когда озвучка пропала, прогресс не откатывается назад).
    private func beginTimerPacing(from origin: Date, base: Double) {
        pacer.useTimer()
        imageTimerBase = max(0, base)
        imageTimerStart = origin
        startImageTimer()
        imageTick()
    }

    private func startImageTimer() {
        imageTimer?.invalidate()
        // 5 Гц достаточно: картинки стоят в тексте на десятки секунд.
        let timer = Timer(
            timeInterval: 0.2,
            target: self,
            selector: #selector(imageTick),
            userInfo: nil,
            repeats: true
        )
        timer.tolerance = 0.05
        RunLoop.main.add(timer, forMode: .common)
        imageTimer = timer
    }

    private func stopImageTimer() {
        imageTimer?.invalidate()
        imageTimer = nil
    }

    @objc private func imageTick() {
        guard pacer.mode == .timer else {
            stopImageTimer()
            return
        }
        let elapsed = max(0, Date().timeIntervalSince(imageTimerStart))
        pacer.updateTimer(elapsed: elapsed, base: imageTimerBase)
        advanceImages()
    }

    /// Прогресс озвучки: показываем картинки, до которых дошла речь.
    private func voiceDidSpeak(_ spoken: Int) {
        guard pacer.mode == .speech else { return }
        pacer.updateSpoken(Double(spoken))
        advanceImages()
    }

    /// TTS не поднялся (сервер молчит, WAV не сыгрался): без прогресса
    /// картинки застряли бы на первой. Дальше ведёт таймер с текущей
    /// позиции, чтобы смена не откатилась назад.
    private func voiceUnavailable() {
        guard !silent, pacer.mode == .speech else { return }
        log("картинки: озвучка недоступна — темп по таймеру "
            + "(\(Int(pacer.spoken)) симв. уже позади)")
        beginTimerPacing(from: Date(), base: pacer.spoken)
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

/// Очередь картинок, привязанная к прогрессу чтения (задача 0227).
///
/// Ответ плагина-сказки приходит одним куском: все `delta` и `image`
/// прилетают за миллисекунды, поэтому показ «в момент события» оставлял на
/// экране только последнюю картинку. Здесь запоминается позиция каждой
/// картинки — сколько символов ответа должно быть уже сказано, — и показ
/// начинается только когда прогресс дошёл до неё.
///
/// Два источника прогресса:
/// * `.speech` — `VoicePlayer.onProgress` (символы фраз, ушедших в TTS);
///   позиция `afterVoice` считается по символам, реально попавшим в озвучку;
/// * `.timer` — оценка 15 символов в секунду от начала ответа (озвучка
///   выключена, играет внешняя запись или TTS недоступен); позиция
///   `afterVisible` — по видимому тексту, включая пробелы.
///
/// Чистый тип без таймеров и `Date`: время и темп передаются снаружи, поэтому
/// сценарий «900 символов, картинки на 300/600/900» проверяется в
/// `--selftest-images` без реального ожидания.
struct TaleImagePacer {
    enum Mode: Equatable {
        /// Прогресс речи из `VoicePlayer`.
        case speech
        /// Оценочный темп 15 символов/с.
        case timer
    }

    struct Item: Equatable {
        /// Номер события `image` (1, 2, 3…): порядок показа.
        let order: Int
        /// Сколько символов видимого текста должно быть позади (таймер).
        let afterVisible: Int
        /// Сколько символов ушло в озвучку к моменту события (речь).
        let afterVoice: Int
    }

    /// Оценка темпа русской речи baya: 13 символов в секунду. Столько
    /// показал живой прогон (12,8 сим/с) — при 15 картинки убегали вперёд.
    static let charsPerSecond = 12.8
    /// Допуск на дробную арифметику: позиция 300 достигается ровно на 20,0 с.
    private static let epsilon = 0.000_001

    private(set) var mode: Mode
    private(set) var spoken = 0.0
    private(set) var shown = 0
    private var items: [Item] = []

    init(mode: Mode) {
        self.mode = mode
    }

    mutating func reset(mode: Mode) {
        self.mode = mode
        spoken = 0
        shown = 0
        items.removeAll()
    }

    mutating func useTimer() {
        mode = .timer
    }

    mutating func enqueue(_ item: Item) {
        items.append(item)
    }

    /// Следующая картинка в очереди (ещё не показанная и не пропущенная).
    var next: Item? {
        shown < items.count ? items[shown] : nil
    }

    /// Прогресс озвучки: символы, уже сказанные с начала ответа.
    /// Назад не откатывается — запоздавший колбэк не вернёт старую картинку.
    mutating func updateSpoken(_ chars: Double) {
        spoken = max(spoken, chars)
    }

    /// Прогресс по оценочному темпу: `base` — символы, пройденные до запуска
    /// часов, `elapsed` — секунды с их запуска.
    mutating func updateTimer(
        elapsed: Double,
        base: Double = 0,
        rate: Double = TaleImagePacer.charsPerSecond
    ) {
        let seconds = max(0, elapsed)
        let speed = max(0, rate)
        spoken = max(spoken, max(0, base) + seconds * speed)
    }

    /// Пора ли показывать картинку: прогресс дошёл до её позиции.
    func isReached(_ item: Item) -> Bool {
        let position = mode == .speech ? item.afterVoice : item.afterVisible
        return spoken + TaleImagePacer.epsilon >= Double(position)
    }

    /// Картинка показана (или пропущена): очередь сдвигается.
    mutating func markShown() {
        guard shown < items.count else { return }
        shown += 1
    }
}
