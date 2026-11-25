import AVFoundation
import Foundation
import Speech

/// Фоновое ожидание фразы «Привет, Ева» (задача 0187).
///
/// Это отдельный `AVAudioEngine` и отдельный `SFSpeechRecognizer` (ru-RU,
/// только on-device): обычная запись `SpeechRecorder` в это время не идёт,
/// а слушатель живёт только в `idle` и выключается на время записи и озвучки.
///
/// Поток звука устроен так, чтобы realtime-очередь аудио вообще не касалась
/// SFSpeech и @MainActor-состояния:
///
/// ```
/// tap (realtime) → WakeAudioRing (замок) → WakeAudioPump (своя очередь)
///                → SFSpeechAudioBufferRecognitionRequest
/// ```
///
/// Тап только копирует сэмплы в кольцевой буфер (2 с), поэтому не хватает
/// @MainActor-полей — иначе Swift 6 роняет процесс через dispatch_assert_queue.
/// Буфер ограничивает память и сглаживает паузы главного потока, а распознанный
/// текст приходит в `handleRecognition` уже на главном акторе.
@MainActor
final class WakeWordListener {
    /// Слушатель действительно держит микрофон и распознавание.
    private(set) var isRunning = false

    /// Услышали фразу: сюда приходит распознанный текст (как услышали).
    /// Вызывающий обязан сразу остановить слушатель и начать запись.
    var onWake: ((String) -> Void)?
    /// Ожидание не поднялось (нет прав, нет офлайн-модели, микрофон занят).
    /// Текст — для строки статуса, в журнал он уже попал с префиксом «wake:».
    var onIssue: ((String) -> Void)?
    /// Слушатель поднялся или опустился (для пульсации круга).
    var onState: ((Bool) -> Void)?

    /// Разрешения запрашиваются асинхронно: между `start()` и включённым
    /// микрофоном есть окно, в котором `stop()` должен отменить запуск.
    private var starting = false
    private var generation = 0

    private var engine: AVAudioEngine?
    private var ring: WakeAudioRing?
    private var pump: WakeAudioPump?
    private var requestBox: WakeRequestBox?
    private var recognizer: SFSpeechRecognizer?
    private var task: SFSpeechRecognitionTask?
    private var tapInstalled = false
    /// Сколько подряд ошибок распознавания: после нескольких не пытаемся
    /// бесконечно перезапускать задачу, а честно выключаем ожидание.
    private var recognitionErrors = 0
    private var fired = false
    /// Токен текущей задачи распознавания: отменённая задача может прислать
    /// ещё один колбэк, и он не должен перезапускать новую.
    private var recognitionToken = 0

    /// 2 секунды кольцевого буфера: с запасом на паузу главного потока и
    /// всё ещё немного памяти (≈0,8 МБ на стерео 48 кГц).
    private let ringSeconds: Double = 2

    // MARK: - Включение и выключение

    func start() {
        guard !isRunning, !starting else { return }
        starting = true
        generation += 1
        let run = generation
        fired = false
        log("wake: включаю ожидание")
        Task { [weak self] in
            guard let self else { return }
            let speech = await WakeWordListener.requestSpeechAuthorization()
            guard self.generation == run, self.starting else { return }
            switch speech {
            case .authorized:
                break
            case .denied:
                self.fail("нет разрешения на распознавание речи: Системные настройки → "
                    + "Конфиденциальность → Распознавание речи → включить Еву")
                return
            case .restricted:
                self.fail("распознавание речи ограничено системой")
                return
            case .notDetermined:
                self.fail("система не дала ответа по правам на распознавание речи")
                return
            @unknown default:
                self.fail("неизвестный статус прав на распознавание речи")
                return
            }

            let microphone = await AVCaptureDevice.requestAccess(for: .audio)
            guard self.generation == run, self.starting else { return }
            guard microphone else {
                self.fail("нет доступа к микрофону: Системные настройки → "
                    + "Конфиденциальность → Микрофон → включить Еву")
                return
            }
            self.beginCapture(run: run)
        }
    }

    /// Полностью отпускает микрофон: движок остановлен, тап снят, запрос и
    /// задача распознавания закрыты. Повторный вызов безопасен.
    func stop() {
        guard isRunning || starting else { return }
        let wasRunning = isRunning
        starting = false
        isRunning = false
        generation += 1
        teardownCapture()
        teardownRecognition()
        requestBox = nil
        onState?(false)
        if wasRunning {
            log("wake: ожидание выключено")
        } else {
            log("wake: ожидание выключено (запуск отменён)")
        }
    }

    // MARK: - Захват

    private func beginCapture(run: Int) {
        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "ru-RU")) else {
            fail("в системе нет распознавания для ru-RU")
            return
        }
        guard recognizer.isAvailable else {
            fail("распознавание сейчас недоступно — попробуйте позже")
            return
        }
        guard recognizer.supportsOnDeviceRecognition else {
            fail("на этом Mac не установлена офлайн-модель ru-RU: "
                + "Системные настройки → Клавиатура → Диктовка → добавить русский")
            return
        }
        self.recognizer = recognizer

        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            fail("микрофон не найден (нет входного устройства)")
            return
        }
        guard let standard = AVAudioFormat(
            standardFormatWithSampleRate: format.sampleRate,
            channels: format.channelCount
        ) else {
            fail("не удалось подготовить формат микрофона")
            return
        }
        let capacity = Int(format.sampleRate * ringSeconds)
        let ring = WakeAudioRing(format: standard, capacity: capacity)
        // Колбэк тапа приходит на realtime-очереди аудио, а не на главном
        // акторе. Замыкание объявлено @Sendable и держит только неизолированную
        // обёртку буфера — @MainActor-состояние слушателя оно не видит.
        let sink: @Sendable (AVAudioPCMBuffer, AVAudioTime) -> Void = { buffer, _ in
            ring.append(buffer)
        }
        input.installTap(onBus: 0, bufferSize: 2048, format: format, block: sink)
        tapInstalled = true

        engine.prepare()
        do {
            try engine.start()
        } catch {
            engine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
            fail("не удалось включить микрофон для ожидания: \(error.localizedDescription)")
            return
        }
        self.engine = engine
        self.ring = ring

        let box = WakeRequestBox()
        requestBox = box
        pump = WakeAudioPump(ring: ring, request: box)
        pump?.start()

        isRunning = true
        starting = false
        onState?(true)
        log("wake: ожидание включено (микрофон \(Int(format.sampleRate)) Гц, "
            + "каналов \(format.channelCount), on-device)")

        beginRecognition(run: run)
    }

    private func beginRecognition(run: Int) {
        guard isRunning, generation == run, let recognizer else { return }
        guard recognizer.isAvailable else {
            fail("распознавание перестало быть доступным")
            return
        }
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        // Принципиально: звук не уходит в сеть, распознавание локальное.
        request.requiresOnDeviceRecognition = true
        request.taskHint = .dictation
        if #available(macOS 13.0, *) {
            request.addsPunctuation = true
        }
        // Подсказка распознавателю: фраза короткая и знакомая, это уменьшает
        // число «привет ева» → «привет, дева» и подобных ослышек.
        request.contextualStrings = ["Привет, Ева", "Эй, Ева"]
        self.request = request
        requestBox?.set(request)

        recognitionToken += 1
        let token = recognitionToken
        let generation = self.generation
        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            // Колбэк приходит из потока распознавателя: наружу отдаём только
            // Sendable-значения, состояние трогаем на главном акторе.
            let text = result?.bestTranscription.formattedString ?? ""
            let isFinal = result?.isFinal ?? false
            let message = error?.localizedDescription
            Task { @MainActor [weak self] in
                self?.handleRecognition(
                    text: text,
                    isFinal: isFinal,
                    error: message,
                    run: generation,
                    token: token
                )
            }
        }
    }

    private func handleRecognition(
        text: String,
        isFinal: Bool,
        error: String?,
        run: Int,
        token: Int
    ) {
        guard isRunning, !starting, generation == run, token == recognitionToken else {
            return
        }

        if !text.isEmpty {
            recognitionErrors = 0
            if WakeWordListener.isWakePhrase(text), !fired {
                fired = true
                log("wake: услышала „\(text)“")
                onWake?(text)
                // `onWake` останавливает слушатель; если по какой-то причине
                // не остановил — не продолжаем кормить ту же задачу.
                return
            }
        }

        if let error {
            recognitionErrors += 1
            log("wake: ошибка распознавания: \(error)")
            if recognitionErrors >= 5 {
                fail("распознавание не запускается: \(error)")
                return
            }
            restartRecognition(run: run, after: 0.5)
        } else if isFinal {
            // Короткая задача на устройстве заканчивается сама: начинаем
            // новую, движок и кольцевой буфер при этом не трогаем.
            restartRecognition(run: run, after: 0.05)
        }
    }

    private func restartRecognition(run: Int, after delay: Double) {
        teardownRecognition()
        guard isRunning, generation == run else { return }
        if delay <= 0 {
            beginRecognition(run: run)
            return
        }
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            guard let self, self.isRunning, self.generation == run else { return }
            self.beginRecognition(run: run)
        }
    }

    // MARK: - Сворачивание

    private func teardownCapture() {
        pump?.stop()
        pump = nil
        if let engine {
            engine.stop()
            if tapInstalled {
                engine.inputNode.removeTap(onBus: 0)
            }
        }
        engine = nil
        ring = nil
        tapInstalled = false
    }

    private func teardownRecognition() {
        // Токен растёт сразу: колбэки отменённой задачи уже не считаются.
        recognitionToken += 1
        requestBox?.set(nil)
        request?.endAudio()
        task?.cancel()
        task = nil
        request = nil
    }

    private var request: SFSpeechAudioBufferRecognitionRequest?

    private func fail(_ message: String) {
        guard isRunning || starting else { return }
        log("wake: \(message)")
        onIssue?(message)
        stop()
    }

    // MARK: - Права

    /// Nonisolated по той же причине, что и в `SpeechRecorder`: колбэк TCC
    /// приходит на своей очереди, а замыкание внутри @MainActor-функции Swift 6
    /// считает изолированным и падает с dispatch_assert_queue (SIGTRAP).
    nonisolated
    private static func requestSpeechAuthorization() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { (continuation: CheckedContinuation<SFSpeechRecognizerAuthorizationStatus, Never>) in
            let handler: @Sendable (SFSpeechRecognizerAuthorizationStatus) -> Void = { status in
                continuation.resume(returning: status)
            }
            SFSpeechRecognizer.requestAuthorization(handler)
        }
    }

    // MARK: - Фраза

    /// «Привет, Ева!» → «привет ева»: нижний регистр, только буквы и пробелы.
    nonisolated static func normalized(_ text: String) -> String {
        var result = ""
        result.reserveCapacity(text.count)
        for scalar in text.lowercased().unicodeScalars {
            if CharacterSet.letters.contains(scalar) {
                result.unicodeScalars.append(scalar)
            } else if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                result.append(" ")
            }
        }
        return result.split(separator: " ").joined(separator: " ")
    }

    /// Триггер: минимум два слова. Либо «привет» + «ева» в любом порядке
    /// (частичный результат может переставить слова), либо «эй ева» рядом.
    nonisolated static func isWakePhrase(_ text: String) -> Bool {
        let words = normalized(text).split(separator: " ").map(String.init)
        guard words.count >= 2 else { return false }
        let hasGreeting = words.contains("привет")
        let hasName = words.contains("ева")
        if hasGreeting && hasName { return true }
        for index in 0..<(words.count - 1) {
            if words[index] == "эй" && words[index + 1] == "ева" { return true }
        }
        return false
    }
}

// MARK: - Кольцевой буфер

/// Кольцевой буфер последних сэмплов. Между realtime-тапом и очередью
/// распознавания: тап пишет под коротким замком, насос читает. `@unchecked
/// Sendable` честен — всё изменяемое состояние защищено замком.
final class WakeAudioRing: @unchecked Sendable {
    let format: AVAudioFormat
    private let channelCount: Int
    private let capacity: Int
    private var storage: [Float]
    private var writeIndex = 0
    private var available = 0
    private var written = 0
    private let lock = NSLock()

    init(format: AVAudioFormat, capacity: Int) {
        self.format = format
        self.channelCount = max(Int(format.channelCount), 1)
        self.capacity = max(capacity, 1024)
        self.storage = [Float](repeating: 0, count: self.capacity * self.channelCount)
    }

    /// Вызывается из аудио-тапа. Формат тапа может быть interleaved или нет;
    /// наружу буфер всегда отдаётся в стандартном (deinterleaved float32).
    func append(_ buffer: AVAudioPCMBuffer) {
        let frames = Int(buffer.frameLength)
        guard frames > 0,
              Int(buffer.format.channelCount) == channelCount,
              let data = buffer.floatChannelData
        else {
            return
        }
        lock.lock()
        if buffer.format.isInterleaved {
            let source = data[0]
            for index in 0..<frames {
                let slot = ((writeIndex + index) % capacity) * channelCount
                for channel in 0..<channelCount {
                    storage[slot + channel] = source[index * channelCount + channel]
                }
            }
        } else {
            for channel in 0..<channelCount {
                let source = data[channel]
                for index in 0..<frames {
                    storage[((writeIndex + index) % capacity) * channelCount + channel] = source[index]
                }
            }
        }
        writeIndex = (writeIndex + frames) % capacity
        available = min(capacity, available + frames)
        written += frames
        lock.unlock()
    }

    /// Сколько кадров прошло через буфер с момента создания (диагностика).
    var writtenFrames: Int {
        lock.lock()
        defer { lock.unlock() }
        return written
    }

    /// Забрать до `maxFrames` кадров по порядку записи. `nil` — данных нет.
    func take(maxFrames: Int = 8192) -> AVAudioPCMBuffer? {
        lock.lock()
        let frames = min(available, min(capacity, maxFrames))
        guard frames > 0,
              let buffer = AVAudioPCMBuffer(
                  pcmFormat: format,
                  frameCapacity: AVAudioFrameCount(frames)
              ),
              let destination = buffer.floatChannelData
        else {
            lock.unlock()
            return nil
        }
        let start = (writeIndex - available + capacity) % capacity
        buffer.frameLength = AVAudioFrameCount(frames)
        for index in 0..<frames {
            let slot = ((start + index) % capacity) * channelCount
            for channel in 0..<channelCount {
                destination[channel][index] = storage[slot + channel]
            }
        }
        available -= frames
        lock.unlock()
        return buffer
    }
}

// MARK: - Насос и контейнер запроса

/// Текущий запрос распознавания за неизолированной обёрткой: его меняет
/// главный актор при перезапуске задачи, а читает очередь насоса. Замок
/// защищает только указатель; `append` у SFSpeechAudioBufferRecognitionRequest
/// можно звать из одного потока записи.
final class WakeRequestBox: @unchecked Sendable {
    private let lock = NSLock()
    private var request: SFSpeechAudioBufferRecognitionRequest?

    func set(_ request: SFSpeechAudioBufferRecognitionRequest?) {
        lock.lock()
        self.request = request
        lock.unlock()
    }

    /// Есть ли куда отдавать звук: во время перезапуска задачи насос не
    /// вычерпывает буфер, поэтому речь на стыке задач не теряется.
    var isReady: Bool {
        lock.lock()
        defer { lock.unlock() }
        return request != nil
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        let current = request
        lock.unlock()
        current?.append(buffer)
    }
}

/// Переносит звук из кольцевого буфера в запрос распознавания на отдельной
/// последовательной очереди, 25 раз в секунду. Главный поток и realtime-тап
/// в этом не участвуют.
final class WakeAudioPump: @unchecked Sendable {
    private let queue = DispatchQueue(
        label: "com.eva32.eva.wake.audio",
        qos: .utility
    )
    private let ring: WakeAudioRing
    private let request: WakeRequestBox
    private var timer: DispatchSourceTimer?

    init(ring: WakeAudioRing, request: WakeRequestBox) {
        self.ring = ring
        self.request = request
    }

    func start() {
        queue.async { [weak self] in
            guard let self, self.timer == nil else { return }
            let timer = DispatchSource.makeTimerSource(queue: self.queue)
            timer.schedule(
                deadline: .now() + .milliseconds(40),
                repeating: .milliseconds(40),
                leeway: .milliseconds(10)
            )
            timer.setEventHandler { [weak self] in
                self?.drain()
            }
            timer.resume()
            self.timer = timer
        }
    }

    func stop() {
        queue.sync {
            timer?.cancel()
            timer = nil
        }
    }

    private func drain() {
        while request.isReady, let buffer = ring.take(maxFrames: 8192) {
            request.append(buffer)
        }
    }
}
