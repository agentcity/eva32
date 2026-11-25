import AVFoundation
import Foundation
import Speech

/// Чем закончился follow-up без вопроса (задача 0190). Тишина — это не
/// ошибка: вызывающий тихо возвращает wake-ожидание и ничего не отправляет.
enum FollowUpEnd: Sendable, Equatable {
    /// Ни слова за `followUpSilence` секунд; число — выдержанная пауза.
    case silence(Int)
    /// Микрофон закрыли сами (⌥Space/«Стоп»), не сказав ни слова.
    case manual
}

/// Запись с микрофона и распознавание ru-RU **на устройстве**.
///
/// Права: Apple Speech требует `NSSpeechRecognitionUsageDescription`, микрофон —
/// `NSMicrophoneUsageDescription`; в CLI-бинарнике авторизация не выдаётся,
/// поэтому приложение обязано быть .app-бандлом (см. Info.plist и сборку).
///
/// Поток: `start()` → разрешения → AVAudioEngine + installTap → частичные
/// результаты в окно; `stop()` (повторное ⌥Space) отдаёт последний распознанный
/// текст в чат.
@MainActor
final class SpeechRecorder {
    private(set) var isRecording = false

    var onState: ((Bool) -> Void)?
    var onPartial: ((String) -> Void)?
    var onFinal: ((String) -> Void)?
    var onError: ((String) -> Void)?
    /// Follow-up закончился без слов: ни ошибки, ни отправки (задача 0190).
    var onFollowUpEnd: ((FollowUpEnd) -> Void)?
    /// Уровень микрофона 0…1 для большого круга. Обновляется с главного
    /// актора 30 раз в секунду, а RMS считается в аудио-тапе (см. `LevelMeter`).
    var onLevel: ((Float) -> Void)?

    /// Текущая запись — follow-up: ждём вопрос без «Привет, Ева». Пока не
    /// сказано ни слова, пауза ограничена `followUpSilence`, а закрытие тихое.
    private(set) var isFollowUp = false
    /// Сколько секунд ждать первую фразу в follow-up (из настроек).
    var followUpSilence: Double = Double(Settings.defaultFollowUpSilence)

    private var generation = 0
    private var engine: AVAudioEngine?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var recognizer: SFSpeechRecognizer?
    private var tapInstalled = false
    private var lastText = ""
    private var delivered = true
    private var timeout: Task<Void, Never>?
    /// RMS, посчитанный в realtime-тапе, и тикер, который читает его на
    /// главном акторе. Сам тап @MainActor-состояние не захватывает.
    private var levelMeter: LevelMeter?
    private var levelPump: Task<Void, Never>?
    /// Авто-стоп по тишине: не заставляем пользователя жать ⌥Space второй раз.
    private var silence: Task<Void, Never>?
    /// Сколько секунд тишины терпим, если уже есть распознанный текст.
    private let silenceLimit: Double = 2.2
    /// Пока ни одного слова не расслышано — ждём дольше, но не бесконечно.
    private let silenceWarmup: Double = 6.0
    /// Когда последний раз приходил текст (или началась запись).
    private var lastVoiceAt = Date()

    /// Дольше минуты не слушаем: забытый микрофон — это и батарея, и приватность.
    private let maxDuration: UInt64 = 60

    func start(followUp: Bool = false) {
        guard !isRecording else { return }
        generation += 1
        let run = generation
        isRecording = true
        isFollowUp = followUp
        lastText = ""
        delivered = false
        onState?(true)
        onPartial?("")
        log("речь: старт записи" + (followUp ? " (follow-up)" : ""))

        timeout?.cancel()
        timeout = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(nanoseconds: self.maxDuration * 1_000_000_000)
            guard !Task.isCancelled, self.isRecording else { return }
            log("речь: авто-стоп через \(self.maxDuration) с")
            self.stop()
        }

        Task { [weak self] in
            guard let self else { return }
            let speech = await SpeechRecorder.requestSpeechAuthorization()
            guard self.generation == run, self.isRecording else { return }
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
            guard self.generation == run, self.isRecording else { return }
            guard microphone else {
                self.fail("нет доступа к микрофону: Системные настройки → "
                    + "Конфиденциальность → Микрофон → включить Еву")
                return
            }
            self.beginCapture(run: run)
        }
    }

    func stop() {
        guard isRecording else { return }
        finish(error: nil)
    }

    // MARK: - Запись

    private func beginCapture(run: Int) {
        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "ru-RU")) else {
            fail("в системе нет распознавания для ru-RU")
            return
        }
        guard recognizer.isAvailable else {
            fail("распознавание сейчас недоступно — попробуйте ещё раз")
            return
        }
        guard recognizer.supportsOnDeviceRecognition else {
            fail("на этом Mac не установлена офлайн-модель ru-RU: "
                + "Системные настройки → Клавиатура → Диктовка → добавить русский")
            return
        }
        self.recognizer = recognizer

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        // Принципиально: только на устройстве, без отправки звука в сеть.
        request.requiresOnDeviceRecognition = true
        request.taskHint = .dictation
        if #available(macOS 13.0, *) {
            request.addsPunctuation = true
        }

        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            fail("микрофон не найден (нет входного устройства)")
            return
        }
        // Тап на родном формате микрофона: Apple Speech сам приводит звук
        // к 16 кГц внутри распознавателя, внешний конвертер только добавил бы
        // точку отказа.
        // Колбэк тапа AVAudioEngine приходит на realtime-очереди аудио, а НЕ на
        // главном акторе. Замыкание внутри @MainActor-класса Swift 6 считает
        // изолированным и роняет процесс через dispatch_assert_queue — поэтому
        // оно объявлено @Sendable и работает только с захваченным request.
        // Класс-обёртка: замыкание тапа не должно захватывать @MainActor-состояние,
        // а SFSpeechAudioBufferRecognitionRequest не Sendable — передаём его через
        // неизолированный контейнер (append потокобезопасен). Туда же кладём
        // RMS для круга: 2048 кадров ≈ 43 мс при 48 кГц, реакция <100 мс.
        let meter = LevelMeter()
        let sinkBox = RequestSink(request: request, meter: meter)
        let sink: @Sendable (AVAudioPCMBuffer, AVAudioTime) -> Void = { buffer, _ in
            sinkBox.append(buffer)
        }
        input.installTap(onBus: 0, bufferSize: 2048, format: format, block: sink)
        tapInstalled = true
        lastVoiceAt = Date()
        startSilenceWatch()
        engine.prepare()
        do {
            try engine.start()
        } catch {
            teardown()
            fail("не удалось включить микрофон: \(error.localizedDescription)")
            return
        }
        self.engine = engine
        self.request = request
        self.levelMeter = meter
        startLevelPump(run: run, meter: meter)
        log("речь: микрофон \(Int(format.sampleRate)) Гц, каналов \(format.channelCount), on-device")

        self.task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            // Колбэк приходит из потока распознавателя: наружу отдаём только
            // Sendable-значения, состояние трогаем на главном акторе.
            let text = result?.bestTranscription.formattedString ?? ""
            let isFinal = result?.isFinal ?? false
            let message = error?.localizedDescription
            Task { @MainActor [weak self] in
                self?.handle(text: text, isFinal: isFinal, error: message, run: run)
            }
        }
    }

    private func handle(text: String, isFinal: Bool, error: String?, run: Int) {
        guard run == generation, isRecording else { return }
        if !text.isEmpty {
            lastText = text
            lastVoiceAt = Date()
            onPartial?(text)
        }
        if let error {
            log("речь: ошибка распознавания: \(error)")
            finish(error: error)
            return
        }
        if isFinal {
            finish(error: nil)
        }
    }

    /// Тикер для круга: читает RMS из неизолированной обёртки и отдаёт его на
    /// главном акторе. 30 Гц достаточно для плавной волны и не грузит CPU.
    private func startLevelPump(run: Int, meter: LevelMeter) {
        levelPump?.cancel()
        levelPump = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self, self.generation == run, self.isRecording else { return }
                self.onLevel?(meter.displayLevel)
                try? await Task.sleep(nanoseconds: 33_000_000)
            }
        }
    }

    /// Сторож тишины: если человек замолчал, сами заканчиваем запись и отдаём
    /// текст. Без него «слушаю» висело вечно — пользователь ждал, что Ева
    /// остановится сама, и не понимал, нужно ли жать ⌥Space ещё раз.
    private func startSilenceWatch() {
        silence?.cancel()
        silence = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 300_000_000)
                guard let self, self.isRecording else { return }
                let idle = Date().timeIntervalSince(self.lastVoiceAt)
                let hasText = !self.lastText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                let limit = SpeechRecorder.silenceTimeout(
                    hasText: hasText,
                    followUp: self.isFollowUp,
                    followUpSilence: self.followUpSilence,
                    warmup: self.silenceWarmup,
                    afterSpeech: self.silenceLimit
                )
                if idle >= limit {
                    if self.isFollowUp, !hasText {
                        // Follow-up без единого слова: это не ошибка, вопрос
                        // отправлять нечего — тихо уходим в ожидание.
                        log("речь: follow-up — тишина "
                            + String(format: "%.0f", limit) + " с, закрываю")
                        self.finishFollowUp(.silence(Int(limit.rounded())))
                        return
                    }
                    log("речь: тишина \(String(format: "%.1f", idle)) с — заканчиваю сам")
                    self.finish(error: nil)
                    return
                }
            }
        }
    }

    /// Порог тишины для текущей записи. Вынесен в `nonisolated static`, чтобы
    /// проверить без микрофона (`--selftest-followup`): после речи — 2,2 с,
    /// follow-up без слов — пауза из настроек, обычная запись без слов — 6 с.
    nonisolated static func silenceTimeout(
        hasText: Bool,
        followUp: Bool,
        followUpSilence: Double,
        warmup: Double,
        afterSpeech: Double
    ) -> Double {
        if hasText { return afterSpeech }
        if followUp { return max(followUpSilence, 0.5) }
        return warmup
    }

    // MARK: - Завершение

    private func finish(error: String?) {
        guard isRecording else { return }
        isRecording = false
        generation += 1
        let followUp = isFollowUp
        isFollowUp = false
        timeout?.cancel()
        timeout = nil
        silence?.cancel()
        silence = nil
        teardown()
        onState?(false)

        let text = lastText.trimmingCharacters(in: .whitespacesAndNewlines)
        log("речь: стоп, распознано \(text.count) символов")
        if !delivered {
            delivered = true
            if text.isEmpty {
                if followUp {
                    // В follow-up пустая запись — не «не расслышала», а
                    // нормальный уход в ожидание.
                    log("речь: follow-up остановлен без слов — не отправляю")
                    onFollowUpEnd?(.manual)
                    return
                }
                onError?(error.map { "распознавание: \($0)" } ?? "не расслышала — попробуйте ещё раз")
                return
            }
            onFinal?(text)
        }
    }

    /// Тихое закрытие follow-up: микрофон отпущен, ни ошибки, ни отправки.
    /// `delivered = true` — `finish` больше ничего не пришлёт.
    private func finishFollowUp(_ end: FollowUpEnd) {
        guard isRecording else { return }
        isRecording = false
        generation += 1
        isFollowUp = false
        timeout?.cancel()
        timeout = nil
        silence?.cancel()
        silence = nil
        teardown()
        delivered = true
        onState?(false)
        onFollowUpEnd?(end)
    }

    private func fail(_ message: String) {
        guard isRecording else { return }
        isRecording = false
        generation += 1
        isFollowUp = false
        timeout?.cancel()
        timeout = nil
        silence?.cancel()
        silence = nil
        teardown()
        onState?(false)
        delivered = true
        log("речь: \(message)")
        onError?(message)
    }

    private func teardown() {
        if let engine {
            engine.stop()
            if tapInstalled {
                engine.inputNode.removeTap(onBus: 0)
            }
        }
        levelPump?.cancel()
        levelPump = nil
        levelMeter = nil
        onLevel?(0)
        request?.endAudio()
        task?.cancel()
        engine = nil
        request = nil
        task = nil
        recognizer = nil
        tapInstalled = false
    }

    // MARK: - Права

    /// Запрос прав на распознавание. Обязательно **nonisolated**: колбэк TCC
    /// приходит на своей очереди, а замыкание внутри @MainActor-функции Swift 6
    /// считает изолированным и падает с dispatch_assert_queue (SIGTRAP) — именно
    /// это случилось при первом ⌥Space после выдачи разрешения.
    nonisolated
    private static func requestSpeechAuthorization() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { (continuation: CheckedContinuation<SFSpeechRecognizerAuthorizationStatus, Never>) in
            let handler: @Sendable (SFSpeechRecognizerAuthorizationStatus) -> Void = { status in
                continuation.resume(returning: status)
            }
            SFSpeechRecognizer.requestAuthorization(handler)
        }
    }
}

/// Неизолированная обёртка над запросом распознавания: нужна, чтобы колбэк
/// аудио-тапа (realtime-очередь, @Sendable) мог складывать буферы, не хватая
/// @MainActor-состояние SpeechRecorder. `append` у SFSpeechAudioBufferRecognitionRequest
/// безопасен при вызове из одного потока записи.
final class RequestSink: @unchecked Sendable {
    private let request: SFSpeechAudioBufferRecognitionRequest
    private let meter: LevelMeter?

    init(request: SFSpeechAudioBufferRecognitionRequest, meter: LevelMeter? = nil) {
        self.request = request
        self.meter = meter
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        request.append(buffer)
        meter?.update(buffer)
    }
}

/// RMS последнего аудио-буфера. Живёт между realtime-тапом и главным актором:
/// тап пишет под коротким замком, тикер круга читает. `@unchecked Sendable`
/// честен: всё изменяемое состояние защищено замком, а буфер наружу не отдаётся.
final class LevelMeter: @unchecked Sendable {
    private let lock = NSLock()
    private var rms: Float = 0

    func update(_ buffer: AVAudioPCMBuffer) {
        let value = LevelMeter.rms(of: buffer)
        lock.lock()
        rms = value
        lock.unlock()
    }

    /// Нормированный 0…1 уровень: −50 дБ — тишина, 0 дБ — громко.
    /// Обычная речь даёт примерно 0,4–0,8, поэтому волна заметна сразу.
    var displayLevel: Float {
        lock.lock()
        let value = rms
        lock.unlock()
        let db = 20 * log10f(max(value, 0.000001))
        return min(max((db + 50) / 50, 0), 1)
    }

    private static func rms(of buffer: AVAudioPCMBuffer) -> Float {
        guard let channels = buffer.floatChannelData,
              buffer.frameLength > 0
        else {
            return 0
        }
        let frames = Int(buffer.frameLength)
        let channelCount = max(Int(buffer.format.channelCount), 1)
        var sum: Float = 0
        for channel in 0..<channelCount {
            let samples = channels[channel]
            for index in 0..<frames {
                let sample = samples[index]
                sum += sample * sample
            }
        }
        return sqrtf(sum / Float(frames * channelCount))
    }
}
