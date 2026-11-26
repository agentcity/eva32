import AVFoundation
import Foundation
import Speech

/// Голосовая команда «Стоп, Ева» (задача 0194).
///
/// Слушатель живёт только пока Ева «думает» или говорит: wake-ожидание в это
/// время выключено, поэтому `AVAudioEngine` у микрофона ровно один (тот же
/// принцип, что в `WakeWordListener`, откуда взяты кольцевой буфер и насос).
///
/// Эхо собственного голоса гасится дважды:
///
/// 1. voice processing (AEC) на входном узле — `setVoiceProcessingEnabled(true)`;
///    API есть с macOS 10.15, но железо/драйвер может его не поддержать,
///    поэтому вызов обёрнут в `do/catch`, а при отказе слушатель поднимается
///    без AEC (и это честно пишется в журнал);
/// 2. защита в матчере: если фраза, которая звучит из колонок прямо сейчас,
///    сама содержит «стоп»/«хватит», совпадение считается эхом и игнорируется
///    (`shouldIgnore(text:spoken:)`), иначе Ева глушила бы себя собственной
///    репликой «…и стоп.».
///
/// Поток звука — как у wake: realtime-тап только копирует сэмплы в кольцевой
/// буфер (2 с) под коротким замком, в запрос распознавания их переносит своя
/// очередь (25 Гц). Тап не захватывает @MainActor-состояние — иначе Swift 6
/// роняет процесс через dispatch_assert_queue (SIGTRAP, уже ловили).
@MainActor
final class StopCommandListener {
    /// Слушатель действительно держит микрофон и распознавание.
    private(set) var isRunning = false
    /// Запуск уже идёт (права и офлайн-модель проверяются асинхронно).
    var isActive: Bool { isRunning || starting }

    /// Услышали команду: сюда приходит распознанный текст. Вызывающий сразу
    /// останавливает слушатель и глушит озвучку с генерацией.
    var onStop: ((String) -> Void)?
    /// Слушание не поднялось (нет прав, нет офлайн-модели, микрофон занят).
    /// Текст — для журнала и статуса, префикс «стоп:» уже добавлен.
    var onIssue: ((String) -> Void)?
    /// Слушатель поднялся или опустился (диагностика/журнал).
    var onState: ((Bool) -> Void)?
    /// Текущая озвучиваемая фраза — для эхо-защиты. Зовётся только с главного
    /// актора; `nil` — сейчас ничего не звучит.
    var currentSpokenPhrase: (() -> String?)?

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
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var tapInstalled = false
    /// Сколько подряд ошибок распознавания: после нескольких не пытаемся
    /// бесконечно перезапускать задачу, а честно выключаем слушание.
    private var recognitionErrors = 0
    private var fired = false
    /// Токен текущей задачи распознавания: отменённая задача может прислать
    /// ещё один колбэк, и он не должен перезапускать новую.
    private var recognitionToken = 0
    /// Когда последний раз писали в журнал про эхо: частичные результаты идут
    /// десятками в секунду, и одна и та же реплика Евы не должна залить журнал.
    private var echoLoggedAt = Date.distantPast

    /// 2 секунды кольцевого буфера — как у wake: сглаживает паузы главного
    /// потока и стоит меньше мегабайта.
    private let ringSeconds: Double = 2

    // MARK: - Включение и выключение

    func start() {
        guard !isRunning, !starting else { return }
        starting = true
        generation += 1
        let run = generation
        fired = false
        log("стоп: включаю слушание команды")
        Task { [weak self] in
            guard let self else { return }
            let speech = await StopCommandListener.requestSpeechAuthorization()
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
            log("стоп: слушание выключено")
        } else {
            log("стоп: слушание выключено (запуск отменён)")
        }
    }

    // MARK: - Захват

    /// Пытаемся поднять микрофон с эхоподавлением, а если драйвер/железо его не
    /// поддержали — честно повторяем без AEC: лучше слушать без него, чем не
    /// слышать команду вовсе (от собственного «стоп» в тексте всё равно
    /// защищает `shouldIgnore`).
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

        if capture(run: run, voiceProcessing: true) { return }
        guard isRunning == false, starting, generation == run else { return }
        log("стоп: пробую микрофон без эхоподавления")
        if capture(run: run, voiceProcessing: false) { return }
        fail("не удалось включить микрофон для стоп-команды")
    }

    /// Одна попытка поднять `AVAudioEngine`. Возвращает `true`, если слушатель
    /// поднялся и задача распознавания создана.
    private func capture(run: Int, voiceProcessing: Bool) -> Bool {
        let engine = AVAudioEngine()
        let input = engine.inputNode
        if voiceProcessing {
            do {
                try input.setVoiceProcessingEnabled(true)
            } catch {
                log("стоп: voice processing недоступен "
                    + "(\(error.localizedDescription)) — работаю без AEC")
                return false
            }
            guard input.isVoiceProcessingEnabled else {
                log("стоп: voice processing не включился — работаю без AEC")
                return false
            }
        }

        // Формат читаем после включения AEC: voice processing меняет формат
        // входного узла, и тап должен ставиться на актуальный.
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            log("стоп: микрофон не найден (voiceProcessing=\(voiceProcessing))")
            return false
        }
        guard format.commonFormat == .pcmFormatFloat32 else {
            log("стоп: неожиданный формат микрофона "
                + "(commonFormat=\(format.commonFormat.rawValue), "
                + "\(format.channelCount) каналов) — пробую без AEC")
            return false
        }
        // Кольцо всегда моно: AEC открывает всю микрофонную решётку (на ноутбуке
        // это 3+ канала), а `AVAudioFormat(standardFormatWithSampleRate:channels:)`
        // для >2 каналов возвращает nil (проверено). Речь распознавателю моно и
        // нужна; тап при этом ставится на родной формат узла, а `WakeAudioRing`
        // сводит каналы сам.
        guard let ringFormat = AVAudioFormat(
            standardFormatWithSampleRate: format.sampleRate,
            channels: 1
        ) else {
            log("стоп: не удалось подготовить моно-формат микрофона")
            return false
        }
        let capacity = Int(format.sampleRate * ringSeconds)
        let ring = WakeAudioRing(format: ringFormat, capacity: capacity)
        // Колбэк тапа приходит на realtime-очереди аудио, а не на главном
        // акторе. Замыкание @Sendable держит только неизолированную обёртку
        // буфера — @MainActor-состояние слушателя оно не видит.
        let sink: @Sendable (AVAudioPCMBuffer, AVAudioTime) -> Void = { buffer, _ in
            ring.append(buffer)
        }
        input.installTap(onBus: 0, bufferSize: 2048, format: format, block: sink)
        tapInstalled = true

        engine.prepare()
        do {
            try engine.start()
        } catch {
            engine.stop()
            input.removeTap(onBus: 0)
            tapInstalled = false
            log("стоп: движок не запустился (voiceProcessing=\(voiceProcessing)): "
                + error.localizedDescription)
            return false
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
        log("стоп: слушаю команду (микрофон \(Int(format.sampleRate)) Гц, "
            + "вход \(format.channelCount) кан. → моно, on-device, "
            + "эхоподавление=\(voiceProcessing ? "вкл" : "выкл"))")

        beginRecognition(run: run)
        return true
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
        // Команды короткие; подсказка уменьшает ослышки вроде «стоп» →
        // «сто» и помогает распознать «Стоп, Ева» с первого раза.
        request.contextualStrings = ["Стоп, Ева", "Стоп", "Хватит", "Тише"]
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
            if !fired, StopCommandListener.isStopPhrase(text) {
                // Эхо: Ева сейчас произносит фразу со «стоп»/«хватит» — этот
                // же текст приходит с микрофона. Пропускаем, слушаем дальше.
                let spoken = currentSpokenPhrase?()
                if StopCommandListener.shouldIgnore(text: text, spoken: spoken) {
                    let now = Date()
                    if now.timeIntervalSince(echoLoggedAt) > 1.5 {
                        echoLoggedAt = now
                        log("стоп-команда: „\(text)“ — эхо собственной фразы, игнорирую")
                    }
                } else {
                    fired = true
                    log("стоп-команда: услышала „\(text)“")
                    onStop?(text)
                    // `onStop` останавливает слушатель; если по какой-то
                    // причине не остановил — не кормим ту же задачу дальше.
                    return
                }
            }
        }

        if let error {
            recognitionErrors += 1
            log("стоп: ошибка распознавания: \(error)")
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

    private func fail(_ message: String) {
        guard isRunning || starting else { return }
        log("стоп: \(message)")
        onIssue?(message)
        stop()
    }

    // MARK: - Права

    /// Nonisolated по той же причине, что в `SpeechRecorder`: колбэк TCC
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

    // MARK: - Матчер

    /// «Стоп!» → «стоп»; нормализация та же, что у wake-фразы: нижний
    /// регистр, только буквы и пробелы между словами.
    nonisolated static func normalized(_ text: String) -> String {
        WakeWordListener.normalized(text)
    }

    /// Слова-триггеры. Одного слова достаточно: «стоп», «хватит», «тише».
    nonisolated static let triggers: Set<String> = ["стоп", "хватит", "тише"]

    /// Фраза — команда остановки, если в ней есть слово-триггер. Поэтому
    /// «Стоп, Ева», «Стоп!», «хватит», «тише, тише» и даже «Стоп, хватит»
    /// срабатывают одинаково; «останови» и «стоп-кран» не выдумываем — это
    /// отдельные слова, которых в списке нет.
    nonisolated static func isStopPhrase(_ text: String) -> Bool {
        let words = normalized(text).split(separator: " ").map(String.init)
        guard !words.isEmpty else { return false }
        return words.contains { triggers.contains($0) }
    }

    /// Содержит ли фраза, которую Ева произносит прямо сейчас, «стоп» или
    /// «хватит». Именно эти два слова включены в эхо-защиту (по заданию):
    /// «тише» в собственной реплике Евы практически не встречается.
    nonisolated static func speaksStopWord(_ text: String?) -> Bool {
        guard let text else { return false }
        let words = normalized(text).split(separator: " ").map(String.init)
        return words.contains { $0 == "стоп" || $0 == "хватит" }
    }

    /// Совпадение — эхо, а не команда: Ева сама сейчас говорит фразу со
    /// «стоп»/«хватит», и микрофон слышит её из колонок.
    nonisolated static func shouldIgnore(text: String, spoken: String?) -> Bool {
        isStopPhrase(text) && speaksStopWord(spoken)
    }
}
