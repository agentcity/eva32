import AVFoundation
import Foundation
import Speech

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

    private var generation = 0
    private var engine: AVAudioEngine?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var recognizer: SFSpeechRecognizer?
    private var tapInstalled = false
    private var lastText = ""
    private var delivered = true
    private var timeout: Task<Void, Never>?

    /// Дольше минуты не слушаем: забытый микрофон — это и батарея, и приватность.
    private let maxDuration: UInt64 = 60

    func start() {
        guard !isRecording else { return }
        generation += 1
        let run = generation
        isRecording = true
        lastText = ""
        delivered = false
        onState?(true)
        onPartial?("")
        log("речь: старт записи")

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
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
            request.append(buffer)
        }
        tapInstalled = true
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

    // MARK: - Завершение

    private func finish(error: String?) {
        guard isRecording else { return }
        isRecording = false
        generation += 1
        timeout?.cancel()
        timeout = nil
        teardown()
        onState?(false)

        let text = lastText.trimmingCharacters(in: .whitespacesAndNewlines)
        log("речь: стоп, распознано \(text.count) символов")
        if !delivered {
            delivered = true
            if text.isEmpty {
                onError?(error.map { "распознавание: \($0)" } ?? "не расслышала — попробуйте ещё раз")
                return
            }
            onFinal?(text)
        }
    }

    private func fail(_ message: String) {
        guard isRecording else { return }
        isRecording = false
        generation += 1
        timeout?.cancel()
        timeout = nil
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
        request?.endAudio()
        task?.cancel()
        engine = nil
        request = nil
        task = nil
        recognizer = nil
        tapInstalled = false
    }

    // MARK: - Права

    @MainActor
    private static func requestSpeechAuthorization() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
    }
}
