import AVFoundation
import Foundation
import Speech

/// Самопроверка без окна: прогоняет те же `ChatSession`, `ThinkFilter` и
/// `VoicePlayer.fetchWav`, что и приложение, но печатает результат в stdout.
///
/// Нужна, чтобы проверить протокол `/api/chat` и `/api/tts` на машине без
/// мышки и без разрешений: голосовую часть проверяет человек (см.
/// docs/voice-app.md).
enum SelfTest {
    enum Mode: Sendable, Equatable {
        case chat(String)
        case tts(String)
        case backends
        case speech
        case text
        case wake
        case followUp
        case wakeAudio(Double)
        case version
    }

    @MainActor
    static func run(_ mode: Mode) async -> Int32 {
        switch mode {
        case .chat(let question):
            return await runChat(question)
        case .tts(let text):
            return await runTTS(text)
        case .backends:
            return await runBackends()
        case .speech:
            return runSpeech()
        case .text:
            return runTextTools()
        case .wake:
            return runWakeWord()
        case .followUp:
            return runFollowUp()
        case .wakeAudio(let seconds):
            return await runWakeAudio(seconds: seconds)
        case .version:
            print("EVA voice app 0.1.0 (Swift 6.2, AppKit/SwiftUI-free)")
            return 0
        }
    }

    // MARK: - /api/chat

    @MainActor
    private final class ChatProbe {
        let started = Date()
        var filter = ThinkFilter()
        var deltas = 0
        var firstDelta: Double?
        var hadError = false
        var resumed = false
        var continuation: CheckedContinuation<Int32, Never>?
        let session = ChatSession()

        func handle(_ event: ChatEvent) {
            let elapsed = Date().timeIntervalSince(started)
            switch event {
            case .started(let label, let cold):
                print(String(
                    format: "[%6.2f с] start: %@ cold=%@",
                    elapsed, label, cold.map(String.init) ?? "—"
                ))
            case .delta(let chunk):
                deltas += 1
                if firstDelta == nil { firstDelta = elapsed }
                let visible = filter.feed(chunk)
                if !visible.isEmpty {
                    print(String(
                        format: "[%6.2f с] delta: %@",
                        elapsed, visible.debugDescription
                    ))
                }
            case .stats(let line):
                print(String(format: "[%6.2f с] done: %@", elapsed, line))
            case .failed(let message):
                hadError = true
                print(String(format: "[%6.2f с] ОШИБКА: %@", elapsed, message))
            case .finished:
                filter.flush()
                print("---")
                let first = firstDelta.map { String(format: "%.2f", $0) } ?? "—"
                print("кусков: \(deltas), первый delta через \(first) с")
                print("ответ: \(filter.visible)")
                print("размышления: \(filter.thinking.count) символов (скрыты от глаз и уха)")
                guard !resumed, let continuation else { return }
                resumed = true
                continuation.resume(returning: (filter.visible.isEmpty || hadError) ? 1 : 0)
            }
        }
    }

    @MainActor
    private static func runChat(_ question: String) async -> Int32 {
        guard let server = serverURL() else {
            print("ошибка: не задан адрес сервера")
            return 2
        }
        let model = modelSpec()
        let config = ChatConfig(
            serverURL: server,
            model: model,
            maxTokens: Settings.shared.maxTokens
        )
        print("сервер: \(server.absoluteString)")
        print("модель: \(model.storageValue)")
        print("вопрос: \(question)")
        let probe = ChatProbe()
        // Страховка от зависшего сервера: самопроверка не должна висеть вечно.
        DispatchQueue.global().asyncAfter(deadline: .now() + 300) {
            FileHandle.standardError.write(Data("таймаут 300 с: сервер не закрыл поток\n".utf8))
            exit(4)
        }
        return await withCheckedContinuation { continuation in
            probe.continuation = continuation
            probe.session.send(question: question, config: config) { event in
                probe.handle(event)
            }
        }
    }

    // MARK: - /api/tts

    @MainActor
    private static func runTTS(_ text: String) async -> Int32 {
        guard let server = serverURL() else {
            print("ошибка: не задан адрес сервера")
            return 2
        }
        let voice = Launch.value("--voice") ?? Settings.shared.voice
        print("сервер: \(server.absoluteString), голос: \(voice), текст: \(text)")
        switch await VoicePlayer.fetchWav(text: text, base: server, voice: voice) {
        case .failure(let message):
            print("озвучка: ошибка: \(message)")
            return 1
        case .success(let wav):
            let path = Launch.value("--out") ?? "/tmp/eva-selftest.wav"
            let url = URL(fileURLWithPath: path)
            do {
                try wav.write(to: url)
            } catch {
                print("озвучка: WAV \(wav.count) байт, но файл не записался: \(error)")
                return 1
            }
            let header = String(decoding: wav.prefix(4), as: UTF8.self)
            print("озвучка: \(wav.count) байт, заголовок \(header), файл \(url.path)")
            return header == "RIFF" ? 0 : 1
        }
    }

    // MARK: - /api/backends

    @MainActor
    private static func runBackends() async -> Int32 {
        guard let server = serverURL() else {
            print("ошибка: не задан адрес сервера")
            return 2
        }
        var request = URLRequest(url: server.appendingPathComponent("api/backends"))
        request.timeoutInterval = 8
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            print("GET \(request.url?.absoluteString ?? "?") → HTTP \(code)")
            guard code == 200,
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let backends = object["backends"] as? [[String: Any]]
            else {
                print(String(decoding: data.prefix(400), as: UTF8.self))
                return 1
            }
            for backend in backends {
                let id = backend["id"] as? String ?? "?"
                print("бэкенд \(id):")
                for model in backend["models"] as? [[String: Any]] ?? [] {
                    let modelID = model["id"] as? String ?? "?"
                    print("  Modelspec: \(id)|\(modelID)")
                }
            }
            return 0
        } catch {
            print("сервер не ответил: \(error.localizedDescription)")
            return 1
        }
    }

    // MARK: - Права Apple Speech (без запроса)

    private static func runSpeech() -> Int32 {
        let status = SFSpeechRecognizer.authorizationStatus()
        let statusText: String
        switch status {
        case .authorized: statusText = "authorized"
        case .denied: statusText = "denied"
        case .restricted: statusText = "restricted"
        case .notDetermined: statusText = "notDetermined"
        @unknown default: statusText = "unknown(\(status.rawValue))"
        }
        print("Speech authorization: \(statusText)")
        if let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "ru-RU")) {
            print("ru-RU: available=\(recognizer.isAvailable) "
                + "onDevice=\(recognizer.supportsOnDeviceRecognition)")
        } else {
            print("ru-RU: распознаватель не создался")
        }
        let microphone = AVCaptureDevice.authorizationStatus(for: .audio)
        print("Микрофон: \(microphone.rawValue) (0 — не спрашивали, 3 — разрешён)")
        return 0
    }

    // MARK: - Скрытие <think> и нарезка фраз

    private static func runTextTools() -> Int32 {
        var failures = 0
        func check(_ name: String, _ ok: Bool, _ detail: String = "") {
            print((ok ? "OK   " : "FAIL ") + name + (detail.isEmpty ? "" : " — " + detail))
            if !ok { failures += 1 }
        }

        // Теги разрезаны на куски так, как их отдаёт SSE.
        var filter = ThinkFilter()
        var visible = ""
        for chunk in ["Прив", "ет <thi", "nk>сек", "рет</thi", "nk> мир", "!"] {
            visible += filter.feed(chunk)
        }
        filter.flush()
        check("think вырезан на границах кусков",
              filter.visible == "Привет  мир!" && filter.thinking == "секрет",
              "visible=\(filter.visible.debugDescription) thinking=\(filter.thinking.debugDescription)")
        check("feed отдаёт только прирост", visible == "Привет  мир!",
              "visible=\(visible.debugDescription)")

        // Незакрытый think не должен утекать в ответ.
        var open = ThinkFilter()
        _ = open.feed("Ответ <think>долгое рассуждение без закрытия")
        open.flush()
        check("незакрытый think не виден", open.visible == "Ответ " && open.thinking.contains("долгое"),
              "visible=\(open.visible.debugDescription)")

        // Обычный текст без тегов не теряется.
        var plain = ThinkFilter()
        var plainText = ""
        for chunk in ["раз", " два", " три"] { plainText += plain.feed(chunk) }
        plain.flush()
        check("текст без тегов не теряется", plainText == "раз два три",
              "visible=\(plainText.debugDescription)")

        // Фразы для /api/tts: точки, вопросы, переводы строк.
        var splitter = PhraseSplitter()
        let phrases = splitter.feed("Привет. Как дела?\nВсё хорошо! ") + splitter.flush()
        check("фразы режутся по . ! ? и переводу строки",
              phrases == ["Привет.", "Как дела?", "Всё хорошо!"],
              "\(phrases)")

        // Длинный текст без знаков — режется по словам, каждая часть ≤560.
        var long = PhraseSplitter()
        let sentence = Array(repeating: "слово", count: 400).joined(separator: " ")
        let chunks = long.feed(sentence + " ") + long.flush()
        let longest = chunks.map(\.count).max() ?? 0
        check("длинный текст режется на части ≤560 символов",
              !chunks.isEmpty && longest <= 560 && chunks.count >= 2,
              "частей=\(chunks.count) максимум=\(longest)")

        print(failures == 0 ? "все проверки текста прошли" : "провалено: \(failures)")
        return failures == 0 ? 0 : 1
    }

    // MARK: - Wake-word без микрофона

    /// Проверяет то, что можно проверить без живого микрофона: нормализацию
    /// фразы, триггер и кольцевой буфер. Само распознавание проверяет человек
    /// (сказать «Привет, Ева» — см. docs/voice-app.md).
    private static func runWakeWord() -> Int32 {
        var failures = 0
        func check(_ name: String, _ ok: Bool, _ detail: String = "") {
            print((ok ? "OK   " : "FAIL ") + name + (detail.isEmpty ? "" : " — " + detail))
            if !ok { failures += 1 }
        }

        check("нормализация «Привет, Ева!»",
              WakeWordListener.normalized("Привет, Ева!") == "привет ева",
              WakeWordListener.normalized("Привет, Ева!").debugDescription)

        let phrases = [
            "Привет, Ева!",
            "привет ева",
            "Привет! Ева, ты здесь?",
            "эй ева",
            "Эй, Ева!",
        ]
        for phrase in phrases {
            check("триггер: \(phrase.debugDescription)",
                  WakeWordListener.isWakePhrase(phrase))
        }

        let negatives = [
            "привет",
            "ева",
            "привет как дела",
            "новая ева",
            "привет дева",
            "эй",
            "как тебя зовут",
        ]
        for phrase in negatives {
            check("не триггер: \(phrase.debugDescription)",
                  !WakeWordListener.isWakePhrase(phrase))
        }

        // Кольцевой буфер: порядок сэмплов и переполнение.
        guard let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1) else {
            print("FAIL кольцевой буфер: формат не создался")
            return 1
        }
        let ring = WakeAudioRing(format: format, capacity: 2048)
        for start in [0, 100, 200] {
            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: 100
            ), let data = buffer.floatChannelData else {
                print("FAIL кольцевой буфер: буфер не создался")
                return 1
            }
            buffer.frameLength = 100
            for index in 0..<100 { data[0][index] = Float(start + index) }
            ring.append(buffer)
        }
        guard let drained = ring.take(maxFrames: 1000), let samples = drained.floatChannelData else {
            print("FAIL кольцевой буфер: ничего не отдал")
            return 1
        }
        let values = (0..<Int(drained.frameLength)).map { samples[0][$0] }
        let ordered = Int(drained.frameLength) == 300
            && values.first == 0
            && values.last == 299
            && values[150] == 150
        check("кольцевой буфер: порядок и 300 кадров", ordered,
              "кадров=\(drained.frameLength), первый=\(values.first ?? -1), последний=\(values.last ?? -1)")
        check("кольцевой буфер: после чтения пусто", ring.take() == nil)

        // Переполнение: пишем больше ёмкости — остаются последние кадры.
        let small = WakeAudioRing(format: format, capacity: 1024)
        for start in [0, 1024] {
            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: 700
            ), let data = buffer.floatChannelData else { return 1 }
            buffer.frameLength = 700
            for index in 0..<700 { data[0][index] = Float(start + index) }
            small.append(buffer)
        }
        guard let wrapped = small.take(maxFrames: 4096), let wrappedData = wrapped.floatChannelData else {
            print("FAIL кольцевой буфер: переполнение")
            return 1
        }
        let wrappedValues = (0..<Int(wrapped.frameLength)).map { wrappedData[0][$0] }
        check("кольцевой буфер: переполнение хранит последние 1024 кадра",
              Int(wrapped.frameLength) == 1024
                && wrappedValues.last == 1723
                && wrappedValues.first == 376,
              "кадров=\(wrapped.frameLength), первый=\(wrappedValues.first ?? -1), последний=\(wrappedValues.last ?? -1)")

        print(failures == 0 ? "все проверки wake-слова прошли" : "провалено: \(failures)")
        return failures == 0 ? 0 : 1
    }

    // MARK: - Follow-up без микрофона

    /// Проверяет правила задачи 0190, которые не требуют живого микрофона:
    /// пауза тишины в follow-up, приоритет обычного авто-стопа после речи и
    /// запрет автоматики при выключенном ожидании «Привет, Ева».
    @MainActor
    private static func runFollowUp() -> Int32 {
        var failures = 0
        func check(_ name: String, _ ok: Bool, _ detail: String = "") {
            print((ok ? "OK   " : "FAIL ") + name + (detail.isEmpty ? "" : " — " + detail))
            if !ok { failures += 1 }
        }

        check("«слушать после ответа» по умолчанию включено",
              Settings.defaultFollowUpListen)
        check("варианты паузы: 8/12/20 с",
              Settings.followUpSilenceOptions == [8, 12, 20],
              "\(Settings.followUpSilenceOptions)")
        check("пауза по умолчанию 12 с", Settings.defaultFollowUpSilence == 12)

        let afterSpeech = 2.2
        let warmup = 6.0
        func timeout(hasText: Bool, followUp: Bool, silence: Double) -> Double {
            SpeechRecorder.silenceTimeout(
                hasText: hasText,
                followUp: followUp,
                followUpSilence: silence,
                warmup: warmup,
                afterSpeech: afterSpeech
            )
        }

        check("follow-up без слов ждёт паузу из настроек",
              timeout(hasText: false, followUp: true, silence: 12) == 12,
              "\(timeout(hasText: false, followUp: true, silence: 12))")
        check("follow-up с паузой 8 с", timeout(hasText: false, followUp: true, silence: 8) == 8)
        check("follow-up с паузой 20 с", timeout(hasText: false, followUp: true, silence: 20) == 20)
        check("follow-up после речи — обычный авто-стоп 2,2 с",
              timeout(hasText: true, followUp: true, silence: 12) == afterSpeech)
        check("обычная запись без слов — 6 с",
              timeout(hasText: false, followUp: false, silence: 12) == warmup)
        check("обычная запись после речи — 2,2 с",
              timeout(hasText: true, followUp: false, silence: 12) == afterSpeech)
        check("нулевая пауза не превращается в мгновенное закрытие",
              timeout(hasText: false, followUp: true, silence: 0) >= 0.5)

        check("follow-up + ожидание → автоматика разрешена",
              AppDelegate.followUpAllowed(followUpListen: true, wakeWord: true))
        check("follow-up без ожидания «Привет, Ева» запрещён",
              !AppDelegate.followUpAllowed(followUpListen: true, wakeWord: false))
        check("выключенный follow-up запрещён",
              !AppDelegate.followUpAllowed(followUpListen: false, wakeWord: true))
        check("после ошибки ответа follow-up не открывается",
              !AppDelegate.followUpAllowed(
                  followUpListen: true,
                  wakeWord: true,
                  answerFailed: true
              ))

        print(failures == 0 ? "все проверки follow-up прошли" : "провалено: \(failures)")
        return failures == 0 ? 0 : 1
    }

    // MARK: - Аудио-путь ожидания без распознавания

    /// Держит микрофон, кольцевой буфер и насос N секунд, но не создаёт
    /// `SFSpeechRecognizer` — поэтому не требует прав на распознавание речи и
    /// не показывает системный запрос. Нужен, чтобы замерить CPU и убедиться,
    /// что `AVAudioEngine` честно останавливается и отпускает микрофон.
    @MainActor
    private static func runWakeAudio(seconds: Double) async -> Int32 {
        let microphone = AVCaptureDevice.authorizationStatus(for: .audio)
        guard microphone == .authorized else {
            print("микрофон недоступен (статус \(microphone.rawValue)); "
                + "дайте Еве доступ к микрофону в Системных настройках")
            return 1
        }
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0,
              let standard = AVAudioFormat(
                  standardFormatWithSampleRate: format.sampleRate,
                  channels: format.channelCount
              )
        else {
            print("микрофон не найден (нет входного устройства)")
            return 1
        }
        let ring = WakeAudioRing(
            format: standard,
            capacity: Int(format.sampleRate * 2)
        )
        let sink: @Sendable (AVAudioPCMBuffer, AVAudioTime) -> Void = { buffer, _ in
            ring.append(buffer)
        }
        input.installTap(onBus: 0, bufferSize: 2048, format: format, block: sink)
        engine.prepare()
        let cpuBefore = cpuSeconds()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            print("не удалось включить микрофон: \(error.localizedDescription)")
            return 1
        }
        // Запрос-заглушка: насос переносит звук в неё, чтобы путь был тем же,
        // что и в боевом слушателе (без SFSpeech).
        let box = WakeRequestBox()
        box.set(nil)
        let pump = WakeAudioPump(ring: ring, request: box)
        pump.start()
        print("аудио-путь ожидания: \(Int(format.sampleRate)) Гц, "
            + "каналов \(format.channelCount), \(String(format: "%.0f", seconds)) с")
        let started = Date()
        try? await Task.sleep(nanoseconds: UInt64(max(seconds, 0.5) * 1_000_000_000))
        pump.stop()
        engine.stop()
        input.removeTap(onBus: 0)
        let elapsed = Date().timeIntervalSince(started)
        let cpu = cpuSeconds() - cpuBefore
        let frames = ring.writtenFrames
        let rate = elapsed > 0 ? Double(frames) / elapsed : 0
        let percent = elapsed > 0 ? cpu / elapsed * 100 : 0
        print("кадров: \(frames) за \(String(format: "%.1f", elapsed)) с "
            + "(\(Int(rate)) кадр/с), движок остановлен: \(!engine.isRunning)")
        print("CPU процесса за интервал: \(String(format: "%.2f", percent)) % "
            + "(пользователь+\(String(format: "%.2f", cpu)) с)")
        return frames > 0 && !engine.isRunning ? 0 : 1
    }

    /// Пользовательское и системное время процесса: свой замер CPU без
    /// внешних `top`/`ps` (в этой среде они недоступны).
    private static func cpuSeconds() -> Double {
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        return Double(usage.ru_utime.tv_sec) + Double(usage.ru_utime.tv_usec) / 1_000_000
            + Double(usage.ru_stime.tv_sec) + Double(usage.ru_stime.tv_usec) / 1_000_000
    }

    // MARK: - Значения

    @MainActor
    private static func serverURL() -> URL? {
        if let override = Launch.serverOverride { return Settings.normalize(override) }
        return Settings.shared.baseURL
    }

    @MainActor
    private static func modelSpec() -> ModelSpec {
        if let override = Launch.modelOverride, let model = ModelSpec.parse(override) {
            return model
        }
        return Settings.shared.model
    }
}
