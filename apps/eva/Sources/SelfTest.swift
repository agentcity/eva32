import AVFoundation
import AppKit
import CoreGraphics
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
        case stop
        case wakeAudio(Double)
        case compact
        case images
        case pauses
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
        case .stop:
            return runStopCommand()
        case .wakeAudio(let seconds):
            return await runWakeAudio(seconds: seconds)
        case .compact:
            return runCompact()
        case .images:
            return await runImages()
        case .pauses:
            return await runPauses()
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
        var images = 0
        var silent = false
        var plugin = false
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
            case .image(let url, let alt):
                images += 1
                print(String(
                    format: "[%6.2f с] image #%d: %@ alt=%@",
                    elapsed, images, url, alt
                ))
            case .pluginAnswer:
                plugin = true
                print(String(
                    format: "[%6.2f с] plugin: ответ плагина — паузы сказки (0230)",
                    elapsed
                ))
            case .silent(let flag):
                silent = silent || flag
                print(String(
                    format: "[%6.2f с] silent: ответ озвучен записью — в /api/tts не отправляем",
                    elapsed
                ))
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
                print("image-событий: \(images)")
                print("источник: " + (plugin ? "plugin (сказка — паузы длиннее)" : "модель"))
                print(silent
                    ? "озвучка: пропущена (silent)"
                    : "озвучка: не проверялась (в ответе нет признака silent)")
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

    // MARK: - «Стоп, Ева» без микрофона (задача 0194)

    /// Проверяет то, что можно проверить без живого микрофона и не «глуша»
    /// настоящее приложение: нормализацию и матчер команд, эхо-защиту от
    /// собственной речи, план «какой слушатель держит микрофон» (ровно один)
    /// и настройку. Живой микрофон — за человеком (docs/voice-app.md).
    @MainActor
    private static func runStopCommand() -> Int32 {
        var failures = 0
        func check(_ name: String, _ ok: Bool, _ detail: String = "") {
            print((ok ? "OK   " : "FAIL ") + name + (detail.isEmpty ? "" : " — " + detail))
            if !ok { failures += 1 }
        }

        check("«Слушать „Стоп, Ева“» по умолчанию включена",
              Settings.defaultStopCommand)
        check("ключ настройки — eva.stopCommand",
              Settings.stopCommandKey == "eva.stopCommand",
              Settings.stopCommandKey)

        check("нормализация «Стоп, Ева!»",
              StopCommandListener.normalized("Стоп, Ева!") == "стоп ева",
              StopCommandListener.normalized("Стоп, Ева!").debugDescription)

        let commands = [
            "Стоп",
            "Стоп!",
            "стоп, ева",
            "Стоп, Ева!",
            "Ева, стоп",
            "Хватит",
            "хватит!",
            "Тише",
            "тише, тише",
            "Стоп, хватит",
        ]
        for phrase in commands {
            check("команда: \(phrase.debugDescription)",
                  StopCommandListener.isStopPhrase(phrase))
        }

        let notCommands = [
            "",
            "привет ева",
            "как дела",
            "останови",
            "стопкран",
            "продолжай",
        ]
        for phrase in notCommands {
            check("не команда: \(phrase.debugDescription)",
                  !StopCommandListener.isStopPhrase(phrase))
        }

        // Эхо: Ева сама говорит «стоп»/«хватит» — её голос из колонок не
        // должен глушить озвучку.
        check("эхо: «стоп» в текущей фразе Евы → игнорирую",
              StopCommandListener.shouldIgnore(
                  text: "стоп",
                  spoken: "Сейчас остановлюсь. Стоп."
              ))
        check("эхо: «хватит» в текущей фразе Евы → игнорирую",
              StopCommandListener.shouldIgnore(
                  text: "хватит",
                  spoken: "Хватит на сегодня."
              ))
        check("не эхо: обычная фраза Евы → команда работает",
              !StopCommandListener.shouldIgnore(
                  text: "стоп ева",
                  spoken: "Привет! Как дела?"
              ))
        check("не эхо: тишина → команда работает",
              !StopCommandListener.shouldIgnore(text: "стоп", spoken: nil))
        check("не команда: «привет ева» не глушит даже при «стоп» в речи",
              !StopCommandListener.shouldIgnore(
                  text: "привет ева",
                  spoken: "Стоп."
              ))
        check("«тише» в речи Евы не относится к эхо-защите",
              !StopCommandListener.shouldIgnore(
                  text: "тише",
                  spoken: "Говори тише, пожалуйста."
              ))

        // План слушателей: wake и stop никогда не вместе.
        func plan(
            speak: Bool = false,
            thinking: Bool = false,
            recording: Bool = false,
            error: Bool = false,
            wakeWord: Bool = true,
            stopCommand: Bool = true,
            wakeSuppressed: Bool = false,
            stopSuppressed: Bool = false,
            followUp: Bool = false
        ) -> EvaListenerPlan {
            AppDelegate.listenerPlan(
                wakeWord: wakeWord,
                stopCommand: stopCommand,
                wakeSuppressed: wakeSuppressed,
                stopSuppressed: stopSuppressed,
                isRecording: recording,
                isSpeaking: speak,
                isThinking: thinking,
                followUpArmed: followUp,
                errorActive: error
            )
        }

        let idle = plan()
        check("в покое микрофон у wake-ожидания",
              idle.wake && !idle.stop,
              "wake=\(idle.wake), stop=\(idle.stop)")

        let speaking = plan(speak: true)
        check("во время озвучки микрофон у стоп-слушателя",
              speaking.stop && !speaking.wake,
              "wake=\(speaking.wake), stop=\(speaking.stop)")

        let thinking = plan(thinking: true)
        check("пока «думаю» — стоп-слушатель уже слушает",
              thinking.stop && !thinking.wake,
              "wake=\(thinking.wake), stop=\(thinking.stop)")

        let mix = plan(speak: true, thinking: true)
        check("«думаю» + озвучка — по-прежнему один слушатель",
              mix.stop && !mix.wake,
              "wake=\(mix.wake), stop=\(mix.stop)")

        let disabled = plan(speak: true, stopCommand: false)
        check("галочка снята — во время речи микрофон не держим",
              !disabled.wake && !disabled.stop,
              "wake=\(disabled.wake), stop=\(disabled.stop)")

        let recording = plan(speak: true, recording: true)
        check("идёт запись — ни один слушатель не работает",
              !recording.wake && !recording.stop,
              "wake=\(recording.wake), stop=\(recording.stop)")

        let failure = plan(speak: true, error: true)
        check("вспышка ошибки — ни один слушатель не работает",
              !failure.wake && !failure.stop,
              "wake=\(failure.wake), stop=\(failure.stop)")

        let followUpWaiting = plan(followUp: true)
        check("follow-up вооружён — wake ждёт, стоп не слушает",
              !followUpWaiting.wake && !followUpWaiting.stop,
              "wake=\(followUpWaiting.wake), stop=\(followUpWaiting.stop)")

        let wakeDenied = plan(wakeSuppressed: true)
        check("wake не поднялся из-за прав — стоп-слушатель не глушит его навсегда",
              !wakeDenied.wake && !wakeDenied.stop,
              "wake=\(wakeDenied.wake), stop=\(wakeDenied.stop)")

        let stopDenied = plan(speak: true, stopSuppressed: true)
        check("стоп-слушатель не поднялся — wake не занимает его место во время речи",
              !stopDenied.wake && !stopDenied.stop,
              "wake=\(stopDenied.wake), stop=\(stopDenied.stop)")

        // Перебор всех сочетаний: два микрофонных движка не встречаются.
        var both = 0
        for speak in [false, true] {
            for thinking in [false, true] {
                for recording in [false, true] {
                    for error in [false, true] {
                        for wakeWord in [false, true] {
                            for stopCommand in [false, true] {
                                for followUp in [false, true] {
                                    let p = plan(
                                        speak: speak,
                                        thinking: thinking,
                                        recording: recording,
                                        error: error,
                                        wakeWord: wakeWord,
                                        stopCommand: stopCommand,
                                        followUp: followUp
                                    )
                                    if p.wake && p.stop { both += 1 }
                                }
                            }
                        }
                    }
                }
            }
        }
        check("ни одно сочетание флагов не включает оба движка", both == 0,
              "сочетаний с двумя движками: \(both)")

        // Матчер — это несколько сравнений слов; 1000 вызовов должны уложиться
        // в бюджет отклика (<300 мс) с большим запасом.
        let started = Date()
        var hits = 0
        for _ in 0..<1000 {
            if StopCommandListener.isStopPhrase("Стоп, Ева!") { hits += 1 }
        }
        let elapsed = Date().timeIntervalSince(started)
        check("1000 проверок фразы быстрее 300 мс",
              hits == 1000 && elapsed < 0.3,
              String(format: "%.1f мс", elapsed * 1000))

        // Микрофонная решётка Mac при включённом voice processing отдаёт 3+
        // канала, а кольцо стоп-слушателя всегда моно. `AVAudioFormat(...,
        // channels: 3)` для такого формата возвращает nil, поэтому проверяем
        // именно сведение каналов в кольце — без микрофона.
        if let layout = AVAudioChannelLayout(
            layoutTag: kAudioChannelLayoutTag_DiscreteInOrder | 3
        ) {
            let inputFormat = AVAudioFormat(
                standardFormatWithSampleRate: 48_000,
                channelLayout: layout
            )
            if let mono = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1),
               let buffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: 4),
               let data = buffer.floatChannelData {
                buffer.frameLength = 4
                for channel in 0..<3 {
                    for index in 0..<4 { data[channel][index] = Float(channel + 1) }
                }
                let ring = WakeAudioRing(format: mono, capacity: 1024)
                ring.append(buffer)
                if let out = ring.take(maxFrames: 16), let samples = out.floatChannelData {
                    let values = (0..<Int(out.frameLength)).map { samples[0][$0] }
                    check("3 канала микрофона сводятся в моно",
                          values.count == 4 && values.allSatisfy { abs($0 - 2) < 0.0001 },
                          "\(values)")
                    check("распознавателю уходит моно 48 кГц",
                          out.format.channelCount == 1 && out.format.sampleRate == 48_000,
                          "каналов=\(out.format.channelCount), Гц=\(out.format.sampleRate)")
                } else {
                    check("моно-кольцо отдало буфер", false)
                }
            } else {
                check("моно-формат и 3-канальный буфер создались", false)
            }
        } else {
            check("канальный layout для 3 каналов создался", false)
        }

        print(failures == 0 ? "все проверки стоп-команды прошли" : "провалено: \(failures)")
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

    // MARK: - Картинки сказки и silent (задача 0224)

    /// AppKit-часть проверки картинок: сам вид и окно ответа. Синхронная, без
    /// `await` — так её тело гарантированно исполняется на главном потоке.
    @MainActor
    private static func runImagesUI() -> Int {
        var failures = 0
        func check(_ name: String, _ ok: Bool, _ detail: String = "") {
            print((ok ? "OK   " : "FAIL ") + name + (detail.isEmpty ? "" : " — " + detail))
            fflush(stdout)
            if !ok { failures += 1 }
        }

        // Вид: показ, смена (кроссфейд) и очистка без окна.
        let view = TaleImageView(frame: NSRect(x: 0, y: 0, width: 190, height: 240))
        let image = NSImage(size: NSSize(width: 20, height: 30))
        view.show(image, alt: "Колобок", animated: false)
        check("картинка показана, подпись на месте",
              view.hasImage && view.currentImage != nil && view.toolTip == "Колобок")
        view.show(image, alt: "Лиса", animated: false)
        check("смена картинки обновляет подпись", view.altText == "Лиса")
        view.clear()
        check("очистка убирает картинку", !view.hasImage && view.currentImage == nil)
        print("вид: картинка над текстом, доля высоты "
            + "\(MainWindowController.taleImageHeightRatio), "
            + "кроссфейд \(TaleImageView.fadeDuration) с")

        // Окно ответа: картинка рядом с текстом, новый вопрос её убирает.
        // Окно не показываем — вёрстку видно и без WindowServer; речь не
        // трогаем вовсе, поэтому TCC-запросов нет.
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        if !app.isRunning { app.finishLaunching() }
        let windowController = MainWindowController(circles: CircleHub())
        windowController.showTaleImage(image, alt: "Колобок")
        let shown = windowController.taleImageState
        check("окно ответа: картинка видна и занимает большую часть поля",
              shown.visible && shown.width > 0,
              "visible=\(shown.visible), height=\(shown.width)")
        windowController.beginAnswer()
        let cleared = windowController.taleImageState
        check("окно ответа: новый вопрос убирает картинку",
              !cleared.visible && cleared.width == 0,
              "visible=\(cleared.visible), width=\(cleared.width)")
        return failures
    }

    /// Проверяет без окна, сети и звука: разбор SSE `image`/`silent`, сборку
    /// абсолютного URL картинки, правило «silent → не озвучиваем», кэш,
    /// поведение на битом URL и сам вид `TaleImageView` (показ/смена/очистка),
    /// а также темп смены картинок 15 символов/с с подменёнными часами.
    @MainActor
    private static func runImages() async -> Int32 {
        var failures = 0
        func check(_ name: String, _ ok: Bool, _ detail: String = "") {
            print((ok ? "OK   " : "FAIL ") + name + (detail.isEmpty ? "" : " — " + detail))
            fflush(stdout)
            if !ok { failures += 1 }
        }

        // 0. AppKit-часть — первой и без `await`: окно можно создавать только
        // на настоящем главном потоке, а в процессе с `dispatchMain()`
        // продолжение после await приходит на потоке пула.
        failures += runImagesUI()

        // 1. Настройка «Показывать картинки сказок».
        check("ключ настройки — eva.showTaleImages",
              Settings.showTaleImagesKey == "eva.showTaleImages",
              Settings.showTaleImagesKey)
        check("картинки сказок по умолчанию включены", Settings.defaultShowTaleImages)

        // 2. Разбор SSE-события image.
        let imageLine = #"data: {"type":"image","side":0,"url":"/media/fairy-tales/kolobok/01.jpg","alt":"Колобок"}"#
        let imageEvents = ChatSession.parse(line: imageLine)
        check("image-событие разобрано",
              imageEvents == [.image(url: "/media/fairy-tales/kolobok/01.jpg", alt: "Колобок")],
              "\(imageEvents)")
        check("image без side относится к колонке 0",
              ChatSession.parse(line: #"data: {"type":"image","url":"/media/x.jpg"}"#)
                == [.image(url: "/media/x.jpg", alt: "")])
        check("image второй колонки игнорируется",
              ChatSession.parse(line: #"data: {"type":"image","side":1,"url":"/media/x.jpg"}"#).isEmpty)
        check("image без url — просто нет события",
              ChatSession.parse(line: #"data: {"type":"image","side":0,"alt":"без адреса"}"#).isEmpty)
        check("image с пустым url — нет события",
              ChatSession.parse(line: #"data: {"type":"image","side":0,"url":"   "}"#).isEmpty)
        check("битый JSON не роняет парсер",
              ChatSession.parse(line: #"data: {"type":"image", "url": "#).isEmpty)
        check("не-SSE строка игнорируется",
              ChatSession.parse(line: ": keep-alive").isEmpty)
        check("[DONE] закрывает поток",
              ChatSession.parse(line: "data: [DONE]") == [.finished])

        // 3. silent: признак приходит раньше текста (задача 0223/0224).
        let silentDelta = ChatSession.parse(
            line: #"data: {"type":"delta","side":0,"text":"Слушай","silent":true}"#
        )
        check("silent из delta приходит раньше текста",
              silentDelta == [.silent(true), .delta("Слушай")],
              "\(silentDelta)")
        check("silent из done тоже виден",
              ChatSession.parse(line: #"data: {"type":"done","side":0,"silent":true}"#).first
                == .silent(true))
        check("delta без silent не поднимает признак",
              ChatSession.parse(line: #"data: {"type":"delta","side":0,"text":"Привет"}"#)
                == [.delta("Привет")])
        check("правило: silent → в /api/tts не отправляем",
              !AnswerController.shouldSpeak(voiceOn: true, silent: true))
        check("правило: обычный ответ озвучивается",
              AnswerController.shouldSpeak(voiceOn: true, silent: false))
        check("правило: выключенная озвучка молчит",
              !AnswerController.shouldSpeak(voiceOn: false, silent: false))

        // 4. URL картинки относительно адреса сервера.
        let base = URL(string: "http://127.0.0.1:8765")!
        check("относительный URL картинки — от сервера",
              AnswerController.imageURL(
                  setting: true,
                  url: "/media/fairy-tales/kolobok/01.jpg",
                  base: base
              )?.absoluteString == "http://127.0.0.1:8765/media/fairy-tales/kolobok/01.jpg")
        check("абсолютный URL не переписывается",
              AnswerController.imageURL(
                  setting: true,
                  url: "https://example.com/a.jpg",
                  base: base
              )?.absoluteString == "https://example.com/a.jpg")
        check("выключенная настройка — картинки нет",
              AnswerController.imageURL(setting: false, url: "/media/x.jpg", base: base) == nil)
        check("битый URL без хоста не роняет показ",
              AnswerController.imageURL(setting: true, url: "http://", base: base) == nil
                && AnswerController.imageURL(setting: true, url: "://", base: base) == nil
                && AnswerController.imageURL(setting: true, url: "", base: base) == nil)
        check("чужой протокол отброшен",
              AnswerController.imageURL(setting: true, url: "file:///etc/passwd", base: base) == nil
                && AnswerController.imageURL(
                    setting: true, url: "javascript:alert(1)", base: base
                ) == nil)
        check("без адреса сервера картинку не показываем",
              AnswerController.imageURL(setting: true, url: "/media/x.jpg", base: nil) == nil)

        // 5. Смена картинок по прогрессу чтения (задача 0227). Часы и темп
        // подменяются числами: реального ожидания 20/40/60 с нет.
        let positions = [300, 600, 900]
        func timerPacer() -> TaleImagePacer {
            var pace = TaleImagePacer(mode: .timer)
            for (index, position) in positions.enumerated() {
                pace.enqueue(TaleImagePacer.Item(
                    order: index + 1,
                    afterVisible: position,
                    afterVoice: position * 7 / 10
                ))
            }
            return pace
        }
        func dueCount(_ pace: TaleImagePacer, elapsed: Double, rate: Double = 15) -> Int {
            var pace = pace
            pace.updateTimer(elapsed: elapsed, rate: rate)
            var count = 0
            while let item = pace.next, pace.isReached(item) {
                count += 1
                pace.markShown()
            }
            return count
        }
        check("темп 15 симв/с: 300 символов — около 20 с",
              dueCount(timerPacer(), elapsed: 19.9) == 0
                && dueCount(timerPacer(), elapsed: 20.0) == 1,
              "19,9 с → нет, 20,0 с → первая")
        check("темп 15 симв/с: 600 символов — около 40 с",
              dueCount(timerPacer(), elapsed: 39.9) == 1
                && dueCount(timerPacer(), elapsed: 40.0) == 2,
              "39,9 с → одна, 40,0 с → две")
        check("темп 15 симв/с: 900 символов — около 60 с",
              dueCount(timerPacer(), elapsed: 59.9) == 2
                && dueCount(timerPacer(), elapsed: 60.0) == 3,
              "59,9 с → две, 60,0 с → три")
        check("новая картинка не появляется раньше своей позиции",
              dueCount(timerPacer(), elapsed: 5) == 0
                && dueCount(timerPacer(), elapsed: 19.999) == 0)
        check("темп можно подменить: 30 симв/с → 10/20/30 с",
              dueCount(timerPacer(), elapsed: 9.9, rate: 30) == 0
                && dueCount(timerPacer(), elapsed: 10.0, rate: 30) == 1
                && dueCount(timerPacer(), elapsed: 20.0, rate: 30) == 2
                && dueCount(timerPacer(), elapsed: 30.0, rate: 30) == 3)
        var resetPace = timerPacer()
        resetPace.updateTimer(elapsed: 60)
        resetPace.reset(mode: .speech)
        check("сброс (стоп/новый вопрос) очищает очередь и прогресс",
              resetPace.next == nil && resetPace.shown == 0 && resetPace.spoken == 0)

        var speechPace = TaleImagePacer(mode: .speech)
        for (index, position) in positions.enumerated() {
            speechPace.enqueue(TaleImagePacer.Item(
                order: index + 1,
                afterVisible: position,
                afterVoice: position * 7 / 10
            ))
        }
        func speechDue(_ chars: Int) -> Int {
            var pace = speechPace
            pace.updateSpoken(Double(chars))
            var count = 0
            while let item = pace.next, pace.isReached(item) {
                count += 1
                pace.markShown()
            }
            return count
        }
        // afterVoice у позиций 300/600/900 — 210/420/630 символов.
        check("речь: картинка ждёт прогресс озвучки, а не приход события",
              speechDue(209) == 0 && speechDue(210) == 1,
              "209 симв. → нет, 210 симв. → первая")
        check("речь: следующие картинки по своим позициям",
              speechDue(419) == 1 && speechDue(420) == 2
                && speechDue(629) == 2 && speechDue(630) == 3)
        check("речь: прогресс назад не откатывает картинки",
              {
                  var pace = speechPace
                  pace.updateSpoken(630)
                  pace.markShown()
                  pace.markShown()
                  pace.markShown()
                  pace.updateSpoken(10)
                  return pace.shown == 3
              }(),
              "три показаны, после отката всё ещё три")

        check("оценка речи внутри фразы: ~15 символов/с",
              VoicePlayer.spokenEstimate(elapsed: 0, chars: 40) == 0
                && VoicePlayer.spokenEstimate(elapsed: 1, chars: 40) == 15
                && VoicePlayer.spokenEstimate(elapsed: 2, chars: 40) == 30)
        check("оценка речи не перескакивает длину фразы",
              VoicePlayer.spokenEstimate(elapsed: 100, chars: 40) == 40)
        check("темп речи можно подменить в проверке",
              VoicePlayer.spokenEstimate(elapsed: 1, chars: 100, rate: 30) == 30)

        // Хвост, ещё не ставший фразой, тоже учитывается в позиции картинки:
        // иначе событие `image` посреди фразы показало бы её раньше речи.
        var pendingSplitter = PhraseSplitter()
        let splitPhrases = pendingSplitter.feed("Раз. Два три")
        check("незавершённая фраза учтена в позиции картинки",
              splitPhrases == ["Раз."] && pendingSplitter.pendingChars == 7,
              "\(splitPhrases), хвост \(pendingSplitter.pendingChars)")

        // 6. Кэш в памяти и распознавание формата.
        let cachedURL = URL(string: "http://127.0.0.1:8765/media/fairy-tales/kolobok/02.jpg")!
        let jpeg = Data([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10])
        TaleImageCache.shared.store(jpeg, for: cachedURL)
        check("кэш по url отдаёт те же данные",
              TaleImageCache.shared.data(for: cachedURL) == jpeg)
        check("JPEG распознан по заголовку", TaleImageLoader.looksLikeImage(jpeg))
        check("HTML не считается картинкой",
              !TaleImageLoader.looksLikeImage(Data("<html>404</html>".utf8)))
        check("загрузка: две попытки, не больше",
              TaleImageLoader.attempts == 2, "\(TaleImageLoader.attempts)")

        // 7. Недоступный URL: nil, без падения и без зависания ответа.
        let dead = URL(string: "http://127.0.0.1:1/media/fairy-tales/nope.jpg")!
        let started = Date()
        let loaded = await TaleImageLoader.load(dead)
        check("недоступный URL — nil без падения",
              loaded == nil,
              String(format: "%.2f с", Date().timeIntervalSince(started)))

        print(failures == 0 ? "все проверки картинок прошли" : "провалено: \(failures)")
        return failures == 0 ? 0 : 1
    }

    // MARK: - Паузы чтения (задача 0230)

    /// Проверяет без звука, сети и реального ожидания: расчёт паузы по тексту
    /// (обычная фраза, абзац, режим сказки), настройку «скорость чтения
    /// сказок», флаг абзаца в нарезке, разбор `finish_reason=plugin`,
    /// мгновенное прерывание паузы стопом и отсутствие пауз без озвучки.
    @MainActor
    private static func runPauses() async -> Int32 {
        var failures = 0
        func check(_ name: String, _ ok: Bool, _ detail: String = "") {
            print((ok ? "OK   " : "FAIL ") + name + (detail.isEmpty ? "" : " — " + detail))
            fflush(stdout)
            if !ok { failures += 1 }
        }
        func approx(_ value: TimeInterval, _ expected: TimeInterval) -> Bool {
            abs(value - expected) < 0.001
        }

        // 1. База обычного чата: 350 мс после фразы, 900 мс после абзаца.
        let phrase = ReadingPause.duration(text: "Колобок покатился.", tale: false, slowTale: true)
        let paragraph = ReadingPause.duration(text: "Колобок покатился.\n\n", tale: false, slowTale: true)
        check("обычная фраза — 350 мс", approx(phrase, 0.35),
              String(format: "%.0f мс", phrase * 1000))
        check("конец абзаца — 900 мс", approx(paragraph, 0.9),
              String(format: "%.0f мс", paragraph * 1000))
        check("абзац распознан по пустой строке",
              ReadingPause.isParagraph("Фраза.\n\n")
                && ReadingPause.isParagraph("Фраза.\r\n\r\n")
                && !ReadingPause.isParagraph("Фраза.")
                && !ReadingPause.isParagraph("Фраза.\nСледующая строка."))
        check("вид паузы по тексту",
              ReadingPause.kind(for: "Фраза.") == .phrase
                && ReadingPause.kind(for: "Фраза.\n\n") == .paragraph)
        check("флаг абзаца важнее текста (текст уже обрезан нарезкой)",
              approx(
                  ReadingPause.duration(paragraph: true, tale: false, slowTale: true),
                  0.9
              )
                && approx(
                    ReadingPause.duration(paragraph: false, tale: false, slowTale: true),
                    0.35
                ))

        // 2. Сказка: на «медленной» настройке паузы ×1,8 (≈630/1620 мс).
        let talePhrase = ReadingPause.duration(text: "Колобок покатился.", tale: true, slowTale: true)
        let taleParagraph = ReadingPause.duration(text: "Колобок покатился.\n\n", tale: true, slowTale: true)
        check("сказка, медленная: фраза ≈630 мс", approx(talePhrase, 0.63),
              String(format: "%.0f мс", talePhrase * 1000))
        check("сказка, медленная: абзац ≈1620 мс", approx(taleParagraph, 1.62),
              String(format: "%.0f мс", taleParagraph * 1000))
        check("множитель сказки — 1,8", ReadingPause.taleFactor == 1.8)
        check("сказка, обычная скорость — паузы как в чате",
              approx(
                  ReadingPause.duration(text: "Фраза.", tale: true, slowTale: false),
                  0.35
              )
                && approx(
                    ReadingPause.duration(text: "Фраза.\n\n", tale: true, slowTale: false),
                    0.9
                ))
        check("обычный чат не зависит от настройки сказок",
              approx(
                  ReadingPause.duration(text: "Фраза.", tale: false, slowTale: true),
                  0.35
              )
                && approx(
                    ReadingPause.duration(text: "Фраза.", tale: false, slowTale: false),
                    0.35
                ))

        // 3. Настройка «Скорость чтения сказок».
        check("ключ настройки — eva.talePace", Settings.talePaceKey == "eva.talePace",
              Settings.talePaceKey)
        check("по умолчанию сказки читаются медленно", Settings.defaultTalePace == .slow)
        check("значения настройки: обычная/медленная",
              TalePace.normalize("normal") == .normal
                && TalePace.normalize("slow") == .slow)
        check("пустое/чужое значение — медленная (детский режим)",
              TalePace.normalize(nil) == .slow && TalePace.normalize("быстро") == .slow)

        // 4. Нарезка: фраза перед пустой строкой получает флаг абзаца.
        var twoParagraphs = PhraseSplitter()
        let paragraphPhrases = twoParagraphs.feedPhrases(
            "Колобок покатился.\n\nЛиса идёт. Хвост."
        ) + twoParagraphs.flushPhrases()
        check("нарезка: абзац — у фразы перед пустой строкой",
              paragraphPhrases.map(\.text) == ["Колобок покатился.", "Лиса идёт.", "Хвост."]
                && paragraphPhrases.map(\.paragraph) == [true, false, false],
              "\(paragraphPhrases)")
        var oneLine = PhraseSplitter()
        let linePhrases = oneLine.feedPhrases("Раз.\nДва.") + oneLine.flushPhrases()
        check("нарезка: один перевод строки — не абзац",
              linePhrases.map(\.text) == ["Раз.", "Два."]
                && linePhrases.allSatisfy { !$0.paragraph },
              "\(linePhrases)")

        // Абзац может прийти разрезанным между двумя SSE-кусками: первый
        // кончается одним переводом строки, второй начинается со второго.
        var splitParagraph = PhraseSplitter()
        let firstChunk = splitParagraph.feedPhrases("Колобок покатился.\n")
        check("фраза на границе куска придержана и учтена в позиции картинки",
              firstChunk.isEmpty && splitParagraph.pendingChars == 18,
              "фраз: \(firstChunk.count), хвост \(splitParagraph.pendingChars)")
        let splitPhrases = firstChunk
            + splitParagraph.feedPhrases("\nЛиса идёт.")
            + splitParagraph.flushPhrases()
        check("абзац, разрезанный между кусками SSE, не потерян",
              splitPhrases.map(\.text) == ["Колобок покатился.", "Лиса идёт."]
                && splitPhrases.map(\.paragraph) == [true, false],
              "\(splitPhrases)")
        check("после нарезки хвост пуст", splitParagraph.pendingChars == 0)

        var splitLine = PhraseSplitter()
        _ = splitLine.feedPhrases("Раз.\n")
        let splitLinePhrases = splitLine.feedPhrases("Два.") + splitLine.flushPhrases()
        check("один перевод строки между кусками — не абзац",
              splitLinePhrases.map(\.text) == ["Раз.", "Два."]
                && splitLinePhrases.allSatisfy { !$0.paragraph },
              "\(splitLinePhrases)")

        // 5. Какие ответы считаются сказкой: длинные, с картинками, plugin.
        check("сказка: длинный ответ (400+ символов)",
              AnswerController.isTaleAnswer(images: 0, chars: 400, plugin: false))
        check("обычный чат короче порога — не сказка",
              !AnswerController.isTaleAnswer(images: 0, chars: 399, plugin: false))
        check("сказка: картинки или finish_reason=plugin",
              AnswerController.isTaleAnswer(images: 1, chars: 0, plugin: false)
                && AnswerController.isTaleAnswer(images: 0, chars: 0, plugin: true))

        // 6. Разбор `done` плагина: признак сказки не теряется в SSE.
        let pluginDone = ChatSession.parse(
            line: #"data: {"type":"done","side":0,"finish_reason":"plugin","source":"plugin","silent":true}"#
        )
        check("done плагина: silent, сказка и метрики",
              pluginDone.first == .silent(true) && pluginDone.contains(.pluginAnswer),
              "\(pluginDone)")
        check("done модели: сказки нет",
              !ChatSession.parse(
                  line: #"data: {"type":"done","side":0,"finish_reason":"stop"}"#
              ).contains(.pluginAnswer))
        check("source=plugin без finish_reason — тоже сказка",
              ChatSession.parse(
                  line: #"data: {"type":"done","side":0,"source":"plugin"}"#
              ).contains(.pluginAnswer))

        // 7. Пауза живёт и отменяется мгновенно. Реальные 5 с не ждём:
        // отменённая задача просыпается сразу.
        let stopGate = PauseGate()
        var stopFired = 0
        stopGate.schedule(seconds: 5) { stopFired += 1 }
        check("пауза идёт — очередь не опустела", stopGate.isWaiting)
        stopGate.cancel()
        check("стоп гасит паузу немедленно", !stopGate.isWaiting)
        try? await Task.sleep(nanoseconds: 50_000_000)
        check("после стопа пауза не срабатывает", stopFired == 0,
              "срабатываний: \(stopFired)")

        let shortGate = PauseGate()
        var shortFired = 0
        shortGate.schedule(seconds: 0.05) { shortFired += 1 }
        check("короткая пауза ждёт своего срока", shortGate.isWaiting)
        try? await Task.sleep(nanoseconds: 200_000_000)
        check("после паузы очередь освобождается (дренаж — только теперь)",
              shortFired == 1 && !shortGate.isWaiting,
              "срабатываний: \(shortFired)")

        // 8. Выключенная озвучка и silent: фраза не уходит в /api/tts, значит
        // и пауз нет. Пустая очередь паузу не запускает.
        check("выключенная озвучка — фраз нет, пауз нет",
              !AnswerController.shouldSpeak(voiceOn: false, silent: false))
        check("silent (запись afplay) — фраз нет, пауз нет",
              !AnswerController.shouldSpeak(voiceOn: true, silent: true))
        check("обычный ответ — фразы уходят, паузы будут",
              AnswerController.shouldSpeak(voiceOn: true, silent: false))
        let idleVoice = VoicePlayer()
        check("пустая очередь не на паузе и сразу дренирована",
              !idleVoice.isPausing && idleVoice.isDrained)

        print(failures == 0 ? "все проверки пауз прошли" : "провалено: \(failures)")
        return failures == 0 ? 0 : 1
    }

    // MARK: - Компактный круг (задача 0188)

    /// Проверяет компактный режим без мышки: геометрию (правый нижний угол,
    /// зажатие в `visibleFrame` в т.ч. после смены монитора), живое окно
    /// (CGWindowList: размер, уровень floating, прозрачность), отсутствие
    /// перехвата фокуса, клики/меню и запоминание позиции в UserDefaults.
    ///
    /// Окна создаются в отдельном домене UserDefaults
    /// (`com.eva32.eva.selftest.compact`), чтобы не трогать позицию настоящего
    /// круга; домен удаляется в конце. Настоящие жесты мыши проверяются
    /// человеком, но те же обработчики кликов вызываются здесь напрямую.
    ///
    /// Функция намеренно синхронная: AppKit-транзакции WindowServer требуют
    /// витка главного run loop, а в процессе с `dispatchMain()` (см.
    /// `runSelfTest`) `await` может вернуть продолжение не на главном потоке.
    @MainActor
    private static func runCompact() -> Int32 {
        var failures = 0
        func check(_ name: String, _ ok: Bool, _ detail: String = "") {
            print((ok ? "OK   " : "FAIL ") + name + (detail.isEmpty ? "" : " — " + detail))
            // Сброс буфера: иначе падение ниже по коду съело бы напечатанное
            // (stdout — файл/пайп).
            fflush(stdout)
            if !ok { failures += 1 }
        }

        let size = CompactLayout.size
        check("размер круга в пределах 120–160 pt",
              size >= 120 && size <= 160, "\(size) pt")
        check("«Запускаться свёрнутым» по умолчанию выключено",
              !Settings.defaultStartCompact)

        let visible = NSScreen.main?.visibleFrame
            ?? NSScreen.screens.first?.visibleFrame
            ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        print("видимая область: \(CompactLayout.description(visible))")

        // 1. По умолчанию — правый нижний угол с отступом 24 pt.
        let defaultOrigin = CompactLayout.defaultOrigin(in: visible)
        check("позиция по умолчанию — правый нижний угол, отступ 24 pt",
              defaultOrigin.x == visible.maxX - size - CompactLayout.inset
                && defaultOrigin.y == visible.minY + CompactLayout.inset,
              "\(defaultOrigin.x),\(defaultOrigin.y)")

        // 2. Улетевшее за край окно зажимается целиком.
        let clamped = CompactLayout.clampedOrigin(
            NSPoint(x: -500, y: -500),
            size: size,
            in: visible
        )
        check("окно за левым/нижним краем зажато в visibleFrame",
              clamped == NSPoint(x: visible.minX, y: visible.minY),
              "\(clamped.x),\(clamped.y)")

        let pushed = CompactLayout.clampedOrigin(
            NSPoint(x: visible.maxX + 500, y: visible.maxY + 500),
            size: size,
            in: visible
        )
        check("окно за правым/верхним краем зажато (строка меню не перекрыта)",
              pushed.x == visible.maxX - size && pushed.y == visible.maxY - size,
              "\(pushed.x),\(pushed.y)")

        // 3. Смена монитора: позиция принадлежит второму экрану — остаётся
        // внутри него, а не прыгает на основной.
        let second = NSRect(
            x: visible.maxX,
            y: visible.minY,
            width: 1280,
            height: 800
        )
        let onSecond = CompactLayout.clampedOrigin(
            NSPoint(x: second.midX, y: second.midY),
            size: size,
            visibleFrames: [visible, second],
            fallback: visible
        )
        check("окно на втором экране остаётся в его visibleFrame",
              second.contains(NSRect(
                  x: onSecond.x, y: onSecond.y,
                  width: size, height: size
              )),
              "\(onSecond.x),\(onSecond.y)")

        // 4. Второй экран отключили — окно возвращается на основной.
        let detached = CompactLayout.clampedOrigin(
            NSPoint(x: second.midX, y: second.midY),
            size: size,
            visibleFrames: [visible],
            fallback: visible
        )
        check("после отключения монитора окно зажато в основной экран",
              visible.contains(NSRect(
                  x: detached.x, y: detached.y,
                  width: size, height: size
              )),
              "\(detached.x),\(detached.y)")

        // 5. Живое окно. Отдельный домен UserDefaults — настоящая позиция
        // компактного круга не меняется.
        let frontBefore = NSWorkspace.shared.frontmostApplication?.processIdentifier
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        // Без finishLaunching у процесса нет связи с WindowServer: окно
        // считается видимым, но CGWindowList его не видит, а первая же смена
        // frame падает в AppKit. `application.run()` в самопроверке не нужен.
        if !app.isRunning {
            app.finishLaunching()
        }
        // 4b. Один источник состояния: два круга (большой и маленький) видят
        // одно и то же без второй копии логики. Окна не нужны — проверяем
        // сам хаб и зарегистрированные в нём CircleView.
        let sharedHub = CircleHub()
        sharedHub.setAnimationEnabled(false)
        let bigCircle = CircleView(frame: NSRect(x: 0, y: 0, width: 162, height: 162))
        let smallCircle = CircleView(frame: NSRect(x: 0, y: 0, width: 140, height: 140))
        sharedHub.register(bigCircle)
        sharedHub.register(smallCircle)
        sharedHub.setState(.thinking)
        sharedHub.setWakeWaiting(true)
        sharedHub.setFollowUp(true)
        check("оба круга получают одно состояние из хаба",
              bigCircle.visualState == .thinking && smallCircle.visualState == .thinking
                && bigCircle.wakeWaiting && smallCircle.wakeWaiting
                && bigCircle.followUpListening && smallCircle.followUpListening,
              "большой=\(bigCircle.visualState.rawValue), "
                + "маленький=\(smallCircle.visualState.rawValue)")
        sharedHub.showError("сеть: проверка")
        check("ошибка из хаба видна на обоих кругах",
              bigCircle.visualState == .error && smallCircle.visualState == .error)
        sharedHub.setState(.listening)
        check("выход из ошибки — на обоих кругах",
              bigCircle.visualState == .listening && smallCircle.visualState == .listening)

        let suite = "com.eva32.eva.selftest.compact"
        guard let testDefaults = UserDefaults(suiteName: suite) else {
            print("FAIL не создался домен UserDefaults для проверки")
            return 1
        }
        testDefaults.removePersistentDomain(forName: suite)

        let hub = CircleHub()
        let controller = CompactWindowController(circles: hub, defaults: testDefaults)
        controller.show()
        pumpRunLoop(0.5)

        let frame = controller.frame ?? .zero
        print("живое окно: \(CompactLayout.description(frame)), "
            + "level=\(controller.windowLevel?.rawValue ?? -1), "
            + "visible=\(controller.isVisible)")
        check("компактное окно создано и показано", controller.isVisible)
        check("размер окна \(Int(size))×\(Int(size)) pt",
              frame.width == size && frame.height == size,
              "\(frame.width)×\(frame.height)")
        check("окно без рамки и nonactivatingPanel",
              controller.styleMask?.contains(.borderless) == true
                && controller.styleMask?.contains(.nonactivatingPanel) == true,
              "\(controller.styleMask?.rawValue ?? 0)")
        check("уровень floating",
              controller.windowLevel == .floating,
              "\(controller.windowLevel?.rawValue ?? -1)")
        check("фон прозрачный",
              controller.isOpaque == false
                && (controller.backgroundColor?.alphaComponent ?? 1) == 0)

        // 6. CGWindowList: окно действительно есть на экране, на уровне
        // floating. Имена окон без разрешения Screen Recording не читаются,
        // поэтому ищем по pid, слою и размеру.
        let pid = ProcessInfo.processInfo.processIdentifier
        let allWindows = CGWindowListCopyWindowInfo(
            [.optionOnScreenOnly],
            kCGNullWindowID
        ) as? [[String: Any]] ?? []
        let ours = allWindows.filter { info in
            guard let owner = info[kCGWindowOwnerPID as String] as? Int else { return false }
            return Int32(owner) == pid
        }
        let compactWindows = ours.filter { info in
            guard let layer = info[kCGWindowLayer as String] as? Int,
                  let raw = info[kCGWindowBounds as String] as? NSDictionary,
                  let bounds = CGRect(dictionaryRepresentation: raw as CFDictionary)
            else { return false }
            return layer == NSWindow.Level.floating.rawValue
                && abs(bounds.width - size) < 1
                && abs(bounds.height - size) < 1
        }
        check("CGWindowList: компактное окно на экране (pid \(pid), слой floating)",
              compactWindows.count == 1,
              "окон нашего pid: \(ours.count), подходящих: \(compactWindows.count)")
        if let info = compactWindows.first,
           let raw = info[kCGWindowBounds as String] as? NSDictionary,
           let bounds = CGRect(dictionaryRepresentation: raw as CFDictionary) {
            // CGWindowList отдаёт глобальные координаты с началом в левом
            // верхнем углу основного экрана (ось Y вниз), AppKit — в левом
            // нижнем (Y вверх). Переводим, чтобы сравнивать с visibleFrame
            // честно, а не «похоже на правду».
            let primaryHeight = NSScreen.screens.first?.frame.maxY ?? 0
            let appKitFrame = NSRect(
                x: bounds.minX,
                y: primaryHeight - bounds.maxY,
                width: bounds.width,
                height: bounds.height
            )
            print("CGWindowList: \(CompactLayout.description(appKitFrame)) (в координатах AppKit)")
            let frames = NSScreen.screens.map(\.visibleFrame)
            check("frame окна не выходит за visibleFrame",
                  frames.contains { $0.contains(appKitFrame) },
                  "frame=\(CompactLayout.description(appKitFrame))")
            check("окно не перекрывает строку меню",
                  frames.contains { appKitFrame.maxY <= $0.maxY },
                  "maxY=\(appKitFrame.maxY)")
            check("CGWindowList и NSWindow согласны про frame",
                  appKitFrame == frame,
                  "cg=\(CompactLayout.description(appKitFrame)), ns=\(CompactLayout.description(frame))")
        }

        let frontAfter = NSWorkspace.shared.frontmostApplication?.processIdentifier
        check("показ круга не забирает фокус",
              frontBefore == frontAfter && !app.isActive,
              "до=\(frontBefore ?? -1), после=\(frontAfter ?? -1), active=\(app.isActive)")
        check("приложение без иконки в Dock (accessory)",
              app.activationPolicy() == .accessory)

        // 6b. Клики и контекстное меню: та же разводка, что и у настоящей мыши.
        var expansions = 0
        var toggles = 0
        controller.onExpand = { expansions += 1 }
        controller.onToggleListening = { toggles += 1 }

        let menuTitles = controller.makeMenu().items
            .filter { !$0.isSeparatorItem }
            .map(\.title)
        check("правое меню: «Развернуть», «Настройки…», «Выход»",
              menuTitles == ["Развернуть", "Настройки…", "Выход"],
              "\(menuTitles)")

        // Двойной клик: первый клик отложил разворот, второй его отменил.
        controller.simulateClickForTest(clickCount: 1)
        controller.simulateClickForTest(clickCount: 2)
        pumpRunLoop(NSEvent.doubleClickInterval + 0.2)
        check("двойной клик — слушать, отложенный разворот отменён",
              toggles == 1 && expansions == 0,
              "слушать=\(toggles), развернуть=\(expansions)")

        // Одиночный клик разворачивает (с задержкой doubleClickInterval).
        controller.simulateClickForTest(clickCount: 1)
        pumpRunLoop(NSEvent.doubleClickInterval + 0.2)
        check("одиночный клик разворачивает",
              expansions == 1 && toggles == 1,
              "развернуть=\(expansions), слушать=\(toggles)")

        // 7. Позиция запоминается и восстанавливается.
        let moved = NSPoint(x: visible.minX + 80, y: visible.minY + 120)
        controller.setFrameOriginForTest(moved)
        pumpRunLoop(0.15)
        let savedX = testDefaults.object(forKey: Settings.compactXKey) as? Double
        let savedY = testDefaults.object(forKey: Settings.compactYKey) as? Double
        check("позиция перетаскивания записана в UserDefaults",
              savedX == Double(moved.x) && savedY == Double(moved.y),
              "\(savedX ?? .nan),\(savedY ?? .nan)")

        let restored = CompactWindowController(circles: hub, defaults: testDefaults)
        restored.show()
        pumpRunLoop(0.3)
        check("позиция восстановлена при новом показе",
              restored.frame?.origin == moved,
              "\(restored.frame.map(CompactLayout.description) ?? "?")")
        restored.hide()

        // 8. Сохранённая позиция с исчезнувшего монитора зажимается.
        testDefaults.set(9000.0, forKey: Settings.compactXKey)
        testDefaults.set(9000.0, forKey: Settings.compactYKey)
        let offscreen = CompactWindowController(circles: hub, defaults: testDefaults)
        offscreen.show()
        pumpRunLoop(0.3)
        let offscreenFrame = offscreen.frame ?? .zero
        check("сохранённая позиция вне экрана зажата в visibleFrame",
              NSScreen.screens.contains { $0.visibleFrame.contains(offscreenFrame) },
              CompactLayout.description(offscreenFrame))
        offscreen.hide()
        pumpRunLoop(0.2)
        check("после скрытия окно не на экране", !offscreen.isVisible)

        controller.hide()
        testDefaults.removePersistentDomain(forName: suite)
        print(failures == 0 ? "все проверки компактного круга прошли" : "провалено: \(failures)")
        return failures == 0 ? 0 : 1
    }

    // MARK: - Значения

    /// Прокрутить главный run loop: WindowServer-транзакции (показ окна, смена
    /// frame) выполняются только на его витке, а `Task.sleep` в CLI-процессе
    /// без `NSApp.run()` витков не даёт. Для самопроверки это допустимо:
    /// других задач на главном акторе в этот момент нет.
    @MainActor
    private static func pumpRunLoop(_ seconds: TimeInterval) {
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline {
            RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.02))
        }
    }

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
