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
