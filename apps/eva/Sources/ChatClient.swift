import Foundation

/// Событие SSE-потока `/api/chat`, уже разобранное и относящееся к колонке 0
/// (в приложении она одна; вторая — `{"backend": "none"}`).
enum ChatEvent: Sendable, Equatable {
    case started(modelLabel: String, cold: Bool?)
    case delta(String)
    /// Иллюстрация сказки (задача 0224): URL как его отдал сервер
    /// (`/media/fairy-tales/<slug>/NN.jpg`) и подпись. Приходит между
    /// `delta`, поэтому картинка показывается по ходу текста.
    case image(url: String, alt: String)
    /// Ответ уже озвучен записью на сервере (задача 0223, `afplay`): текст
    /// показываем как обычно, а в `/api/tts` его отправлять нельзя — иначе
    /// сказка звучит двумя голосами. Сервер ставит `silent: true` в `delta`
    /// и `done`; событие приходит раньше первого `delta`.
    case silent(Bool)
    /// Короткая строка метрик из `done` (время, ток/с) для статуса.
    case stats(String)
    case failed(String)
    /// Поток закрыт: и при нормальном конце, и при ошибке, и при отмене.
    case finished
}

/// POST `/api/chat` и разбор `data: {...}` построчно.
///
/// Формат сервера (tools/chat_server.py): события `start`, `delta`, `done`,
/// `end`, `error`, терминальное `data: [DONE]`. Запрос — ровно тот, что
/// подтверждён в задаче 0184: question, две стороны (вторая «нет»),
/// max_tokens, use_system.
@MainActor
final class ChatSession {
    private var task: Task<Void, Never>?
    private var urlSession: URLSession?

    func cancel() {
        task?.cancel()
        task = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
    }

    func send(
        question: String,
        config: ChatConfig,
        onEvent: @escaping @MainActor @Sendable (ChatEvent) -> Void
    ) {
        cancel()
        let endpoint = config.serverURL.appendingPathComponent("api/chat")
        let payload: [String: Any] = [
            "question": question,
            "sides": [
                ["backend": config.model.backend, "model": config.model.model],
                ["backend": "none"],
            ],
            "max_tokens": config.maxTokens,
            "use_system": true,
        ]
        guard let body = try? JSONSerialization.data(withJSONObject: payload) else {
            onEvent(.failed("не удалось собрать тело запроса"))
            onEvent(.finished)
            return
        }
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.httpBody = body
        request.timeoutInterval = 300

        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 300
        configuration.timeoutIntervalForResource = 600
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        let session = URLSession(configuration: configuration)
        urlSession = session

        log("чат: POST \(endpoint.absoluteString) \(String(decoding: body, as: UTF8.self))")

        let deliver: @Sendable (ChatEvent) -> Void = { event in
            Task { @MainActor in onEvent(event) }
        }
        task = Task {
            await ChatSession.consume(session: session, request: request, onEvent: deliver)
        }
    }

    // MARK: - Поток (вне главного актора: цикл целиком в фоне)

    private nonisolated static func consume(
        session: URLSession,
        request: URLRequest,
        onEvent: @Sendable (ChatEvent) -> Void
    ) async {
        defer { onEvent(.finished) }
        do {
            let (bytes, response) = try await session.bytes(for: request)
            guard let http = response as? HTTPURLResponse else {
                onEvent(.failed("сервер вернул не HTTP-ответ"))
                return
            }
            guard http.statusCode == 200 else {
                var raw = ""
                for try await line in bytes.lines {
                    raw += line + "\n"
                    if raw.count > 4000 { break }
                }
                onEvent(.failed(describeHTTP(status: http.statusCode, body: raw)))
                return
            }
            lines: for try await line in bytes.lines {
                try Task.checkCancellation()
                for event in parse(line: line) {
                    if event == .finished { break lines }  // это может дать только [DONE]
                    onEvent(event)
                }
            }
        } catch is CancellationError {
            // Отмена «Стоп» — не ошибка.
        } catch {
            let nsError = error as NSError
            if nsError.domain == NSURLErrorDomain, nsError.code == NSURLErrorCancelled {
                return
            }
            onEvent(.failed("сеть: \(error.localizedDescription)"))
        }
    }

    /// Разбор одной строки SSE: `data: {...}`, пустые строки и `[DONE]`.
    ///
    /// Возвращает массив, потому что одно событие сервера может дать два:
    /// `silent` (признак «уже озвучено записью», задача 0223) должен попасть
    /// в `AnswerController` раньше текста, иначе озвучка успеет начаться.
    /// Функция не `private`, чтобы её проверял `--selftest-images`.
    nonisolated static func parse(line: String) -> [ChatEvent] {
        guard line.hasPrefix("data:") else { return [] }
        var payload = line.dropFirst("data:".count)
        if payload.hasPrefix(" ") { payload = payload.dropFirst() }
        if payload == "[DONE]" { return [.finished] }
        guard let data = payload.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = object["type"] as? String
        else {
            return []
        }
        // Колонка 1 в приложении выключена («нет»), её события не показываем.
        if let side = object["side"] as? Int, side != 0, type != "end" {
            return []
        }
        switch type {
        case "start":
            let label = (object["model_label"] as? String)
                ?? (object["model"] as? String)
                ?? "модель"
            return [.started(modelLabel: label, cold: object["cold"] as? Bool)]
        case "delta":
            var events = silentEvent(in: object)
            guard let text = object["text"] as? String, !text.isEmpty else { return events }
            events.append(.delta(text))
            return events
        case "image":
            // Битая картинка не должна ломать ответ: нет URL — просто нет
            // события, текст сказки продолжает идти.
            guard let url = object["url"] as? String,
                  !url.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                return []
            }
            let alt = (object["alt"] as? String) ?? ""
            return [.image(url: url, alt: alt)]
        case "done":
            var events = silentEvent(in: object)
            var parts: [String] = []
            if let milliseconds = object["elapsed_ms"] as? Int, milliseconds > 0 {
                parts.append(String(format: "%.1f с", Double(milliseconds) / 1000.0))
            }
            if let speed = object["tokens_per_second"] as? Double, speed > 0 {
                parts.append(String(format: "%.1f ток/с", speed))
            }
            if let tokens = object["completion_tokens"] as? Int, tokens > 0 {
                parts.append("\(tokens) токенов")
            }
            if object["estimated"] as? Bool == true {
                parts.append("токены — оценка")
            }
            events.append(.stats(parts.joined(separator: " · ")))
            return events
        case "error":
            let message = (object["message"] as? String) ?? "модель вернула ошибку"
            return [.failed(message)]
        default:
            return []
        }
    }

    /// `"silent": true` в `delta`/`done` — ответ уже озвучен записью на
    /// сервере; отдаём отдельным событием, чтобы признак нельзя было потерять.
    private nonisolated static func silentEvent(in object: [String: Any]) -> [ChatEvent] {
        guard object["silent"] as? Bool == true else { return [] }
        return [.silent(true)]
    }

    /// Человеческий текст ошибки HTTP: у сервера `{"error": {"message": ...}}`.
    private nonisolated static func describeHTTP(status: Int, body: String) -> String {
        if let data = body.data(using: .utf8),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let error = object["error"] as? [String: Any],
           let message = error["message"] as? String,
           !message.isEmpty
        {
            return "HTTP \(status): \(message)"
        }
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "HTTP \(status) без тела ответа" }
        return "HTTP \(status): \(String(trimmed.prefix(300)))"
    }

    /// `POST /api/cancel {"side": 0}` — остановить идущую генерацию на сервере
    /// (задача 0185), а не только закрыть SSE. Кнопка «Стоп» в приложении
    /// шлёт это явно: обрыв потока тоже отменяет, но флаг в C-рантайме
    /// ставится сразу, поэтому модель не договаривает в пустоту.
    ///
    /// Возвращает короткую строку для журнала; сетевые ошибки глотаются —
    /// «Стоп» уже сработал локально.
    /// Остановить внешнее воспроизведение на Mac (запись сказки, музыка,
    /// радио). Эти звуки играет сервер через afplay/Music.app — «Стоп, Ева»
    /// их не остановит, если не позвать `/api/audio/stop` (0225).
    nonisolated static func stopRemoteAudio(base: URL) async -> String {
        var request = URLRequest(url: base.appendingPathComponent("api/audio/stop"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("{}".utf8)
        request.timeoutInterval = 3
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard code == 200 else { return "сервер ответил HTTP \(code)" }
            if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let stopped = object["stopped"] as? [String], !stopped.isEmpty
            {
                return "остановлено: \(stopped.joined(separator: ", "))"
            }
            return "нечего останавливать"
        } catch {
            return "сервер не ответил (\(error.localizedDescription))"
        }
    }

    nonisolated static func cancelRemote(base: URL, side: Int = 0) async -> String {
        var request = URLRequest(url: base.appendingPathComponent("api/cancel"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: ["side": side])
        request.timeoutInterval = 3
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard code == 200 else { return "сервер ответил HTTP \(code)" }
            if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let count = object["cancelled"] as? Int
            {
                return "сервер отменил \(count)"
            }
            return "сервер принял отмену"
        } catch {
            return "сервер не ответил (\(error.localizedDescription))"
        }
    }
}
