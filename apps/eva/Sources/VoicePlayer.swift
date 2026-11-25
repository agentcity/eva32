import AVFoundation
import Foundation

/// Очередь озвучки: фразы уходят в `POST /api/tts` по одной, готовые WAV
/// играются подряд через `AVAudioPlayer`. Синтез идёт в фоне, звук — на
/// главном потоке; всё состояние меняется с главного актора.
/// Ошибка синтеза: текст для человека, сервер уже прислал его в `error.message`.
struct VoiceError: Error, Sendable, Equatable {
    let message: String
    init(_ message: String) { self.message = message }
}

@MainActor
final class VoicePlayer {
    private var server: URL?
    private var voice = VoicePresets.defaultID
    private var texts: [String] = []
    private var pendingWav: [Data] = []
    private var synthesis: Task<Void, Never>?
    private var player: AVAudioPlayer?
    private var meterTimer: Timer?
    private var generation = 0
    private var warnedUnavailable = false

    /// Куда сообщать, почему голос молчит (сервер не отвечает, `/api/tts` 503).
    var onIssue: ((String) -> Void)?
    var onSpeakingChanged: ((Bool) -> Void)?
    /// Громкость текущей фразы 0…1 для большого круга (30 Гц, metering плеера).
    var onLevel: ((Float) -> Void)?

    func configure(server: URL?, voice id: String) {
        self.server = server
        self.voice = id
    }

    /// Новый вопрос: глушим всё и начинаем с чистого листа.
    func reset() {
        stop()
    }

    func enqueue(_ phrases: [String]) {
        guard !phrases.isEmpty else { return }
        texts.append(contentsOf: phrases)
        pump()
    }

    func stop() {
        generation += 1
        synthesis?.cancel()
        synthesis = nil
        texts.removeAll()
        pendingWav.removeAll()
        stopMetering()
        player?.stop()
        player = nil
        onSpeakingChanged?(false)
    }

    // MARK: - Синтез

    private func pump() {
        guard synthesis == nil else { return }
        guard !texts.isEmpty else { return }
        guard let base = server else {
            texts.removeAll()
            report("не задан адрес сервера")
            return
        }
        let text = texts.removeFirst()
        let preset = voice
        log("озвучка: синтез \(text.count) симв., голос \(preset)")
        let generation = self.generation
        synthesis = Task { [weak self] in
            let result = await VoicePlayer.fetchWav(text: text, base: base, voice: preset)
            guard let self, self.generation == generation else { return }
            self.synthesis = nil
            switch result {
            case .success(let wav):
                self.pendingWav.append(wav)
                self.playNextIfIdle()
            case .failure(let error):
                self.report(error.message)
            }
            self.pump()
        }
    }

    /// POST `/api/tts` → WAV. Вынесено в nonisolated: сюда приходят только
    /// `Data`/`String`, главный актор не блокируется.
    nonisolated static func fetchWav(
        text: String,
        base: URL,
        voice: String
    ) async -> Result<Data, VoiceError> {
        var request = URLRequest(url: base.appendingPathComponent("api/tts"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 120
        let payload: [String: Any] = ["text": text, "voice": voice]
        guard let body = try? JSONSerialization.data(withJSONObject: payload) else {
            return .failure(VoiceError("не удалось собрать запрос озвучки"))
        }
        request.httpBody = body
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return .failure(VoiceError("сервер вернул не HTTP-ответ"))
            }
            guard http.statusCode == 200 else {
                var message = "HTTP \(http.statusCode)"
                if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let error = object["error"] as? [String: Any],
                   let text = error["message"] as? String,
                   !text.isEmpty
                {
                    message = text
                }
                return .failure(VoiceError(message))
            }
            guard data.count > 44, data.prefix(4) == Data("RIFF".utf8) else {
                return .failure(VoiceError("сервер вернул не WAV"))
            }
            return .success(data)
        } catch {
            return .failure(VoiceError("сеть: \(error.localizedDescription)"))
        }
    }

    // MARK: - Воспроизведение

    private func playNextIfIdle() {
        guard player == nil, !pendingWav.isEmpty else { return }
        let wav = pendingWav.removeFirst()
        do {
            let audio = try AVAudioPlayer(data: wav)
            audio.isMeteringEnabled = true
            audio.prepareToPlay()
            player = audio
            let started = audio.play()
            log("озвучка: играю \(wav.count) байт, старт=\(started)")
            onSpeakingChanged?(true)
            startMetering()
            watchPlayback(generation: generation)
        } catch {
            player = nil
            report("не удалось проиграть WAV: \(error.localizedDescription)")
            playNextIfIdle()
        }
    }

    // MARK: - Уровень для круга

    /// `AVAudioPlayer` умеет metering сам: раз в тик обновляем замеры и отдаём
    /// нормированную громкость. 30 Гц — столько же, сколько рисует круг.
    private func startMetering() {
        meterTimer?.invalidate()
        let timer = Timer(
            timeInterval: 1.0 / 30.0,
            target: self,
            selector: #selector(meterTick),
            userInfo: nil,
            repeats: true
        )
        timer.tolerance = 1.0 / 150.0
        RunLoop.main.add(timer, forMode: .common)
        meterTimer = timer
    }

    private func stopMetering() {
        meterTimer?.invalidate()
        meterTimer = nil
        onLevel?(0)
    }

    @objc private func meterTick() {
        guard let player, player.isPlaying else {
            stopMetering()
            return
        }
        player.updateMeters()
        let db = player.averagePower(forChannel: 0)
        onLevel?(min(max((db + 50) / 50, 0), 1))
    }

    /// Опрос `isPlaying` вместо делегата: так `AVAudioPlayer` не пересекает
    /// границу акторов, а задержка 120 мс на слух незаметна.
    private func watchPlayback(generation: Int) {
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: 120_000_000)
            guard let self, self.generation == generation else { return }
            if let audio = self.player, audio.isPlaying {
                self.watchPlayback(generation: generation)
                return
            }
            if self.player != nil {
                self.stopMetering()
                self.player = nil
                self.onSpeakingChanged?(false)
                self.playNextIfIdle()
            }
        }
    }

    private func report(_ message: String) {
        guard !warnedUnavailable else { return }
        warnedUnavailable = true
        log("озвучка: \(message)")
        onIssue?("голос недоступен: \(message)")
    }
}
