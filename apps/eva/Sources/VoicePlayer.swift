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
    /// Синтезированная фраза вместе с её текстом: текст нужен стоп-слушателю,
    /// чтобы отличить команду от эха собственной реплики (задача 0194).
    private struct PendingVoice {
        let wav: Data
        let text: String
    }

    private var server: URL?
    private var voice = SileroVoices.defaultID
    private var texts: [String] = []
    private var pendingWav: [PendingVoice] = []
    private var synthesis: Task<Void, Never>?
    private var player: AVAudioPlayer?
    private var meterTimer: Timer?
    private var generation = 0
    private var warnedUnavailable = false

    /// Оценка темпа речи baya: ~12,8 символов в секунду (живой замер). По ней считается
    /// прогресс внутри текущей фразы (у `AVAudioPlayer` нет счётчика
    /// произнесённых символов, а длина фразы и время известны).
    nonisolated static let charsPerSecond = 15.0

    /// Сколько символов уже сказано. Складывается из завершённых фраз и
    /// дробной оценки текущей — по этому числу `AnswerController` показывает
    /// очередную картинку сказки (задача 0227).
    private var spokenChars = 0
    /// Длина фразы, которая играет сейчас, и момент её старта.
    private var phraseChars = 0
    private var phraseStartedAt: Date?
    /// Сколько символов текста ушло в озвучку за этот ответ (включая текущую
    /// и уже сказанные фразы). Нужно как позиция картинок в «речевых»
    /// символах: пробелы между фразами в озвучку не попадают.
    private(set) var enqueuedChars = 0

    /// Куда сообщать, почему голос молчит (сервер не отвечает, `/api/tts` 503).
    var onIssue: ((String) -> Void)?
    var onSpeakingChanged: ((Bool) -> Void)?
    /// Прогресс озвучки: (сказано символов, всего ушло в озвучку). Второй
    /// аргумент растёт по мере прихода текста, первый — только вперёд.
    var onProgress: ((Int, Int) -> Void)?
    /// Очередь опустела: всё сказанное доиграно и синтезировать больше нечего.
    /// Нужно follow-up (задача 0190): микрофон открывается только после того,
    /// как голос Евы смолк, иначе она услышит саму себя. Вызывается только при
    /// естественном конце — `stop()` («Стоп», новый вопрос) сюда не относится.
    var onDrained: (() -> Void)?
    /// Громкость текущей фразы 0…1 для большого круга (30 Гц, metering плеера).
    var onLevel: ((Float) -> Void)?

    /// Очередь пуста: ничего не играет, не ждёт и не синтезируется.
    var isDrained: Bool {
        streamDone && player == nil && pendingWav.isEmpty && texts.isEmpty
            && synthesis == nil
    }

    /// Текст фразы, которая звучит прямо сейчас; `nil` — тишина. Эхо-защита
    /// стоп-команды смотрит сюда: если Ева сама говорит «…и стоп.», совпадение
    /// с микрофона — это её же голос, а не команда (задача 0194).
    private(set) var currentText: String?

    /// Уже сообщили, что очередь опустела: не повторяем на каждый чих.
    private var drainedNotified = true

    /// Ответ ещё приходит. Очередь может опустеть между двумя дельтами —
    /// в этот момент «доиграли» ещё не значит «досказали»: длинная сказка
    /// приходит кусками, и ранний дренаж открывал микрофон посреди текста
    /// (жалоба: последняя фраза сказки не звучит).
    private var streamDone = true

    func configure(server: URL?, voice id: String) {
        self.server = server
        self.voice = id
    }

    /// Новый вопрос: глушим всё и начинаем с чистого листа.
    func reset() {
        stop()
    }

    /// Ответ начался: до его конца дренаж не объявляем.
    func beginStream() {
        streamDone = false
        drainedNotified = false
        // Новый ответ — новый шанс: прошлый отказ TTS не должен навсегда
        // лишать `onIssue` (по нему картинки переходят на таймер, 0227).
        warnedUnavailable = false
    }

    /// Стрим закончился: очередь доиграет и только тогда объявит дренаж.
    func endStream() {
        streamDone = true
        notifyDrainedIfIdle()
    }

    func enqueue(_ phrases: [String]) {
        guard !phrases.isEmpty else { return }
        texts.append(contentsOf: phrases)
        enqueuedChars += phrases.reduce(0) { $0 + $1.count }
        drainedNotified = false
        pump()
        notifyProgress()
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
        currentText = nil
        spokenChars = 0
        phraseChars = 0
        phraseStartedAt = nil
        enqueuedChars = 0
        drainedNotified = true
        onSpeakingChanged?(false)
        onProgress?(0, 0)
    }

    // MARK: - Синтез

    private func pump() {
        guard synthesis == nil else { return }
        guard !texts.isEmpty else { return }
        guard let base = server else {
            texts.removeAll()
            report("не задан адрес сервера")
            notifyDrainedIfIdle()
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
                self.pendingWav.append(PendingVoice(wav: wav, text: text))
                self.playNextIfIdle()
            case .failure(let error):
                self.report(error.message)
            }
            self.pump()
            self.notifyDrainedIfIdle()
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
        let pending = pendingWav.removeFirst()
        let wav = pending.wav
        do {
            let audio = try AVAudioPlayer(data: wav)
            audio.isMeteringEnabled = true
            audio.prepareToPlay()
            player = audio
            currentText = pending.text
            phraseChars = pending.text.count
            phraseStartedAt = Date()
            let started = audio.play()
            log("озвучка: играю \(wav.count) байт, старт=\(started)")
            onSpeakingChanged?(true)
            notifyProgress()
            startMetering()
            watchPlayback(generation: generation)
        } catch {
            player = nil
            currentText = nil
            phraseChars = 0
            phraseStartedAt = nil
            report("не удалось проиграть WAV: \(error.localizedDescription)")
            playNextIfIdle()
            notifyDrainedIfIdle()
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
        notifyProgress()
    }

    // MARK: - Прогресс речи для картинок

    /// Оценка сказанного внутри текущей фразы: по прошедшему времени и
    /// длине фразы, ~15 символов/с. Вынесено `nonisolated static`, чтобы
    /// правило проверялось в `--selftest-images` без звука.
    nonisolated static func spokenEstimate(
        elapsed: TimeInterval,
        chars: Int,
        rate: Double = VoicePlayer.charsPerSecond
    ) -> Int {
        guard chars > 0, elapsed > 0, rate > 0 else { return 0 }
        return min(chars, Int(elapsed * rate))
    }

    /// Сказано всего = завершённые фразы + оценка текущей. Вызывается 30 Гц
    /// во время игры и на границах фраз.
    private func notifyProgress() {
        let within = phraseStartedAt.map {
            VoicePlayer.spokenEstimate(
                elapsed: Date().timeIntervalSince($0),
                chars: phraseChars
            )
        } ?? 0
        onProgress?(spokenChars + within, enqueuedChars)
    }

    /// Фраза доиграла: её символы переходят в «сказано» целиком.
    private func completeCurrentPhrase() {
        if phraseChars > 0 {
            spokenChars += phraseChars
            phraseChars = 0
        }
        phraseStartedAt = nil
        notifyProgress()
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
                self.currentText = nil
                self.completeCurrentPhrase()
                self.onSpeakingChanged?(false)
                self.playNextIfIdle()
                self.notifyDrainedIfIdle()
            }
        }
    }

    /// Сообщить ровно один раз: озвучка доиграла и очередь пуста. Отсюда
    /// `AppDelegate` открывает follow-up микрофон.
    private func notifyDrainedIfIdle() {
        guard !drainedNotified, isDrained else { return }
        drainedNotified = true
        onDrained?()
    }

    private func report(_ message: String) {
        guard !warnedUnavailable else { return }
        warnedUnavailable = true
        log("озвучка: \(message)")
        onIssue?("голос недоступен: \(message)")
    }
}
