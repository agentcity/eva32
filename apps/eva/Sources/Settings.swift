import Foundation

/// Модель для запроса: бэкенд сервера и идентификатор модели из реестра.
struct ModelSpec: Equatable, Sendable {
    var backend: String
    var model: String

    /// По умолчанию — та же модель, что в подтверждении задачи 0184.
    static let fallback = ModelSpec(backend: "vtm", model: "runs/q4/3b-q4k.vtm")

    var storageValue: String { backend + "|" + model }

    var displayName: String {
        model.isEmpty ? backend : "\(backend) · \(model)"
    }

    static func parse(_ raw: String) -> ModelSpec? {
        let parts = raw.split(separator: "|", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else { return nil }
        return ModelSpec(backend: String(parts[0]), model: String(parts[1]))
    }
}

/// Пресет голоса, который понимает `/api/tts` (`VOICE_PRESETS` в tools/chat_server.py).
struct VoicePreset: Equatable, Sendable {
    let id: String
    let title: String
}

enum VoicePresets {
    static let all: [VoicePreset] = [
        VoicePreset(id: "younger", title: "моложе и мягче (по умолчанию)"),
        VoicePreset(id: "young", title: "моложе"),
        VoicePreset(id: "young_soft", title: "моложе, спокойнее"),
        VoicePreset(id: "softest", title: "самый мягкий"),
        VoicePreset(id: "soft", title: "спокойнее"),
        VoicePreset(id: "irina", title: "Ирина, как есть"),
    ]
    static let defaultID = "younger"

    static func title(_ id: String) -> String {
        all.first { $0.id == id }?.title ?? id
    }
}

/// Настройки в `UserDefaults`: адрес сервера, модель, голос, озвучка.
///
/// Домен — идентификатор бандла (`com.eva32.eva`), поэтому `defaults write`
/// и приложение видят одни и те же значения. Чтение и запись — только с главного
/// потока: все поля меняются из UI.
@MainActor
final class Settings {
    static let shared = Settings()
    static let didChange = Notification.Name("EVA.settingsDidChange")

    static let serverKey = "eva.serverURL"
    static let modelKey = "eva.modelSpec"
    static let voiceKey = "eva.voice"
    static let speakKey = "eva.speakReplies"
    static let maxTokensKey = "eva.maxTokens"
    /// Галочка «Анимация»: без неё большой круг статичен (слабые машины).
    static let animateCircleKey = "eva.animateCircle"
    /// Фоновое ожидание «Привет, Ева»: постоянно включённый микрофон, поэтому
    /// по умолчанию выключено (батарея и приватность).
    static let wakeWordKey = "eva.wakeWord"
    /// «Слушать после ответа» (задача 0190): микрофон открывается сам, чтобы
    /// следующий вопрос можно было задать без «Привет, Ева».
    static let followUpKey = "eva.followUpListen"
    /// Сколько секунд тишины терпит follow-up, пока не сказано ни слова.
    static let followUpSilenceKey = "eva.followUpSilence"
    /// «Запускаться свёрнутым» (задача 0188): на экране сразу только круг.
    static let startCompactKey = "eva.startCompact"
    /// Позиция компактного круга (задача 0188) — левый нижний угол окна.
    /// Хранится не ради настроек в UI, а чтобы круг оставался там, где его
    /// оставили; при показе зажимается в `visibleFrame`.
    static let compactXKey = "eva.compactX"
    static let compactYKey = "eva.compactY"

    static let defaultServer = "http://127.0.0.1:8765"
    static let defaultMaxTokens = 256
    /// По умолчанию follow-up включён: разговор без кнопок — основной сценарий.
    static let defaultFollowUpListen = true
    /// Варианты паузы «до ухода в ожидание»; 12 с — компромисс между
    /// «не успел собраться с мыслью» и «микрофон висит впустую».
    static let followUpSilenceOptions = [8, 12, 20]
    static let defaultFollowUpSilence = 12
    /// «Запускаться свёрнутым» по умолчанию выключено: обычный запуск
    /// показывает главное окно (задача 0188).
    static let defaultStartCompact = false

    private let defaults: UserDefaults

    private init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    func registerDefaults() {
        defaults.register(defaults: [
            Settings.serverKey: Settings.defaultServer,
            Settings.modelKey: ModelSpec.fallback.storageValue,
            Settings.voiceKey: VoicePresets.defaultID,
            Settings.speakKey: true,
            Settings.maxTokensKey: Settings.defaultMaxTokens,
            Settings.animateCircleKey: true,
            Settings.wakeWordKey: false,
            Settings.followUpKey: Settings.defaultFollowUpListen,
            Settings.followUpSilenceKey: Settings.defaultFollowUpSilence,
            Settings.startCompactKey: Settings.defaultStartCompact,
        ])
    }

    var serverText: String {
        get { defaults.string(forKey: Settings.serverKey) ?? Settings.defaultServer }
        set { store(Settings.serverKey, newValue.trimmingCharacters(in: .whitespacesAndNewlines)) }
    }

    var model: ModelSpec {
        get { ModelSpec.parse(defaults.string(forKey: Settings.modelKey) ?? "") ?? .fallback }
        set { store(Settings.modelKey, newValue.storageValue) }
    }

    var voice: String {
        get { defaults.string(forKey: Settings.voiceKey) ?? VoicePresets.defaultID }
        set { store(Settings.voiceKey, newValue) }
    }

    var speakReplies: Bool {
        get { defaults.bool(forKey: Settings.speakKey) }
        set { store(Settings.speakKey, newValue) }
    }

    var maxTokens: Int {
        get {
            let value = defaults.integer(forKey: Settings.maxTokensKey)
            return value > 0 ? value : Settings.defaultMaxTokens
        }
        set { store(Settings.maxTokensKey, newValue) }
    }

    /// Анимация большого круга. Выключение экономит CPU на слабых машинах:
    /// круг остаётся, но не дышит, не вращается и не повторяет волну.
    var animateCircle: Bool {
        get { defaults.bool(forKey: Settings.animateCircleKey) }
        set { store(Settings.animateCircleKey, newValue) }
    }

    /// Ждать фразу «Привет, Ева» в `idle` (задача 0187). Выключено по
    /// умолчанию: это постоянный микрофон и расход батареи.
    var wakeWord: Bool {
        get { defaults.bool(forKey: Settings.wakeWordKey) }
        set { store(Settings.wakeWordKey, newValue) }
    }

    /// Слушать после ответа (задача 0190). Включено по умолчанию; работает
    /// только вместе с `wakeWord`: без ожидания фразы микрофон после ответа
    /// сам не открывается (прежнее поведение).
    var followUpListen: Bool {
        get { defaults.bool(forKey: Settings.followUpKey) }
        set { store(Settings.followUpKey, newValue) }
    }

    /// Пауза тишины в follow-up, секунды. Значение вне списка вариантов не
    /// ломает поведение: оно честно используется как есть.
    var followUpSilence: Int {
        get {
            let value = defaults.integer(forKey: Settings.followUpSilenceKey)
            return value > 0 ? value : Settings.defaultFollowUpSilence
        }
        set { store(Settings.followUpSilenceKey, max(newValue, 1)) }
    }

    /// Та же пауза в секундах — для таймеров `SpeechRecorder`.
    var followUpSilenceValue: Double { Double(followUpSilence) }

    /// «Запускаться свёрнутым» (задача 0188): при старте главное окно не
    /// показывается, остаётся только круг в правом нижнем углу. По умолчанию
    /// выключено — флаг `--background` и ручной запуск сохраняют прежнее
    /// поведение.
    var startCompact: Bool {
        get { defaults.bool(forKey: Settings.startCompactKey) }
        set { store(Settings.startCompactKey, newValue) }
    }

    /// Нормализованный базовый URL сервера без хвостового слэша.
    /// `127.0.0.1:8765` превращается в `http://127.0.0.1:8765`.
    var baseURL: URL? {
        Settings.normalize(serverText)
    }

    /// Приводит строку к базовому URL; `nil` — строка не похожа на адрес.
    static func normalize(_ raw: String?) -> URL? {
        guard var text = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else {
            return nil
        }
        if !text.contains("://") {
            text = "http://" + text
        }
        guard var components = URLComponents(string: text),
              let scheme = components.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              let host = components.host, !host.isEmpty
        else {
            return nil
        }
        components.query = nil
        components.fragment = nil
        var path = components.path
        while path.hasSuffix("/") { path.removeLast() }
        components.path = path
        return components.url
    }

    private func store(_ key: String, _ value: Any) {
        defaults.set(value, forKey: key)
        NotificationCenter.default.post(name: Settings.didChange, object: nil)
    }
}

/// Снимок настроек для сетевых запросов: передаётся как значение, а не как
/// живой объект, чтобы запрос не менял адрес на середине.
struct ChatConfig: Sendable {
    var serverURL: URL
    var model: ModelSpec
    var maxTokens: Int
}
