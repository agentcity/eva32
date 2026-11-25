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

    static let defaultServer = "http://127.0.0.1:8765"
    static let defaultMaxTokens = 256

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
