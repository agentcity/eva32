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

/// Голос Silero, который понимает `/api/tts` (`SILERO_VOICES` в
/// tools/chat_server.py). Задача 0197: озвучка — только Silero, пресетов
/// young/soft/irina больше нет; `aidar` валиден в API, но в UI не показывается.
struct SileroVoice: Equatable, Sendable {
    let id: String
    let title: String
}

enum SileroVoices {
    /// Ровно три голоса в том же порядке, что в селекторе страницы чата.
    static let all: [SileroVoice] = [
        SileroVoice(id: "baya", title: "baya — живой"),
        SileroVoice(id: "kseniya", title: "kseniya — выше"),
        SileroVoice(id: "xenia", title: "xenia — ниже"),
    ]
    static let defaultID = "baya"

    static func title(_ id: String) -> String {
        all.first { $0.id == id }?.title ?? id
    }

    static func isValid(_ id: String) -> Bool {
        all.contains { $0.id == id }
    }

    /// Годный голос или `baya` по умолчанию (пустое/чужое значение — тоже).
    static func normalize(_ raw: String?) -> String {
        guard let id = raw?.trimmingCharacters(in: .whitespacesAndNewlines),
              isValid(id)
        else { return defaultID }
        return id
    }
}

/// Скорость чтения сказок (задача 0230). «Медленная» растягивает паузы между
/// фразами и абзацами в `ReadingPause.taleFactor` раз; обычный чат настройка
/// не трогает — там паузы всегда короткие.
enum TalePace: String, CaseIterable, Sendable {
    case normal
    case slow

    var title: String {
        switch self {
        case .normal: return "обычная"
        case .slow: return "медленная"
        }
    }

    /// Разбор значения из `UserDefaults`: пусто/чужое — «медленная», детский
    /// режим по умолчанию.
    static func normalize(_ raw: String?) -> TalePace {
        guard let value = raw?.trimmingCharacters(in: .whitespacesAndNewlines),
              let pace = TalePace(rawValue: value)
        else { return .slow }
        return pace
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
    /// «Показывать картинки сказок» (задача 0224): иллюстрации из SSE-события
    /// `image` в окне ответа. По умолчанию включено; на озвучку и круг не
    /// влияет — только на показ.
    static let showTaleImagesKey = "eva.showTaleImages"
    /// «Скорость чтения сказок» (задача 0230): `normal`/`slow`. Читается в
    /// момент паузы, поэтому смена действует без перезапуска.
    static let talePaceKey = "eva.talePace"
    static let maxTokensKey = "eva.maxTokens"
    /// Галочка «Анимация»: без неё большой круг статичен (слабые машины).
    static let animateCircleKey = "eva.animateCircle"
    /// Фоновое ожидание «Привет, Ева»: постоянно включённый микрофон, поэтому
    /// по умолчанию выключено (батарея и приватность).
    static let wakeWordKey = "eva.wakeWord"
    /// «Слушать после ответа» (задача 0190): микрофон открывается сам, чтобы
    /// следующий вопрос можно было задать без «Привет, Ева».
    static let followUpKey = "eva.followUpListen"
    /// Голосовая команда «Стоп, Ева» (задача 0194): микрофон активен, только
    /// пока Ева думает или говорит. По умолчанию включена.
    static let stopCommandKey = "eva.stopCommand"
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
    /// Длинные ответы не должны обрываться на середине: 1024 — потолок,
    /// который модель в 1024 токена контекста ещё может отдать целиком.
    static let defaultMaxTokens = 1024
    /// По умолчанию follow-up включён: разговор без кнопок — основной сценарий.
    static let defaultFollowUpListen = true
    /// «Стоп, Ева» включена по умолчанию: перебить голосом — основной сценарий
    /// разговора, а микрофон поднимается только на время речи Евы.
    static let defaultStopCommand = true
    /// Варианты паузы «до ухода в ожидание»; 12 с — компромисс между
    /// «не успел собраться с мыслью» и «микрофон висит впустую».
    static let followUpSilenceOptions = [8, 12, 20]
    static let defaultFollowUpSilence = 12
    /// «Запускаться свёрнутым» по умолчанию выключено: обычный запуск
    /// показывает главное окно (задача 0188).
    static let defaultStartCompact = false
    /// Картинки сказок по умолчанию включены: сказка с иллюстрациями — то,
    /// ради чего плагин их готовит (задача 0224). Выключение оставляет текст.
    static let defaultShowTaleImages = true
    /// Сказки по умолчанию читаются медленно: ребёнок не успевает за обычным
    /// темпом ответа (задача 0230).
    static let defaultTalePace = TalePace.slow

    private let defaults: UserDefaults

    private init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    func registerDefaults() {
        defaults.register(defaults: [
            Settings.serverKey: Settings.defaultServer,
            Settings.modelKey: ModelSpec.fallback.storageValue,
            Settings.voiceKey: SileroVoices.defaultID,
            Settings.speakKey: true,
            Settings.showTaleImagesKey: Settings.defaultShowTaleImages,
            Settings.talePaceKey: Settings.defaultTalePace.rawValue,
            Settings.maxTokensKey: Settings.defaultMaxTokens,
            Settings.animateCircleKey: true,
            Settings.wakeWordKey: false,
            Settings.followUpKey: Settings.defaultFollowUpListen,
            Settings.stopCommandKey: Settings.defaultStopCommand,
            Settings.followUpSilenceKey: Settings.defaultFollowUpSilence,
            Settings.startCompactKey: Settings.defaultStartCompact,
        ])
        migrateVoice()
    }

    /// Миграция `eva.voice` (задача 0197): пресеты Piper `young`/`soft`/`irina`
    /// (и производные `younger`/`young_soft`/`softest`) заменяются на `baya`.
    /// Неизвестный id — тоже `baya`: сервер не знает такого голоса и ответил
    /// бы 400. Пишем один раз, без уведомления — UI ещё не построен.
    private func migrateVoice() {
        let raw = defaults.string(forKey: Settings.voiceKey)
        let voice = SileroVoices.normalize(raw)
        if raw != voice {
            defaults.set(voice, forKey: Settings.voiceKey)
        }
    }

    var serverText: String {
        get { defaults.string(forKey: Settings.serverKey) ?? Settings.defaultServer }
        set { store(Settings.serverKey, newValue.trimmingCharacters(in: .whitespacesAndNewlines)) }
    }

    var model: ModelSpec {
        get { ModelSpec.parse(defaults.string(forKey: Settings.modelKey) ?? "") ?? .fallback }
        set { store(Settings.modelKey, newValue.storageValue) }
    }

    /// Голос Silero: `baya`/`kseniya`/`xenia`. Старое значение в UserDefaults
    /// (пресет Piper) читается как `baya`, даже если миграция ещё не прошла.
    var voice: String {
        get { SileroVoices.normalize(defaults.string(forKey: Settings.voiceKey)) }
        set { store(Settings.voiceKey, SileroVoices.normalize(newValue)) }
    }

    var speakReplies: Bool {
        get { defaults.bool(forKey: Settings.speakKey) }
        set { store(Settings.speakKey, newValue) }
    }

    /// «Показывать картинки сказок» (задача 0224). При выключенной галочке
    /// SSE-события `image` игнорируются: в окне остаётся только текст.
    var showTaleImages: Bool {
        get { defaults.bool(forKey: Settings.showTaleImagesKey) }
        set { store(Settings.showTaleImagesKey, newValue) }
    }

    /// «Скорость чтения сказок» (задача 0230). По умолчанию «медленная»:
    /// паузы между фразами и абзацами длиннее. Обычный чат не затрагивается.
    var talePace: TalePace {
        get { TalePace.normalize(defaults.string(forKey: Settings.talePaceKey)) }
        set { store(Settings.talePaceKey, newValue.rawValue) }
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

    /// Слушать «Стоп, Ева», пока идёт «думаю» или озвучка (задача 0194).
    /// Включено по умолчанию; выключение гарантирует, что во время речи Евы
    /// микрофон не держится вовсе.
    var stopCommand: Bool {
        get { defaults.bool(forKey: Settings.stopCommandKey) }
        set { store(Settings.stopCommandKey, newValue) }
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
