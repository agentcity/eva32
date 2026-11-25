import AppKit
import Carbon.HIToolbox
import Foundation

/// Уведомление от глобального Carbon-хоткея: C-колбэк не может трогать акторы,
/// поэтому он только публикует событие, а обработчик живёт на главном потоке.
let evaHotKeyNotification = Notification.Name("EVA.hotKeyPressed")

/// Строка меню со статус-баром, глобальный хоткей ⌥Space и связка
/// «микрофон → /api/chat → окно → /api/tts».
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem?
    private var listenItem: NSMenuItem?
    private var hotKeyRef: EventHotKeyRef?
    private var hotKeyHandler: EventHandlerRef?

    private let window = MainWindowController()
    private let settingsWindow = SettingsWindowController()
    private let recorder = SpeechRecorder()
    private let answer = AnswerController()
    /// Фоновое ожидание «Привет, Ева» (задача 0187): живёт только в idle.
    private let wake = WakeWordListener()

    private var isRecording = false
    private var isSpeaking = false
    private var isTerminating = false

    /// Отложенное включение/выключение wake-слушателя: после стопа записи
    /// нужно дать `onFinal` уйти в чат, иначе ожидание поднимется на миг и
    /// тут же погаснет. Одна задача — одно актуальное намерение.
    private var wakeSync: Task<Void, Never>?
    /// Ожидание не поднялось (нет прав/офлайн-модели): не долбим системные
    /// API на каждый переход состояния. Снимается при смене настроек.
    private var wakeSuppressed = false

    // Большой круг: одно место, где решается, что он показывает. Флаги
    // обновляют `wireRecorder`/`wireAnswer`/`ask`, а `refreshVisualState`
    // выбирает состояние по приоритету и пишет переходы в журнал.
    private var visualState: EvaCircleState = .idle
    private var awaitingFirstToken = false
    /// Текст уже пошёл, но первый звук ещё синтезируется: круг остаётся
    /// «думаю», чтобы не мигать idle между первым токеном и озвучкой.
    private var awaitingAudio = false
    private var errorActive = false
    private var errorToken = 0

    // MARK: - Запуск

    func applicationDidFinishLaunching(_ notification: Notification) {
        Settings.shared.registerDefaults()
        NSApp.setActivationPolicy(.accessory)
        buildMainMenu()
        buildStatusItem()
        installHotKey()
        wireRecorder()
        wireAnswer()
        wireWake()
        wireWindow()
        observeSettings()

        let argv = CommandLine.arguments.dropFirst().joined(separator: " ")
        let bundleID = Bundle.main.bundleIdentifier ?? "?"
        let pid = ProcessInfo.processInfo.processIdentifier
        log("запуск: bundle=\(bundleID) pid=\(pid) args=[\(argv)]")
        let server = Settings.shared.serverText
        let model = Settings.shared.model.storageValue
        let voice = Settings.shared.voice
        let speak = Settings.shared.speakReplies
        let animate = Settings.shared.animateCircle
        let wakeWord = Settings.shared.wakeWord
        log("настройки: сервер=\(server) модель=\(model) голос=\(voice) озвучка=\(speak) "
            + "анимация=\(animate) ожидание=\(wakeWord)")

        if Launch.ask == nil {
            // При `--ask` вопрос уйдёт через 0,4 с: поднимать ожидание на этот
            // миг незачем.
            syncWake("запуск")
        }
        startRequestedQuestionIfAny()
        // Показываем окно при обычном запуске: без иконки в доке (accessory)
        // пользователь иначе не понимает, что приложение вообще открылось.
        // Флаг --background оставляет только статус-бар (автозапуск/агент).
        if !CommandLine.arguments.contains("--background") {
            window.showAndFocusInput()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        log("выход")
        isTerminating = true
        wakeSync?.cancel()
        wakeSync = nil
        wake.stop()
        window.setWakeWaiting(false)
        window.stopCircleAnimation()
        recorder.stop()
        answer.stop()
        if let hotKeyRef {
            UnregisterEventHotKey(hotKeyRef)
        }
        if let hotKeyHandler {
            RemoveEventHandler(hotKeyHandler)
        }
    }

    // MARK: - Статус-бар

    private func buildStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = item.button {
            button.image = NSImage(
                systemSymbolName: "waveform",
                accessibilityDescription: "Ева"
            )
            button.image?.isTemplate = true
            button.toolTip = "Ева: ⌥Space — сказать"
        }

        let menu = NSMenu()
        let listen = NSMenuItem(
            title: "Слушать (⌥Space)",
            action: #selector(toggleListening),
            keyEquivalent: ""
        )
        listen.target = self
        menu.addItem(listen)
        listenItem = listen

        let ask = NSMenuItem(
            title: "Спросить текстом…",
            action: #selector(showAskWindow),
            keyEquivalent: ""
        )
        ask.target = self
        menu.addItem(ask)

        menu.addItem(.separator())

        let settings = NSMenuItem(
            title: "Настройки…",
            action: #selector(openSettings),
            keyEquivalent: ","
        )
        settings.target = self
        menu.addItem(settings)

        let quit = NSMenuItem(title: "Выход", action: #selector(quit), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)

        item.menu = menu
        statusItem = item
        updateStatusIcon()
        log("статус-бар: пункт создан (иконка waveform, пунктов меню \(menu.items.count))")
    }

    private func updateStatusIcon() {
        guard let button = statusItem?.button else { return }
        let name: String
        if isRecording {
            name = "mic.fill"
        } else if isSpeaking {
            name = "speaker.wave.2.fill"
        } else {
            name = "waveform"
        }
        let image = NSImage(systemSymbolName: name, accessibilityDescription: "Ева")
        image?.isTemplate = true
        button.image = image
        button.toolTip = isRecording ? "Ева слушает — ⌥Space, чтобы закончить"
            : "Ева: ⌥Space — сказать"
    }

    /// Без этого меню не работают ⌘C/⌘V/⌘Z в полях ввода.
    private func buildMainMenu() {
        let mainMenu = NSMenu()

        let appItem = NSMenuItem()
        mainMenu.addItem(appItem)
        let appMenu = NSMenu()
        appItem.submenu = appMenu
        let about = NSMenuItem(
            title: "О программе Ева",
            action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)),
            keyEquivalent: ""
        )
        appMenu.addItem(about)
        appMenu.addItem(.separator())
        let settings = NSMenuItem(
            title: "Настройки…",
            action: #selector(openSettings),
            keyEquivalent: ","
        )
        settings.target = self
        appMenu.addItem(settings)
        appMenu.addItem(.separator())
        let quit = NSMenuItem(title: "Выход", action: #selector(quit), keyEquivalent: "q")
        quit.target = self
        appMenu.addItem(quit)

        let editItem = NSMenuItem()
        mainMenu.addItem(editItem)
        let editMenu = NSMenu(title: "Правка")
        editItem.submenu = editMenu
        editMenu.addItem(
            withTitle: "Отменить", action: Selector(("undo:")), keyEquivalent: "z"
        )
        editMenu.addItem(
            withTitle: "Повторить", action: Selector(("redo:")), keyEquivalent: "Z"
        )
        editMenu.addItem(.separator())
        editMenu.addItem(
            withTitle: "Вырезать", action: #selector(NSText.cut(_:)), keyEquivalent: "x"
        )
        editMenu.addItem(
            withTitle: "Копировать", action: #selector(NSText.copy(_:)), keyEquivalent: "c"
        )
        editMenu.addItem(
            withTitle: "Вставить", action: #selector(NSText.paste(_:)), keyEquivalent: "v"
        )
        editMenu.addItem(
            withTitle: "Выделить всё",
            action: #selector(NSText.selectAll(_:)),
            keyEquivalent: "a"
        )
        NSApp.mainMenu = mainMenu
    }

    // MARK: - Горячая клавиша ⌥Space

    private func installHotKey() {
        var eventType = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: UInt32(kEventHotKeyPressed)
        )
        // C-колбэк: ничего не захватывает, только публикует уведомление.
        let handler: EventHandlerUPP = { _, _, _ in
            DispatchQueue.main.async {
                NotificationCenter.default.post(name: evaHotKeyNotification, object: nil)
            }
            return noErr
        }
        let install = InstallEventHandler(
            GetApplicationEventTarget(),
            handler,
            1,
            &eventType,
            nil,
            &hotKeyHandler
        )
        guard install == noErr else {
            log("хоткей: не удалось поставить обработчик (\(install))")
            return
        }

        let hotKeyID = EventHotKeyID(signature: OSType(0x45564120), id: 1)  // 'EVA '
        let status = RegisterEventHotKey(
            UInt32(kVK_Space),
            UInt32(optionKey),
            hotKeyID,
            GetApplicationEventTarget(),
            0,
            &hotKeyRef
        )
        if status == noErr {
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(hotKeyPressed),
                name: evaHotKeyNotification,
                object: nil
            )
            log("хоткей: ⌥Space зарегистрирован")
        } else {
            log("хоткей: ⌥Space занят другим приложением (код \(status))")
        }
    }

    @objc private func hotKeyPressed() {
        toggleListening()
    }

    // MARK: - Микрофон

    private func wireRecorder() {
        recorder.onState = { [weak self] recording in
            guard let self else { return }
            self.isRecording = recording
            if recording {
                // Новое прослушивание отменяет затухающую ошибку: человек уже
                // действует, красная вспышка не должна перекрывать микрофон.
                self.clearVisualError()
                self.awaitingFirstToken = false
                self.awaitingAudio = false
            } else {
                self.window.setCircleLevel(0)
            }
            self.listenItem?.title = recording
                ? "Остановить (⌥Space)"
                : "Слушать (⌥Space)"
            self.window.setStatus(recording ? "слушаю…" : "готово", isError: false)
            self.updateStatusIcon()
            self.refreshVisualState(recording ? "микрофон включён" : "микрофон выключен")
            // После стопа запись ещё не значит «idle»: `onFinal` придёт сразу
            // за `onState(false)` и отправит вопрос. Небольшая пауза убирает
            // дребезг «включили ожидание — тут же выключили».
            self.syncWake(
                recording ? "микрофон включён" : "микрофон выключен",
                after: recording ? 0 : 0.45
            )
        }
        recorder.onLevel = { [weak self] level in
            self?.window.setCircleLevel(CGFloat(level))
        }
        recorder.onPartial = { [weak self] text in
            self?.window.showPartial(text)
        }
        recorder.onFinal = { [weak self] text in
            guard let self, !self.isTerminating else { return }
            self.window.showQuestion(text)
            self.ask(text)
        }
        recorder.onError = { [weak self] message in
            guard let self else { return }
            self.window.setStatus("речь: \(message)", isError: true)
            self.window.showPartial("")
            self.showVisualError(message)
            self.syncWake("ошибка речи")
        }
    }

    @objc private func toggleListening() {
        if recorder.isRecording {
            recorder.stop()
        } else {
            startListening()
        }
    }

    private func startListening() {
        // Двух захватов одновременно быть не должно: ожидание отпускает
        // микрофон до того, как запись его займёт.
        wakeSync?.cancel()
        if wake.isRunning {
            wake.stop()
        }
        window.setWakeWaiting(false)
        window.show()
        window.showPartial("")
        recorder.start()
    }

    // MARK: - Чат и ответ

    private func wireAnswer() {
        answer.onStatus = { [weak self] text, isError in
            guard let self else { return }
            self.window.setStatus(text, isError: isError)
            if isError { self.showVisualError(text) }
        }
        answer.onAnswer = { [weak self] visible, thinking in
            guard let self else { return }
            self.window.updateAnswer(visible: visible, thinking: thinking)
            // «Думаю» живёт до первого видимого токена, а если включена
            // озвучка — до первого звука (TTS может думать ещё пару сотен мс).
            if self.awaitingFirstToken, !visible.isEmpty {
                self.awaitingFirstToken = false
                self.awaitingAudio = self.expectsVoice && !self.isSpeaking
                self.refreshVisualState("первый видимый текст")
            }
        }
        answer.onFinished = { [weak self] in
            guard let self else { return }
            self.awaitingFirstToken = false
            if self.isSpeaking || !self.expectsVoice {
                self.awaitingAudio = false
            }
            self.refreshVisualState("ответ завершён")
            self.syncWake("ответ завершён")
            self.scheduleVoiceFallback()
        }
        answer.onSpeakingChanged = { [weak self] speaking in
            guard let self else { return }
            self.isSpeaking = speaking
            if speaking {
                self.awaitingFirstToken = false
                self.awaitingAudio = false
            }
            self.updateStatusIcon()
            self.refreshVisualState(speaking ? "озвучка пошла" : "озвучка кончилась")
            // Пока играет голос, микрофон ожидания выключен: иначе Ева
            // услышит собственное «Ева» и зациклится (задача 0187).
            self.syncWake(speaking ? "озвучка пошла" : "озвучка кончилась")
        }
        answer.onLevel = { [weak self] level in
            self?.window.setCircleLevel(CGFloat(level))
        }
    }

    /// Страховка: если озвучка включена, но звук так и не начался (TTS упал),
    /// круг не должен навсегда остаться «думаю», а ожидание — не вернуться.
    private func scheduleVoiceFallback() {
        guard expectsVoice else { return }
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 4_000_000_000)
            guard let self, !self.isTerminating else { return }
            guard self.awaitingAudio, !self.isSpeaking, !self.isRecording else { return }
            self.awaitingAudio = false
            self.refreshVisualState("озвучка не началась")
            self.syncWake("озвучка не началась")
        }
    }

    private func wireWindow() {
        window.onAsk = { [weak self] text in
            self?.ask(text)
        }
        window.onStop = { [weak self] in
            guard let self else { return }
            // Серверу — явный POST /api/cancel (задача 0185): обрыв SSE тоже
            // отменяет, но флаг в C-рантайме надёжнее и виден в метриках.
            self.cancelServerGeneration()
            self.recorder.stop()
            self.answer.stop()
            self.isSpeaking = false
            self.awaitingFirstToken = false
            self.awaitingAudio = false
            self.clearVisualError()
            self.window.setCircleLevel(0)
            self.refreshVisualState("стоп по кнопке")
            self.updateStatusIcon()
            self.syncWake("стоп по кнопке", after: 0.2)
            log("стоп по кнопке")
        }
        window.onSettings = { [weak self] in
            self?.settingsWindow.show()
        }
        window.onStopVoice = { [weak self] in
            self?.answer.stopVoice()
        }
    }

    private func observeSettings() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(settingsChanged),
            name: Settings.didChange,
            object: nil
        )
    }

    @objc private func settingsChanged() {
        window.reloadFromSettings()
        // Человек мог включить галочку уже после отказа в правах — даём
        // ожиданию ещё один шанс.
        wakeSuppressed = false
        if !Settings.shared.speakReplies {
            answer.stopVoice()
        }
        syncWake("настройки")
    }

    /// Единая точка «вопрос ушёл»: и голос, и клавиатура, и `--ask`.
    private func ask(_ question: String) {
        let trimmed = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let server = effectiveServerURL else {
            window.show()
            let message = "адрес сервера непонятен: \(Settings.shared.serverText) — откройте настройки"
            window.setStatus(message, isError: true)
            showVisualError(message)
            return
        }
        let config = ChatConfig(
            serverURL: server,
            model: effectiveModel,
            maxTokens: Settings.shared.maxTokens
        )
        window.show()
        window.showQuestion(trimmed)
        window.beginAnswer()
        log("вопрос (\"\(trimmed.prefix(120))\"): модель=\(config.model.storageValue) "
            + "сервер=\(server.absoluteString)")
        // «Думаю» — с этого момента до первого видимого текста. Ставим флаг
        // до `answer.ask`: старый поток может синхронно прислать «озвучка
        // кончилась», и это не должно вернуть круг в idle.
        awaitingFirstToken = true
        awaitingAudio = false
        clearVisualError()
        refreshVisualState("вопрос ушёл")
        syncWake("вопрос ушёл")
        answer.ask(
            trimmed,
            config: config,
            speak: Settings.shared.speakReplies && !Launch.noVoice,
            voiceID: Settings.shared.voice
        )
    }

    private var effectiveServerURL: URL? {
        if let override = Launch.serverOverride, let url = Settings.normalize(override) {
            return url
        }
        return Settings.shared.baseURL
    }

    private var effectiveModel: ModelSpec {
        if let override = Launch.modelOverride, let model = ModelSpec.parse(override) {
            return model
        }
        return Settings.shared.model
    }

    // MARK: - Меню

    @objc private func showAskWindow() {
        window.showAndFocusInput()
    }

    @objc private func openSettings() {
        settingsWindow.show()
    }

    @objc private func quit() {
        log("выход из меню")
        NSApp.terminate(nil)
    }

    // MARK: - Ожидание «Привет, Ева»

    private func wireWake() {
        wake.onWake = { [weak self] text in
            guard let self else { return }
            self.handleWake(text)
        }
        wake.onIssue = { [weak self] message in
            guard let self else { return }
            self.wakeSuppressed = true
            self.window.setWakeWaiting(false)
            guard Settings.shared.wakeWord else { return }
            self.window.setStatus("ожидание: \(message)", isError: true)
        }
        // Подъём слушателя асинхронный (сначала права и офлайн-модель),
        // поэтому пульсацию круга включает именно это событие.
        wake.onState = { [weak self] running in
            self?.window.setWakeWaiting(running)
        }
    }

    /// Услышали фразу: слушатель отпускает микрофон, дальше — обычная запись
    /// с авто-стопом по тишине, как после ⌥Space.
    private func handleWake(_ text: String) {
        guard !isTerminating, Settings.shared.wakeWord else { return }
        guard !isRecording, !isSpeaking else { return }
        wake.stop()
        window.setWakeWaiting(false)
        log("wake: перехожу к записи (\(text.prefix(80)))")
        startListening()
    }

    /// Единая точка решения «должно ли сейчас работать ожидание». Причина
    /// попадает только в отладочный журнал: сами строки «ожидание включено/
    /// выключено» пишет `WakeWordListener`.
    private func syncWake(_ reason: String, after delay: Double = 0) {
        wakeSync?.cancel()
        wakeSync = Task { @MainActor [weak self] in
            if delay > 0 {
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            }
            guard !Task.isCancelled, let self else { return }
            self.applyWake(reason)
        }
    }

    private func applyWake(_ reason: String) {
        guard !isTerminating else { return }
        let desired = Settings.shared.wakeWord
            && !wakeSuppressed
            && !isRecording
            && !isSpeaking
            && !awaitingFirstToken
            && !awaitingAudio
            && !errorActive
        if desired {
            if !wake.isRunning {
                log("wake: ожидание просит включиться — \(reason)")
                wake.start()
            }
        } else if wake.isRunning {
            log("wake: ожидание выключается — \(reason)")
            wake.stop()
        }
        window.setWakeWaiting(wake.isRunning)
    }

    /// «Стоп» сообщает серверу, что генерацию пора бросить.
    private func cancelServerGeneration() {
        guard let server = effectiveServerURL else { return }
        log("отмена: POST \(server.appendingPathComponent("api/cancel").absoluteString) {\"side\":0}")
        Task {
            let result = await ChatSession.cancelRemote(base: server, side: 0)
            log("отмена: \(result)")
        }
    }

    // MARK: - Большой круг

    /// Приоритет: ошибка → микрофон → ожидание первого токена → озвучка →
    /// покой. Причина попадает в журнал, чтобы по `eva.log` был виден весь
    /// сценарий: `круг: idle → listening — микрофон включён`.
    private func refreshVisualState(_ reason: String) {
        let next: EvaCircleState
        if errorActive {
            next = .error
        } else if isRecording {
            next = .listening
        } else if awaitingFirstToken || awaitingAudio {
            next = .thinking
        } else if isSpeaking {
            next = .speaking
        } else {
            next = .idle
        }
        applyVisualState(next, reason: reason)
    }

    private func applyVisualState(_ next: EvaCircleState, reason: String) {
        guard next != visualState else { return }
        log("круг: \(visualState.rawValue) → \(next.rawValue) — \(reason)")
        visualState = next
        window.setVisualState(next)
    }

    /// Красная вспышка ~1,2 с, потом круг возвращается к настоящему состоянию.
    /// Токен защищает от запоздавшего сброса прошлой ошибки.
    private func showVisualError(_ message: String) {
        errorActive = true
        errorToken += 1
        let token = errorToken
        awaitingFirstToken = false
        awaitingAudio = false
        window.setVisualError(message)
        // В статусе ошибка уже с префиксом «ошибка:», в журнале префикс не
        // дублируем — состояние и так названо.
        let prefix = "ошибка: "
        let reason = message.hasPrefix(prefix)
            ? String(message.dropFirst(prefix.count))
            : message
        applyVisualState(.error, reason: "\(reason.prefix(120))")
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            guard let self, self.errorToken == token else { return }
            self.errorActive = false
            self.refreshVisualState("вспышка ошибки прошла")
            self.syncWake("вспышка ошибки прошла")
        }
    }

    private func clearVisualError() {
        errorActive = false
        errorToken += 1
    }

    /// Озвучка включена в настройках и не выключена флагом `--no-voice`.
    private var expectsVoice: Bool {
        Settings.shared.speakReplies && !Launch.noVoice
    }

    // MARK: - Автозапуск запроса из аргументов (`--ask`)

    private func startRequestedQuestionIfAny() {
        guard let question = Launch.ask, !question.isEmpty else { return }
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: 400_000_000)
            guard let self else { return }
            log("--ask: \(question.prefix(120))")
            self.ask(question)
        }
    }
}
