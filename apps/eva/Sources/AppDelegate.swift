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

    private var isRecording = false
    private var isSpeaking = false
    private var isTerminating = false

    // MARK: - Запуск

    func applicationDidFinishLaunching(_ notification: Notification) {
        Settings.shared.registerDefaults()
        NSApp.setActivationPolicy(.accessory)
        buildMainMenu()
        buildStatusItem()
        installHotKey()
        wireRecorder()
        wireAnswer()
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
        log("настройки: сервер=\(server) модель=\(model) голос=\(voice) озвучка=\(speak)")

        startRequestedQuestionIfAny()
    }

    func applicationWillTerminate(_ notification: Notification) {
        log("выход")
        isTerminating = true
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
            self.listenItem?.title = recording
                ? "Остановить (⌥Space)"
                : "Слушать (⌥Space)"
            self.window.setStatus(recording ? "слушаю…" : "готово", isError: false)
            self.updateStatusIcon()
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
        window.show()
        window.showPartial("")
        recorder.start()
    }

    // MARK: - Чат и ответ

    private func wireAnswer() {
        answer.onStatus = { [weak self] text, isError in
            self?.window.setStatus(text, isError: isError)
        }
        answer.onAnswer = { [weak self] visible, thinking in
            self?.window.updateAnswer(visible: visible, thinking: thinking)
        }
        answer.onSpeakingChanged = { [weak self] speaking in
            guard let self else { return }
            self.isSpeaking = speaking
            self.updateStatusIcon()
        }
    }

    private func wireWindow() {
        window.onAsk = { [weak self] text in
            self?.ask(text)
        }
        window.onStop = { [weak self] in
            guard let self else { return }
            self.recorder.stop()
            self.answer.stop()
            self.isSpeaking = false
            self.updateStatusIcon()
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
        if !Settings.shared.speakReplies {
            answer.stopVoice()
        }
    }

    /// Единая точка «вопрос ушёл»: и голос, и клавиатура, и `--ask`.
    private func ask(_ question: String) {
        let trimmed = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let server = effectiveServerURL else {
            window.show()
            window.setStatus(
                "адрес сервера непонятен: \(Settings.shared.serverText) — откройте настройки",
                isError: true
            )
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
