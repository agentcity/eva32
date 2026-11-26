import AppKit
import Carbon.HIToolbox
import Foundation

/// Уведомление от глобального Carbon-хоткея: C-колбэк не может трогать акторы,
/// поэтому он только публикует событие, а обработчик живёт на главном потоке.
let evaHotKeyNotification = Notification.Name("EVA.hotKeyPressed")

/// Решение «кто держит микрофон»: wake-ожидание в покое или стоп-слушатель
/// во время речи. Ровно одно из полей истинно (или ни одного) — так два
/// `AVAudioEngine` никогда не работают одновременно.
struct EvaListenerPlan: Equatable, Sendable {
    var wake: Bool
    var stop: Bool
}

/// Строка меню со статус-баром, глобальный хоткей ⌥Space и связка
/// «микрофон → /api/chat → окно → /api/tts».
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem?
    private var listenItem: NSMenuItem?
    private var hotKeyRef: EventHotKeyRef?
    private var hotKeyHandler: EventHandlerRef?

    private let circles: CircleHub
    private let window: MainWindowController
    private let compact: CompactWindowController
    private let settingsWindow = SettingsWindowController()
    private let recorder = SpeechRecorder()
    private let answer = AnswerController()
    /// Фоновое ожидание «Привет, Ева» (задача 0187): живёт только в idle.
    private let wake = WakeWordListener()
    /// «Стоп, Ева» (задача 0194): живёт только пока Ева думает или говорит.
    /// С wake-слушателем никогда не пересекается — один движок на режим.
    private let stop = StopCommandListener()

    /// Компактный режим: главное окно спрятано, на экране только круг.
    private var isCompact = false

    private var isRecording = false
    private var isSpeaking = false
    private var isTerminating = false

    override init() {
        // Оба окна смотрят в один хаб: состояние круга не дублируется.
        let hub = CircleHub()
        circles = hub
        window = MainWindowController(circles: hub)
        compact = CompactWindowController(circles: hub)
        super.init()
    }

    // Follow-up (задача 0190): после ответа микрофон открывается сам, без
    // «Привет, Ева». Автоматика включается только когда ожидание фразы
    // включено: иначе микрофон после ответа не появляется вовсе.
    /// SSE-поток закрыт (ответ дочитан); озвучка может ещё играть.
    private var answerEnded = false
    /// Намерение открыть follow-up: ждём, пока смолкнет голос Евы.
    private var followUpArmed = false
    /// Текущая запись — follow-up (для журнала и приглушённого круга).
    private var isFollowUp = false

    /// Отложенное включение/выключение wake-слушателя: после стопа записи
    /// нужно дать `onFinal` уйти в чат, иначе ожидание поднимется на миг и
    /// тут же погаснет. Одна задача — одно актуальное намерение.
    private var wakeSync: Task<Void, Never>?
    /// Ожидание не поднялось (нет прав/офлайн-модели): не долбим системные
    /// API на каждый переход состояния. Снимается при смене настроек.
    private var wakeSuppressed = false
    /// То же для стоп-слушателя: если микрофон/распознавание не дали, не
    /// пытаемся поднимать его на каждой фразе (задача 0194).
    private var stopSuppressed = false

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
        wireStop()
        wireWindow()
        wireCompact()
        observeSettings()
        // Хаб мог быть создан до `registerDefaults`: значение галочки берём
        // из настроек уже после регистрации значений по умолчанию.
        circles.setAnimationEnabled(Settings.shared.animateCircle)

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
        let stopCommand = Settings.shared.stopCommand
        log("настройки: сервер=\(server) модель=\(model) голос=\(voice) озвучка=\(speak) "
            + "картинки=\(Settings.shared.showTaleImages) "
            + "анимация=\(animate) ожидание=\(wakeWord) "
            + "стоп-команда=\(stopCommand) "
            + "follow-up=\(Settings.shared.followUpListen)/\(Settings.shared.followUpSilence) с "
            + "старт-свёрнутым=\(Settings.shared.startCompact)")

        if Launch.ask == nil {
            // При `--ask` вопрос уйдёт через 0,4 с: поднимать ожидание на этот
            // миг незачем.
            syncWake("запуск")
        }
        startRequestedQuestionIfAny()
        // Показываем окно при обычном запуске: без иконки в доке (accessory)
        // пользователь иначе не понимает, что приложение вообще открылось.
        // Флаг --background оставляет только статус-бар (автозапуск/агент),
        // галочка «Запускаться свёрнутым» — только круг (задача 0188).
        if CommandLine.arguments.contains("--background") {
            log("вид: запуск в фоне — только статус-бар")
        } else if Settings.shared.startCompact, Launch.ask == nil {
            collapseToCompact(reason: "запуск свёрнутым (настройка)")
        } else {
            window.showAndFocusInput()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        log("выход")
        isTerminating = true
        followUpArmed = false
        answerEnded = false
        wakeSync?.cancel()
        wakeSync = nil
        stop.stop()
        wake.stop()
        circles.setWakeWaiting(false)
        circles.stopAnimation()
        compact.hide()
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

        // «Вид» (задача 0188): ⌘M — свернуть в круг, как стандартное «Свернуть»,
        // но окно не улетает в Dock, а прячется и оставляет круг на экране.
        let viewItem = NSMenuItem()
        mainMenu.addItem(viewItem)
        let viewMenu = NSMenu(title: "Вид")
        viewItem.submenu = viewMenu
        let collapse = NSMenuItem(
            title: "Свернуть в круг",
            action: #selector(collapseToCompactAction),
            keyEquivalent: "m"
        )
        collapse.target = self
        viewMenu.addItem(collapse)
        let expand = NSMenuItem(
            title: "Показать главное окно",
            action: #selector(expandAction),
            keyEquivalent: "M"
        )
        expand.target = self
        viewMenu.addItem(expand)

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
                self.circles.setLevel(0)
                // Круг возвращается к обычному виду: приглушение — только
                // пока идёт follow-up-запись.
                self.circles.setFollowUp(false)
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
            self?.circles.setLevel(CGFloat(level))
        }
        recorder.onPartial = { [weak self] text in
            self?.window.showPartial(text)
        }
        recorder.onFinal = { [weak self] text in
            guard let self, !self.isTerminating else { return }
            if self.isFollowUp {
                log("follow-up: услышала вопрос, отправляю")
                self.isFollowUp = false
                self.circles.setFollowUp(false)
            }
            self.window.showQuestion(text)
            self.ask(text)
        }
        // Follow-up без слов — не ошибка: тихо закрываемся и возвращаемся
        // в ожидание «Привет, Ева» (задача 0190).
        recorder.onFollowUpEnd = { [weak self] end in
            guard let self, !self.isTerminating else { return }
            self.isFollowUp = false
            switch end {
            case .silence(let seconds):
                log("follow-up: тишина \(seconds) с — ушла в ожидание")
            case .manual:
                log("follow-up: остановлено без слов — ушла в ожидание")
            }
            self.window.setStatus(
                Settings.shared.wakeWord ? "жду „Привет, Ева“" : "готово",
                isError: false
            )
            self.refreshVisualState("follow-up: тишина")
            self.syncWake("follow-up: тишина", after: 0.2)
        }
        recorder.onError = { [weak self] message in
            guard let self else { return }
            self.isFollowUp = false
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
            // В компактном режиме ⌥Space не разворачивает главное окно:
            // человек свернул Еву именно ради круга.
            startListening(present: !isCompact)
        }
    }

    /// `followUp: true` — автоматическое включение после ответа (задача 0190).
    /// Отличие от ⌥Space: окно не поднимаем и не перехватываем фокус, а
    /// тишина до первого слова считается по настройке, а не 6 с.
    ///
    /// `present: false` — не показывать главное окно (компактный режим:
    /// запись включена двойным кликом по кругу).
    private func startListening(followUp: Bool = false, present: Bool = true) {
        // Двух захватов одновременно быть не должно: ожидание и стоп-слушатель
        // отпускают микрофон до того, как запись его займёт.
        wakeSync?.cancel()
        if wake.isRunning {
            wake.stop()
        }
        if stop.isActive {
            stop.stop()
        }
        circles.setWakeWaiting(false)
        if followUp {
            // Ответ только что показан; активировать приложение после каждого
            // ответа — навязчиво.
            window.showPartial("")
            recorder.followUpSilence = Settings.shared.followUpSilenceValue
            log("follow-up: слушаю")
        } else {
            // Ручное прослушивание отменяет отложенный follow-up.
            followUpArmed = false
            if present {
                window.show()
            }
            window.showPartial("")
        }
        isFollowUp = followUp
        circles.setFollowUp(followUp)
        recorder.start(followUp: followUp)
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
                // silent-ответ (запись сказки уже играет) звука не даст:
                // ждать его — значит держать круг в «думаю» впустую.
                self.awaitingAudio = self.expectsVoice && !self.isSpeaking
                    && !self.answer.isSilentAnswer
                self.refreshVisualState("первый видимый текст")
                // Голос ещё синтезируется — стоп-слушатель остаётся; если
                // озвучки нет, уступаем место wake-ожиданию (задача 0194).
                self.syncWake("первый видимый текст")
            }
        }
        answer.onFinished = { [weak self] in
            guard let self else { return }
            self.awaitingFirstToken = false
            if self.isSpeaking || !self.expectsVoice || self.answer.isSilentAnswer {
                self.awaitingAudio = false
            }
            self.answerEnded = true
            self.refreshVisualState("ответ завершён")
            self.armFollowUp("ответ завершён")
            self.syncWake("ответ завершён")
            self.scheduleVoiceFallback()
        }
        // Озвучка доиграла: только теперь можно открыть микрофон, иначе Ева
        // услышит собственный голос (и «Ева» в нём).
        answer.onVoiceDrained = { [weak self] in
            self?.startFollowUpIfReady("озвучка доиграла")
        }
        answer.onSpeakingChanged = { [weak self] speaking in
            guard let self else { return }
            let changed = self.isSpeaking != speaking
            self.isSpeaking = speaking
            if speaking {
                self.awaitingFirstToken = false
                self.awaitingAudio = false
            } else {
                // Голос смолк: если follow-up ждал именно этого — открываем
                // микрофон до смены состояния круга, чтобы не мигать «idle»
                // между «говорю» и «слушаю».
                self.startFollowUpIfReady("озвучка кончилась")
            }
            // `voice.stop()` в начале нового вопроса тоже присылает `false`;
            // повторный переход не должен дёргать слушателей и журнал.
            guard changed else { return }
            self.updateStatusIcon()
            self.refreshVisualState(speaking ? "озвучка пошла" : "озвучка кончилась")
            // Пока играет голос, микрофон ожидания выключен: иначе Ева
            // услышит собственное «Ева» и зациклится (задача 0187), а на его
            // место встаёт стоп-слушатель (задача 0194).
            self.syncWake(speaking ? "озвучка пошла" : "озвучка кончилась")
        }
        answer.onLevel = { [weak self] level in
            self?.circles.setLevel(CGFloat(level))
        }
        // Иллюстрация сказки (задача 0224): только окно. Круг не трогаем —
        // картинка не озвучка, состояние speaking/idle от неё не зависит.
        answer.onImage = { [weak self] image, alt in
            self?.window.showTaleImage(image, alt: alt)
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
            // Голос не зазвучал: follow-up может открываться — ждать больше
            // нечего (если очередь синтеза не пуста, `onVoiceDrained` догонит).
            self.armFollowUp("озвучка не началась")
            self.syncWake("озвучка не началась")
        }
    }

    private func wireWindow() {
        window.onAsk = { [weak self] text in
            self?.ask(text)
        }
        window.onStop = { [weak self] in
            guard let self else { return }
            log("стоп по кнопке")
            self.performStop(reason: "стоп по кнопке")
        }
        window.onSettings = { [weak self] in
            self?.settingsWindow.show()
        }
        window.onStopVoice = { [weak self] in
            self?.answer.stopVoice()
        }
        window.onCollapse = { [weak self] in
            self?.collapseToCompact(reason: "кнопка «Свернуть»")
        }
    }

    /// Единый путь «остановить всё»: кнопка «Стоп» и голосовая команда
    /// «Стоп, Ева» (задача 0194). Гасит озвучку, отменяет генерацию тем же
    /// `POST /api/cancel`, что и кнопка, возвращает круг в idle/listening и
    /// опускает follow-up: после явного «Стоп» микрофон сам не открывается.
    private func performStop(reason: String) {
        // Что именно глушим — по состоянию ДО остановки: `answer.stop()`
        // синхронно пришлёт «озвучка кончилась» и обнулит флаги.
        let voiceWasBusy = isSpeaking || !answer.voiceIsIdle
        let generationWasBusy = awaitingFirstToken || awaitingAudio || !answerEnded
        answerEnded = false
        followUpArmed = false
        isFollowUp = false
        circles.setFollowUp(false)
        // Серверу — явный POST /api/cancel (задача 0185): обрыв SSE тоже
        // отменяет, но флаг в C-рантайме надёжнее и виден в метриках.
        cancelServerGeneration()
        recorder.stop()
        answer.stop()
        isSpeaking = false
        awaitingFirstToken = false
        awaitingAudio = false
        clearVisualError()
        circles.setLevel(0)
        refreshVisualState(reason)
        updateStatusIcon()
        syncWake(reason, after: 0.2)
        if voiceWasBusy {
            log("стоп: озвучка остановлена")
        }
        if generationWasBusy {
            log("стоп: генерация отменена")
        }
    }

    /// Компактный режим (задача 0188). Обратные вызовы ведут в те же методы,
    /// что и обычные действия: второго «слушать» или второго «развернуть» нет.
    private func wireCompact() {
        compact.onExpand = { [weak self] in
            self?.expandFromCompact(reason: "клик по кругу")
        }
        compact.onToggleListening = { [weak self] in
            self?.toggleListeningInCompact()
        }
        compact.onSettings = { [weak self] in
            self?.settingsWindow.show()
        }
        compact.onQuit = { [weak self] in
            self?.quit()
        }
    }

    // MARK: - Компактный режим

    @objc private func collapseToCompactAction() {
        collapseToCompact(reason: "⌘M")
    }

    @objc private func expandAction() {
        if isCompact {
            expandFromCompact(reason: "меню «Вид»")
        } else {
            window.show()
        }
    }

    /// Спрятать главное окно и показать маленький круг. Приложение не
    /// активируется: компактное окно — nonactivatingPanel.
    private func collapseToCompact(reason: String) {
        guard !isTerminating else { return }
        guard !isCompact else {
            log("вид: уже свёрнут — \(reason) игнорирую")
            return
        }
        isCompact = true
        // Окно не показываем, но собираем: ответ и статус копятся в нём, и
        // после разворота человек видит тот же разговор, а не пустое окно.
        window.prepare()
        window.hide()
        compact.show()
        let frame = compact.frame.map(CompactLayout.description) ?? "?"
        log("вид: свёрнут — \(reason); круг \(frame), главное окно скрыто")
    }

    /// Вернуть главное окно, спрятав круг. Это единственный путь разворота,
    /// поэтому окно и круг не могут оказаться видимыми одновременно.
    private func expandFromCompact(reason: String) {
        guard isCompact else {
            log("вид: уже развёрнут — \(reason) игнорирую")
            return
        }
        isCompact = false
        compact.hide()
        window.show()
        log("вид: развёрнут — \(reason); круг скрыт")
    }

    /// Двойной клик по кругу — то же, что ⌥Space, но без разворота окна:
    /// человек свернул Еву именно для того, чтобы она не занимала место.
    private func toggleListeningInCompact() {
        if recorder.isRecording {
            recorder.stop()
        } else {
            startListening(present: false)
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
        // Галочка «Анимация круга» действует и на компактный круг.
        circles.setAnimationEnabled(Settings.shared.animateCircle)
        // Человек мог включить галочку уже после отказа в правах — даём
        // ожиданию и стоп-слушателю ещё один шанс.
        wakeSuppressed = false
        stopSuppressed = false
        if !Settings.shared.speakReplies {
            answer.stopVoice()
        }
        // Выключили «Слушать после ответа» или само ожидание — отложенный
        // follow-up больше не нужен.
        if !AppDelegate.followUpAllowed(
            followUpListen: Settings.shared.followUpListen,
            wakeWord: Settings.shared.wakeWord
        ) {
            followUpArmed = false
        }
        syncWake("настройки")
    }

    /// Единая точка «вопрос ушёл»: и голос, и клавиатура, и `--ask`.
    private func ask(_ question: String) {
        let trimmed = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let server = effectiveServerURL else {
            if !isCompact { window.show() }
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
        // В компактном режиме окно не всплывает на каждый ответ: вопрос и
        // ответ обновляют скрытое окно, а на экране остаётся круг.
        if !isCompact { window.show() }
        window.showQuestion(trimmed)
        window.beginAnswer()
        log("вопрос (\"\(trimmed.prefix(120))\"): модель=\(config.model.storageValue) "
            + "сервер=\(server.absoluteString)")
        // «Думаю» — с этого момента до первого видимого текста. Ставим флаг
        // до `answer.ask`: старый поток может синхронно прислать «озвучка
        // кончилась», и это не должно вернуть круг в idle.
        awaitingFirstToken = true
        awaitingAudio = false
        // Новый вопрос начинает цикл заново: старые намерения follow-up
        // относятся к прошлому ответу.
        answerEnded = false
        followUpArmed = false
        isFollowUp = false
        circles.setFollowUp(false)
        // Новый ответ — новый шанс поднять стоп-слушатель (например, права
        // выдали только что): неудача прошлого ответа не глушит команду
        // навсегда. Wake-подавление так не сбрасываем — оно про TCC.
        stopSuppressed = false
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
            self.circles.setWakeWaiting(false)
            guard Settings.shared.wakeWord else { return }
            self.window.setStatus("ожидание: \(message)", isError: true)
        }
        // Подъём слушателя асинхронный (сначала права и офлайн-модель),
        // поэтому пульсацию круга включает именно это событие.
        wake.onState = { [weak self] running in
            self?.circles.setWakeWaiting(running)
        }
    }

    /// Услышали фразу: слушатель отпускает микрофон, дальше — обычная запись
    /// с авто-стопом по тишине, как после ⌥Space.
    private func handleWake(_ text: String) {
        guard !isTerminating, Settings.shared.wakeWord else { return }
        guard !isRecording, !isSpeaking else { return }
        wake.stop()
        circles.setWakeWaiting(false)
        log("wake: перехожу к записи (\(text.prefix(80)))")
        startListening(present: !isCompact)
    }

    // MARK: - «Стоп, Ева» (задача 0194)

    private func wireStop() {
        stop.onStop = { [weak self] text in
            guard let self else { return }
            self.handleStopCommand(text)
        }
        stop.onIssue = { [weak self] message in
            guard let self else { return }
            // Причина уже в журнале («стоп: …»); повторять её в статусе поверх
            // ответа/озвучки не нужно. До смены настроек не поднимаем повторно.
            self.stopSuppressed = true
            self.logStopIssue(message)
        }
        // Текущая фраза нужна эхо-защите: если Ева сама произносит «стоп»,
        // её же голос из колонок — не команда.
        stop.currentSpokenPhrase = { [weak self] in
            self?.answer.currentVoiceText
        }
    }

    /// Услышали «стоп»/«хватит»/«тише»: сначала глушим звук, и только потом
    /// отпускаем микрофон — отклик важнее, чем порядок сворачивания движка.
    private func handleStopCommand(_ text: String) {
        guard !isTerminating, Settings.shared.stopCommand else { return }
        log("стоп-команда: глушу („\(text.prefix(80))“)")
        performStop(reason: "стоп по голосу")
        stop.stop()
    }

    /// Слушание стоп-команды не поднялось — пишем в статус только когда он не
    /// занят ответом: иначе сообщение мгновенно перекрылось бы озвучкой.
    private func logStopIssue(_ message: String) {
        if !isSpeaking, !awaitingFirstToken, !awaitingAudio {
            window.setStatus("стоп-команда: \(message)", isError: true)
        }
    }

    /// Единая точка решения «какой слушатель держит микрофон». Причина
    /// попадает только в отладочный журнал: сами строки «включено/выключено»
    /// пишут слушатели.
    ///
    /// Один движок на режим: пока Ева думает или говорит — только
    /// `StopCommandListener`, в покое — только `WakeWordListener`. Запись
    /// (`SpeechRecorder`) и вспышка ошибки исключают обоих.
    private func syncWake(_ reason: String, after delay: Double = 0) {
        wakeSync?.cancel()
        wakeSync = Task { @MainActor [weak self] in
            if delay > 0 {
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            }
            guard !Task.isCancelled, let self else { return }
            self.applyListeners(reason)
        }
    }

    private func applyListeners(_ reason: String) {
        guard !isTerminating else { return }
        let plan = AppDelegate.listenerPlan(
            wakeWord: Settings.shared.wakeWord,
            stopCommand: Settings.shared.stopCommand,
            wakeSuppressed: wakeSuppressed,
            stopSuppressed: stopSuppressed,
            isRecording: isRecording,
            isSpeaking: isSpeaking,
            isThinking: awaitingFirstToken || awaitingAudio,
            followUpArmed: followUpArmed,
            errorActive: errorActive
        )
        // Сначала отпускаем микрофон теми, кому он больше не нужен, и только
        // потом поднимаем нужный: два `AVAudioEngine` не должны пересечься.
        if !plan.stop, stop.isActive {
            log("стоп: слушание выключается — \(reason)")
            stop.stop()
        }
        if !plan.wake, wake.isRunning {
            log("wake: ожидание выключается — \(reason)")
            wake.stop()
        }
        if plan.stop, !stop.isActive {
            log("стоп: слушание включается — \(reason)")
            stop.start()
        }
        if plan.wake, !wake.isRunning {
            log("wake: ожидание просит включиться — \(reason)")
            wake.start()
        }
        circles.setWakeWaiting(wake.isRunning)
    }

    // MARK: - Follow-up: слушаю после ответа (задача 0190)

    /// Решение «открыть микрофон после ответа». Настройки проверяем в момент
    /// решения, а не только на старте ответа: галочку могли снять, пока Ева
    /// говорила. Если голос ещё звучит или синтезируется — ждём
    /// `onVoiceDrained`, а не открываем микрофон поверх собственной озвучки.
    private func armFollowUp(_ reason: String) {
        guard AppDelegate.followUpAllowed(
            followUpListen: Settings.shared.followUpListen,
            wakeWord: Settings.shared.wakeWord,
            answerFailed: answer.didFail
        ) else {
            followUpArmed = false
            return
        }
        guard answerEnded, !isTerminating, !isRecording else { return }
        followUpArmed = true
        guard !isSpeaking, answer.voiceIsIdle else { return }
        startFollowUpIfReady(reason)
    }

    /// Открыть follow-up, если намерение ещё в силе и голос умолк. Вызывается
    /// из `armFollowUp`, по `onVoiceDrained` и из страховки озвучки — поэтому
    /// проверки повторяются: между ними мог прийти «Стоп» или новый вопрос.
    private func startFollowUpIfReady(_ reason: String) {
        guard followUpArmed, answerEnded, !isTerminating, !isRecording else { return }
        guard AppDelegate.followUpAllowed(
            followUpListen: Settings.shared.followUpListen,
            wakeWord: Settings.shared.wakeWord,
            answerFailed: answer.didFail
        ) else {
            followUpArmed = false
            return
        }
        guard !isSpeaking, answer.voiceIsIdle else { return }
        followUpArmed = false
        log("follow-up: открываю микрофон — \(reason)")
        startListening(followUp: true)
    }

    /// Автоматическое открытие микрофона после ответа возможно только при
    /// включённом ожидании «Привет, Ева»: без галочки поведение прежнее
    /// (микрофон — только по ⌥Space). Ошибка ответа тоже отменяет автоматику,
    /// иначе «слушаю…» мгновенно перекрыло бы текст ошибки. Вынесено сюда,
    /// чтобы правило проверялось без микрофона и UI.
    nonisolated static func followUpAllowed(
        followUpListen: Bool,
        wakeWord: Bool,
        answerFailed: Bool = false
    ) -> Bool {
        followUpListen && wakeWord && !answerFailed
    }

    /// Кто из двух слушателей держит микрофон прямо сейчас. `wake` и `stop`
    /// никогда не включены одновременно — это и есть требование «один движок
    /// на режим» (задача 0194); проверяется в `--selftest-stop`.
    nonisolated static func listenerPlan(
        wakeWord: Bool,
        stopCommand: Bool,
        wakeSuppressed: Bool,
        stopSuppressed: Bool,
        isRecording: Bool,
        isSpeaking: Bool,
        isThinking: Bool,
        followUpArmed: Bool,
        errorActive: Bool
    ) -> EvaListenerPlan {
        let busy = isRecording || errorActive
        let stop = stopCommand && !stopSuppressed && !busy && (isSpeaking || isThinking)
        let wake = wakeWord && !wakeSuppressed && !busy
            && !isSpeaking && !isThinking && !followUpArmed
            && !stop
        return EvaListenerPlan(wake: wake, stop: stop)
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
        circles.setState(next)
    }

    /// Красная вспышка ~1,2 с, потом круг возвращается к настоящему состоянию.
    /// Токен защищает от запоздавшего сброса прошлой ошибки.
    private func showVisualError(_ message: String) {
        errorActive = true
        errorToken += 1
        let token = errorToken
        awaitingFirstToken = false
        awaitingAudio = false
        circles.showError(message)
        // В статусе ошибка уже с префиксом «ошибка:», в журнале префикс не
        // дублируем — состояние и так названо.
        let prefix = "ошибка: "
        let reason = message.hasPrefix(prefix)
            ? String(message.dropFirst(prefix.count))
            : message
        applyVisualState(.error, reason: "\(reason.prefix(120))")
        // Пока висит красная вспышка, микрофон не поднимаем ни одним из
        // слушателей: сначала пусть человек увидит, что случилось.
        syncWake("ошибка")
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
