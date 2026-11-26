import AppKit

/// Единственное окно приложения: живой текст распознавания, ответ (SSE) и
/// органы управления — озвучка, голос, стоп, ввод с клавиатуры.
@MainActor
final class MainWindowController: NSObject, NSWindowDelegate {
    private var panel: NSPanel?
    private let circles: CircleHub
    private var answerView: NSTextView!
    private var answerScroll: NSScrollView!
    private var thinkView: NSTextView!
    private var thinkScroll: NSScrollView!
    private var thinkBox: NSView!
    private var thinkButton: NSButton!
    private var thinkHeight: NSLayoutConstraint!
    /// Иллюстрация сказки (задача 0224) справа от текста ответа. Ширина 0,
    /// пока картинки нет: текст занимает всё место.
    private var taleImage: TaleImageView!
    private var taleImageExpanded: NSLayoutConstraint!
    private var taleImageCollapsed: NSLayoutConstraint!
    private var taleImageGap: NSLayoutConstraint!
    /// Ряд «текст ответа + картинка».
    private var answerRow: NSView!
    private var statusLabel: NSTextField!
    private var transcriptLabel: NSTextField!
    private var speakCheck: NSButton!
    private var voicePopup: NSPopUpButton!
    private var inputField: NSTextField!

    private var thinking = ""
    private var expandedThinking = false

    /// Отправить текст (голосом или с клавиатуры) в чат.
    var onAsk: ((String) -> Void)?
    var onStop: (() -> Void)?
    var onSettings: (() -> Void)?
    /// Свернуть в компактный круг (задача 0188).
    var onCollapse: (() -> Void)?

    /// Картинка занимает большую часть поля ответа (жалоба: в колонке 190 pt
    /// иллюстрации были крошечными). Доля высоты поля, не больше этого.
    static let taleImageHeightRatio: CGFloat = 0.62
    /// Зазор между текстом и картинкой (0, когда картинки нет).
    static let taleImageSpacing: CGFloat = 10

    init(circles: CircleHub) {
        self.circles = circles
        super.init()
    }

    // MARK: - Показ

    func show() {
        if panel == nil { build() }
        guard let panel else { return }
        panel.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    /// Спрятать главное окно, не закрывая его: компактный режим оставляет
    /// только круг, а окно с ответом возвращается по клику.
    func hide() {
        panel?.orderOut(nil)
    }

    /// Собрать окно, не показывая его. Нужно компактному режиму: вопрос и
    /// ответ копятся в скрытом окне, чтобы при развороте всё было на месте
    /// (задача 0188).
    func prepare() {
        if panel == nil { build() }
    }

    func showAndFocusInput() {
        show()
        inputField?.becomeFirstResponder()
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        sender.orderOut(nil)
        return false
    }

    // MARK: - Содержимое

    func beginAnswer() {
        answerView?.string = ""
        thinking = ""
        expandedThinking = false
        thinkView?.string = ""
        updateThinkVisibility()
        // Новый вопрос — чистая страница: прошлая иллюстрация не должна
        // висеть над новым ответом.
        clearTaleImage()
    }

    /// Показать картинку сказки рядом с текстом. Картинки не трогают круг:
    /// это не озвучка, состояние `speaking`/`idle` остаётся прежним.
    func showTaleImage(_ image: NSImage, alt: String = "") {
        if panel == nil { build() }
        guard let taleImage else { return }
        taleImage.isHidden = false
        // Большая картинка НАД текстом: поле не делится на узкую колонку,
        // иллюстрация занимает широкую область, текст идёт под ней.
        taleImageCollapsed?.isActive = false
        taleImageExpanded?.isActive = true
        taleImageGap?.constant = MainWindowController.taleImageSpacing
        // Проявление и кроссфейд — внутри TaleImageView; колонка при этом
        // занимает свою ширину сразу, чтобы текст не «дышал» вёрсткой.
        taleImage.show(image, alt: alt)
    }

    /// Убрать иллюстрацию: текст снова занимает всю ширину.
    func clearTaleImage() {
        taleImageExpanded?.isActive = false
        taleImageCollapsed?.isActive = true
        taleImageGap?.constant = 0
        taleImage?.isHidden = true
        taleImage?.clear()
    }

    /// Для самопроверки: картинка действительно в окне ответа, а не только
    /// загружена. Ширина 0 — колонка скрыта (только текст).
    var taleImageState: (visible: Bool, width: CGFloat) {
        let visible = taleImage?.hasImage == true && taleImage?.isHidden == false
        // Высоту кадра до первого layout не измерить (окно не показано),
        // поэтому сообщаем состояние активного констрейнта: он и задаёт
        // «большая картинка / картинки нет».
        let expanded = taleImageExpanded?.isActive == true
        return (visible, expanded ? 1 : 0)
    }

    func showQuestion(_ text: String) {
        transcriptLabel?.stringValue = text
    }

    func showPartial(_ text: String) {
        transcriptLabel?.stringValue = text.isEmpty ? "…" : text
    }

    func updateAnswer(visible: String, thinking newThinking: String) {
        answerView?.string = visible
        if let answerView, answerView.window != nil {
            answerView.scrollToEndOfDocument(nil)
        }
        if newThinking != thinking {
            thinking = newThinking
            thinkView?.string = newThinking
            updateThinkVisibility()
        }
    }

    func setStatus(_ text: String, isError: Bool) {
        statusLabel?.stringValue = text
        statusLabel?.textColor = isError ? .systemRed : .secondaryLabelColor
    }

    // MARK: - Сборка окна

    private func build() {
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 580, height: 560),
            styleMask: [.titled, .closable, .resizable, .utilityWindow],
            backing: .buffered,
            defer: false
        )
        panel.title = "Ева"
        panel.isReleasedWhenClosed = false
        panel.hidesOnDeactivate = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.delegate = self
        panel.contentMinSize = NSSize(width: 460, height: 460)
        panel.setFrameAutosaveName("EVA.mainWindow")
        panel.center()
        self.panel = panel

        let content = panel.contentView!
        let title = NSTextField(labelWithString: "Ева")
        title.font = .systemFont(ofSize: 15, weight: .semibold)

        statusLabel = NSTextField(labelWithString: "готово")
        statusLabel.font = .systemFont(ofSize: 11)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.alignment = .right
        statusLabel.lineBreakMode = .byTruncatingTail
        statusLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        let header = NSStackView(views: [title, statusLabel])
        header.orientation = .horizontal
        header.spacing = 8
        header.distribution = .fill

        // Новый верхний блок: большой круг состояния. Высота — доля ширины
        // окна (круг масштабируется), но не больше 240 pt и не меньше 110 pt.
        // Круг регистрируется в общем хабе: компактное окно получает то же
        // состояние, уровень и ошибки без второй копии логики (задача 0188).
        let circle = CircleView(frame: .zero)
        circles.register(circle)

        transcriptLabel = NSTextField(wrappingLabelWithString: "")
        transcriptLabel.font = .systemFont(ofSize: 14)
        transcriptLabel.textColor = .labelColor
        transcriptLabel.maximumNumberOfLines = 3
        transcriptLabel.lineBreakMode = .byTruncatingTail
        transcriptLabel.isSelectable = true
        transcriptLabel.stringValue = ""

        answerView = NSTextView(frame: .zero)
        answerView.isEditable = false
        answerView.isSelectable = true
        answerView.isRichText = false
        answerView.drawsBackground = false
        answerView.font = .systemFont(ofSize: 13.5)
        answerView.textContainerInset = NSSize(width: 8, height: 8)
        answerView.isVerticallyResizable = true
        answerView.isHorizontallyResizable = false
        answerView.autoresizingMask = [.width]
        answerView.minSize = .zero
        answerView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        answerView.textContainer?.widthTracksTextView = true
        answerView.textContainer?.containerSize = NSSize(
            width: 0,
            height: CGFloat.greatestFiniteMagnitude
        )

        answerScroll = NSScrollView(frame: .zero)
        answerScroll.documentView = answerView
        answerScroll.hasVerticalScroller = true
        answerScroll.hasHorizontalScroller = false
        answerScroll.autohidesScrollers = true
        answerScroll.borderType = .bezelBorder

        // Иллюстрация сказки — отдельная колонка справа от текста: вертикаль
        // окна не тратится, а картинка видна по ходу рассказа. Ряд собран
        // вручную, без `NSStackView`: у стека виды появляются в иерархии не
        // сразу, и констрейнт «высота картинки = высота текста» падал бы с
        // «no common ancestor». Колонка скрыта шириной 0 и зазором 0.
        taleImage = TaleImageView(frame: .zero)
        taleImage.translatesAutoresizingMaskIntoConstraints = false
        taleImage.isHidden = true

        let row = NSView(frame: .zero)
        row.translatesAutoresizingMaskIntoConstraints = false
        answerScroll.translatesAutoresizingMaskIntoConstraints = false
        row.addSubview(answerScroll)
        row.addSubview(taleImage)

        // Картинка — сверху на всю ширину поля; текст занимает остаток.
        // Высота: доля высоты ряда (большая иллюстрация) либо 0, когда
        // картинки нет. Зазор между ними — taleImageGap.
        taleImageExpanded = taleImage.heightAnchor.constraint(
            equalTo: row.heightAnchor,
            multiplier: MainWindowController.taleImageHeightRatio
        )
        taleImageCollapsed = taleImage.heightAnchor.constraint(equalToConstant: 0)
        taleImageCollapsed.isActive = true
        taleImageGap = answerScroll.topAnchor.constraint(
            equalTo: taleImage.bottomAnchor,
            constant: 0
        )
        NSLayoutConstraint.activate([
            answerScroll.bottomAnchor.constraint(equalTo: row.bottomAnchor),
            answerScroll.leadingAnchor.constraint(equalTo: row.leadingAnchor),
            answerScroll.trailingAnchor.constraint(equalTo: row.trailingAnchor),
            taleImage.topAnchor.constraint(equalTo: row.topAnchor),
            taleImage.leadingAnchor.constraint(equalTo: row.leadingAnchor),
            taleImage.trailingAnchor.constraint(equalTo: row.trailingAnchor),
            taleImageGap,
        ])
        answerRow = row

        thinkButton = NSButton(title: "размышления", target: self, action: #selector(toggleThinking))
        thinkButton.isBordered = false
        thinkButton.font = .systemFont(ofSize: 11)
        thinkButton.contentTintColor = .secondaryLabelColor
        thinkButton.alignment = .left

        thinkView = NSTextView(frame: .zero)
        thinkView.isEditable = false
        thinkView.isSelectable = true
        thinkView.isRichText = false
        thinkView.drawsBackground = false
        thinkView.font = .systemFont(ofSize: 12)
        thinkView.textColor = .secondaryLabelColor
        thinkView.textContainerInset = NSSize(width: 6, height: 6)
        thinkView.isVerticallyResizable = true
        thinkView.isHorizontallyResizable = false
        thinkView.autoresizingMask = [.width]
        thinkView.textContainer?.widthTracksTextView = true
        thinkView.textContainer?.containerSize = NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)

        thinkScroll = NSScrollView(frame: .zero)
        thinkScroll.documentView = thinkView
        thinkScroll.hasVerticalScroller = true
        thinkScroll.autohidesScrollers = true
        thinkScroll.borderType = .bezelBorder

        thinkBox = NSView(frame: .zero)
        thinkBox.addSubview(thinkButton)
        thinkBox.addSubview(thinkScroll)
        thinkButton.translatesAutoresizingMaskIntoConstraints = false
        thinkScroll.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            thinkButton.topAnchor.constraint(equalTo: thinkBox.topAnchor),
            thinkButton.leadingAnchor.constraint(equalTo: thinkBox.leadingAnchor),
            thinkButton.heightAnchor.constraint(equalToConstant: 16),
            thinkScroll.topAnchor.constraint(equalTo: thinkButton.bottomAnchor, constant: 4),
            thinkScroll.leadingAnchor.constraint(equalTo: thinkBox.leadingAnchor),
            thinkScroll.trailingAnchor.constraint(equalTo: thinkBox.trailingAnchor),
            thinkScroll.bottomAnchor.constraint(equalTo: thinkBox.bottomAnchor),
        ])
        thinkHeight = thinkBox.heightAnchor.constraint(equalToConstant: 0)
        thinkHeight.isActive = true
        updateThinkVisibility()

        speakCheck = NSButton(
            checkboxWithTitle: "Озвучивать",
            target: self,
            action: #selector(speakToggled)
        )
        speakCheck.state = Settings.shared.speakReplies ? .on : .off
        speakCheck.controlSize = .small

        voicePopup = NSPopUpButton(frame: .zero, pullsDown: false)
        voicePopup.controlSize = .small
        voicePopup.font = .systemFont(ofSize: 11)
        voicePopup.addItems(withTitles: SileroVoices.all.map(\.title))
        selectVoice(Settings.shared.voice)
        voicePopup.target = self
        voicePopup.action = #selector(voiceChanged)

        let spacer = NSView(frame: .zero)
        spacer.setContentHuggingPriority(.defaultLow, for: .horizontal)

        let stopButton = NSButton(title: "Стоп", target: self, action: #selector(stopTapped))
        stopButton.bezelStyle = .rounded
        stopButton.controlSize = .small

        // «Свернуть» (задача 0188): главное окно прячется, остаётся круг.
        let collapseButton = NSButton(
            title: "Свернуть",
            target: self,
            action: #selector(collapseTapped)
        )
        collapseButton.bezelStyle = .rounded
        collapseButton.controlSize = .small
        collapseButton.toolTip = "Оставить только круг поверх окон (⌘M)"

        let settingsButton = NSButton(
            title: "Настройки…",
            target: self,
            action: #selector(settingsTapped)
        )
        settingsButton.bezelStyle = .rounded
        settingsButton.controlSize = .small

        let controls = NSStackView(
            views: [speakCheck, voicePopup, spacer, stopButton, collapseButton, settingsButton]
        )
        controls.orientation = .horizontal
        controls.spacing = 8
        controls.distribution = .fill

        inputField = NSTextField(frame: .zero)
        inputField.placeholderString = "Спросите голосом (⌥Space) или напишите здесь…"
        inputField.font = .systemFont(ofSize: 13)
        inputField.target = self
        inputField.action = #selector(sendTyped)

        let sendButton = NSButton(title: "Отправить", target: self, action: #selector(sendTyped))
        sendButton.bezelStyle = .rounded
        sendButton.controlSize = .small

        let inputRow = NSStackView(views: [inputField, sendButton])
        inputRow.orientation = .horizontal
        inputRow.spacing = 8
        inputRow.distribution = .fill

        let stack = NSStackView(views: [
            header, circle, transcriptLabel, answerRow, thinkBox, controls, inputRow,
        ])
        stack.orientation = .vertical
        stack.alignment = .width
        stack.spacing = 10
        stack.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(stack)
        let circleHeight = circle.heightAnchor.constraint(
            equalTo: content.widthAnchor,
            multiplier: 0.28
        )
        circleHeight.priority = .defaultHigh
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: content.topAnchor, constant: 14),
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 16),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -16),
            stack.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -14),
            circle.heightAnchor.constraint(lessThanOrEqualToConstant: 240),
            circle.heightAnchor.constraint(greaterThanOrEqualToConstant: 110),
            circleHeight,
            answerScroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 120),
            transcriptLabel.heightAnchor.constraint(equalToConstant: 44),
        ])
    }

    // MARK: - Размышления

    private func updateThinkVisibility() {
        let hasThinking = !thinking.isEmpty
        let height: CGFloat = hasThinking ? (expandedThinking ? 110 : 20) : 0
        thinkHeight?.constant = height
        thinkBox?.isHidden = !hasThinking
        thinkScroll?.isHidden = !expandedThinking
        let mark = expandedThinking ? "▾" : "▸"
        thinkButton?.title = "\(mark) размышления (\(thinking.count) симв.)"
    }

    @objc private func toggleThinking() {
        expandedThinking.toggle()
        updateThinkVisibility()
        if expandedThinking { thinkView?.scrollToEndOfDocument(nil) }
    }

    // MARK: - Действия

    @objc private func speakToggled() {
        Settings.shared.speakReplies = speakCheck.state == .on
        if !Settings.shared.speakReplies { onStopVoice?() }
    }

    @objc private func voiceChanged() {
        guard let id = selectedVoiceID() else { return }
        Settings.shared.voice = id
    }

    @objc private func stopTapped() {
        onStop?()
    }

    @objc private func collapseTapped() {
        onCollapse?()
    }

    @objc private func settingsTapped() {
        onSettings?()
    }

    @objc private func sendTyped() {
        let text = inputField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        inputField.stringValue = ""
        onAsk?(text)
    }

    /// Остановить уже начатую озвучку, не трогая галочку.
    var onStopVoice: (() -> Void)?

    // MARK: - Синхронизация с настройками

    func reloadFromSettings() {
        speakCheck?.state = Settings.shared.speakReplies ? .on : .off
        selectVoice(Settings.shared.voice)
        // Сняли галочку «Показывать картинки сказок» — убираем и ту, что уже
        // на экране: текст остаётся один.
        if !Settings.shared.showTaleImages {
            clearTaleImage()
        }
    }

    private func selectVoice(_ id: String) {
        if let index = SileroVoices.all.firstIndex(where: { $0.id == id }) {
            voicePopup?.selectItem(at: index)
        } else {
            voicePopup?.selectItem(at: 0)
        }
    }

    private func selectedVoiceID() -> String? {
        guard let index = voicePopup?.indexOfSelectedItem,
              SileroVoices.all.indices.contains(index)
        else {
            return nil
        }
        return SileroVoices.all[index].id
    }
}
