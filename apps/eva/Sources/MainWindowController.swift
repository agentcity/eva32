import AppKit

/// Единственное окно приложения: живой текст распознавания, ответ (SSE) и
/// органы управления — озвучка, голос, стоп, ввод с клавиатуры.
@MainActor
final class MainWindowController: NSObject, NSWindowDelegate {
    private var panel: NSPanel?
    private var circleView: CircleView?
    private var answerView: NSTextView!
    private var answerScroll: NSScrollView!
    private var thinkView: NSTextView!
    private var thinkScroll: NSScrollView!
    private var thinkBox: NSView!
    private var thinkButton: NSButton!
    private var thinkHeight: NSLayoutConstraint!
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

    // MARK: - Показ

    func show() {
        if panel == nil { build() }
        guard let panel else { return }
        panel.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
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

    // MARK: - Большой круг

    /// Смена визуального состояния (AppDelegate уже залогировал переход).
    func setVisualState(_ state: EvaCircleState) {
        circleView?.visualState = state
    }

    /// Уровень микрофона или озвучки, 0…1.
    func setCircleLevel(_ level: CGFloat) {
        circleView?.setLevel(level)
    }

    /// Красная вспышка с текстом ошибки (полный текст — в статусе шапки).
    func setVisualError(_ message: String) {
        circleView?.showError(message)
    }

    /// При закрытии приложения/окна не оставляем тикающий таймер.
    func stopCircleAnimation() {
        circleView?.stopAnimation()
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
        let circle = CircleView(frame: .zero)
        circle.animationEnabled = Settings.shared.animateCircle
        circleView = circle

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
        voicePopup.addItems(withTitles: VoicePresets.all.map(\.title))
        selectVoice(Settings.shared.voice)
        voicePopup.target = self
        voicePopup.action = #selector(voiceChanged)

        let spacer = NSView(frame: .zero)
        spacer.setContentHuggingPriority(.defaultLow, for: .horizontal)

        let stopButton = NSButton(title: "Стоп", target: self, action: #selector(stopTapped))
        stopButton.bezelStyle = .rounded
        stopButton.controlSize = .small

        let settingsButton = NSButton(
            title: "Настройки…",
            target: self,
            action: #selector(settingsTapped)
        )
        settingsButton.bezelStyle = .rounded
        settingsButton.controlSize = .small

        let controls = NSStackView(views: [speakCheck, voicePopup, spacer, stopButton, settingsButton])
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
            header, circle, transcriptLabel, answerScroll, thinkBox, controls, inputRow,
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
        circleView?.animationEnabled = Settings.shared.animateCircle
        selectVoice(Settings.shared.voice)
    }

    private func selectVoice(_ id: String) {
        if let index = VoicePresets.all.firstIndex(where: { $0.id == id }) {
            voicePopup?.selectItem(at: index)
        } else {
            voicePopup?.selectItem(at: 0)
        }
    }

    private func selectedVoiceID() -> String? {
        guard let index = voicePopup?.indexOfSelectedItem,
              VoicePresets.all.indices.contains(index)
        else {
            return nil
        }
        return VoicePresets.all[index].id
    }
}
