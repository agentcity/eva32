import AppKit

/// Окно настроек: адрес сервера, модель из `/api/backends`, пресет голоса и
/// озвучка. Значения хранятся в UserDefaults (см. `Settings`).
@MainActor
final class SettingsWindowController: NSObject, NSWindowDelegate {
    private var window: NSWindow?
    private var serverField: NSTextField!
    private var modelPopup: NSPopUpButton!
    private var voicePopup: NSPopUpButton!
    private var speakCheck: NSButton!
    private var animateCheck: NSButton!
    private var startCompactCheck: NSButton!
    private var wakeCheck: NSButton!
    private var followUpCheck: NSButton!
    private var followUpPopup: NSPopUpButton!
    private var statusLabel: NSTextField!

    /// Список моделей, полученный от сервера; текущая настройка добавляется
    /// даже если сервер её не знает (чтобы не потерять выбор).
    private var models: [ModelSpec] = []
    private var loadingModels = false

    func show() {
        if window == nil { build() }
        guard let window else { return }
        reloadControls()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        loadModels()
    }

    func windowWillClose(_ notification: Notification) {
        // Окно переиспользуется: прячем, а не уничтожаем.
        window?.orderOut(nil)
    }

    // MARK: - Сборка

    private func build() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 520, height: 520),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Настройки Евы"
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.center()
        self.window = window

        serverField = NSTextField(frame: .zero)
        serverField.placeholderString = Settings.defaultServer
        serverField.font = .systemFont(ofSize: 12)
        serverField.target = self
        serverField.action = #selector(save)

        modelPopup = NSPopUpButton(frame: .zero, pullsDown: false)
        modelPopup.controlSize = .small
        modelPopup.font = .systemFont(ofSize: 11)
        modelPopup.target = self
        modelPopup.action = #selector(modelChanged)

        let refresh = NSButton(title: "Обновить", target: self, action: #selector(loadModels))
        refresh.bezelStyle = .rounded
        refresh.controlSize = .small
        let modelRow = NSStackView(views: [modelPopup, refresh])
        modelRow.orientation = .horizontal
        modelRow.spacing = 8
        modelRow.distribution = .fill

        voicePopup = NSPopUpButton(frame: .zero, pullsDown: false)
        voicePopup.controlSize = .small
        voicePopup.font = .systemFont(ofSize: 11)
        voicePopup.addItems(withTitles: VoicePresets.all.map(\.title))
        voicePopup.target = self
        voicePopup.action = #selector(voiceChanged)

        speakCheck = NSButton(
            checkboxWithTitle: "Озвучивать ответы голосом",
            target: self,
            action: #selector(speakChanged)
        )
        speakCheck.controlSize = .small

        animateCheck = NSButton(
            checkboxWithTitle: "Анимация круга (меньше CPU)",
            target: self,
            action: #selector(animateChanged)
        )
        animateCheck.controlSize = .small

        // Запуск сразу в компактном режиме (задача 0188): по умолчанию
        // выключено, обычный запуск по-прежнему показывает главное окно.
        startCompactCheck = NSButton(
            checkboxWithTitle: "Запускаться свёрнутым (только круг)",
            target: self,
            action: #selector(startCompactChanged)
        )
        startCompactCheck.controlSize = .small

        let startCompactHint = NSTextField(
            wrappingLabelWithString: "при следующем запуске — без главного окна, "
                + "круг в правом нижнем углу; ⌘M сворачивает и сейчас"
        )
        startCompactHint.font = .systemFont(ofSize: 11)
        startCompactHint.textColor = .secondaryLabelColor

        wakeCheck = NSButton(
            checkboxWithTitle: "Слушать „Привет, Ева“",
            target: self,
            action: #selector(wakeChanged)
        )
        wakeCheck.controlSize = .small

        // Follow-up (задача 0190): микрофон после ответа, без «Привет, Ева».
        followUpCheck = NSButton(
            checkboxWithTitle: "Слушать после ответа",
            target: self,
            action: #selector(followUpChanged)
        )
        followUpCheck.controlSize = .small

        followUpPopup = NSPopUpButton(frame: .zero, pullsDown: false)
        followUpPopup.controlSize = .small
        followUpPopup.font = .systemFont(ofSize: 11)
        followUpPopup.addItems(
            withTitles: Settings.followUpSilenceOptions.map { "\($0) с" }
        )
        followUpPopup.target = self
        followUpPopup.action = #selector(followUpSilenceChanged)

        let followUpPauseLabel = NSTextField(
            labelWithString: "тишины — и возвращаюсь к „Привет, Ева“"
        )
        followUpPauseLabel.font = .systemFont(ofSize: 11)
        followUpPauseLabel.textColor = .secondaryLabelColor
        let followUpRow = NSStackView(views: [followUpPopup, followUpPauseLabel])
        followUpRow.orientation = .horizontal
        followUpRow.spacing = 8
        followUpRow.distribution = .fill

        let followUpHint = NSTextField(
            wrappingLabelWithString: "после ответа микрофон открыт сразу; "
                + "работает при включённом ожидании «Привет, Ева»"
        )
        followUpHint.font = .systemFont(ofSize: 11)
        followUpHint.textColor = .secondaryLabelColor

        // Приватность важна: человек должен видеть, что микрофон не отправляет
        // звук в сеть, даже когда ожидание включено постоянно.
        let wakeHint = NSTextField(
            wrappingLabelWithString: "распознавание локальное, звук никуда не уходит"
        )
        wakeHint.font = .systemFont(ofSize: 11)
        wakeHint.textColor = .secondaryLabelColor

        let hint = NSTextField(
            wrappingLabelWithString: "Модель и голос берутся у сервера. "
                + "Если список моделей пуст — сервер выключен, но адрес и голос "
                + "всё равно сохранятся."
        )
        hint.font = .systemFont(ofSize: 11)
        hint.textColor = .secondaryLabelColor

        statusLabel = NSTextField(labelWithString: "")
        statusLabel.font = .systemFont(ofSize: 11)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.lineBreakMode = .byTruncatingTail

        let saveButton = NSButton(title: "Сохранить", target: self, action: #selector(save))
        saveButton.bezelStyle = .rounded
        saveButton.keyEquivalent = "\r"
        let checkButton = NSButton(title: "Проверить", target: self, action: #selector(check))
        checkButton.bezelStyle = .rounded
        let closeButton = NSButton(title: "Закрыть", target: self, action: #selector(close))
        closeButton.bezelStyle = .rounded

        let spacer = NSView(frame: .zero)
        spacer.setContentHuggingPriority(.defaultLow, for: .horizontal)
        let buttons = NSStackView(views: [saveButton, checkButton, spacer, closeButton])
        buttons.orientation = .horizontal
        buttons.spacing = 8
        buttons.distribution = .fill

        let rows = NSStackView(views: [
            row("Сервер", serverField),
            row("Модель", modelRow),
            row("Голос", voicePopup),
            row("", speakCheck),
            row("", animateCheck),
            row("", startCompactCheck),
            row("", startCompactHint),
            row("", wakeCheck),
            row("", wakeHint),
            row("", followUpCheck),
            row("", followUpHint),
            row("Пауза", followUpRow),
            hint,
            statusLabel,
            buttons,
        ])
        rows.orientation = .vertical
        rows.alignment = .width
        rows.spacing = 10
        rows.translatesAutoresizingMaskIntoConstraints = false
        let content = window.contentView!
        content.addSubview(rows)
        NSLayoutConstraint.activate([
            rows.topAnchor.constraint(equalTo: content.topAnchor, constant: 16),
            rows.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 16),
            rows.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -16),
            rows.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -16),
            serverField.widthAnchor.constraint(greaterThanOrEqualToConstant: 280),
            modelPopup.widthAnchor.constraint(greaterThanOrEqualToConstant: 220),
            voicePopup.widthAnchor.constraint(greaterThanOrEqualToConstant: 220),
        ])
        reloadControls()
    }

    private func row(_ title: String, _ view: NSView) -> NSStackView {
        let label = NSTextField(labelWithString: title)
        label.alignment = .right
        label.textColor = .secondaryLabelColor
        label.font = .systemFont(ofSize: 12)
        label.widthAnchor.constraint(equalToConstant: 72).isActive = true
        let row = NSStackView(views: [label, view])
        row.orientation = .horizontal
        row.spacing = 10
        row.alignment = .firstBaseline
        row.distribution = .fill
        return row
    }

    // MARK: - Значения

    private func reloadControls() {
        guard serverField != nil else { return }
        serverField.stringValue = Settings.shared.serverText
        speakCheck.state = Settings.shared.speakReplies ? .on : .off
        animateCheck.state = Settings.shared.animateCircle ? .on : .off
        startCompactCheck.state = Settings.shared.startCompact ? .on : .off
        wakeCheck.state = Settings.shared.wakeWord ? .on : .off
        followUpCheck.state = Settings.shared.followUpListen ? .on : .off
        // Пауза может быть выставлена и вручную (`defaults write`) — тогда
        // показываем ближайший вариант, а хранимое значение не переписываем.
        let pause = Settings.shared.followUpSilence
        if let index = Settings.followUpSilenceOptions.firstIndex(of: pause) {
            followUpPopup.selectItem(at: index)
        } else {
            let fallback = Settings.followUpSilenceOptions
                .firstIndex(of: Settings.defaultFollowUpSilence) ?? 1
            followUpPopup.selectItem(at: fallback)
        }
        let voiceID = Settings.shared.voice
        if let index = VoicePresets.all.firstIndex(where: { $0.id == voiceID }) {
            voicePopup.selectItem(at: index)
        }
        fillModelPopup(current: Settings.shared.model)
    }

    private func fillModelPopup(current: ModelSpec) {
        if !models.contains(current) {
            models.insert(current, at: 0)
        }
        modelPopup.removeAllItems()
        modelPopup.addItems(withTitles: models.map(\.displayName))
        if let index = models.firstIndex(of: current) {
            modelPopup.selectItem(at: index)
        }
    }

    private var selectedModel: ModelSpec? {
        guard let index = modelPopup?.indexOfSelectedItem, models.indices.contains(index) else {
            return nil
        }
        return models[index]
    }

    // MARK: - Сервер

    /// `GET /api/backends` → плоский список моделей. Сеть — в фоне.
    @objc private func loadModels() {
        guard !loadingModels, modelPopup != nil else { return }
        guard let base = Settings.normalize(serverField.stringValue) else {
            statusLabel.stringValue = "адрес сервера непонятен"
            statusLabel.textColor = .systemRed
            return
        }
        loadingModels = true
        statusLabel.stringValue = "спрашиваю список моделей…"
        statusLabel.textColor = .secondaryLabelColor
        Task { [weak self] in
            let result = await SettingsWindowController.fetchModels(base: base)
            guard let self else { return }
            self.loadingModels = false
            switch result {
            case .success(let list):
                self.models = list
                self.fillModelPopup(current: Settings.shared.model)
                self.statusLabel.stringValue = "моделей: \(list.count)"
                self.statusLabel.textColor = .secondaryLabelColor
            case .failure(let error):
                self.models = [Settings.shared.model]
                self.fillModelPopup(current: Settings.shared.model)
                self.statusLabel.stringValue = "сервер не ответил: \(error.message)"
                self.statusLabel.textColor = .systemRed
            }
        }
    }

    private struct ModelListError: Error, Sendable {
        let message: String
        init(_ message: String) { self.message = message }
    }

    private nonisolated static func fetchModels(
        base: URL
    ) async -> Result<[ModelSpec], ModelListError> {
        var request = URLRequest(url: base.appendingPathComponent("api/backends"))
        request.timeoutInterval = 8
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                return .failure(ModelListError("HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)"))
            }
            guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let backends = object["backends"] as? [[String: Any]]
            else {
                return .failure(ModelListError("непонятный ответ"))
            }
            var specs: [ModelSpec] = []
            for backend in backends {
                guard let id = backend["id"] as? String else { continue }
                let models = backend["models"] as? [[String: Any]] ?? []
                for model in models {
                    guard let modelID = model["id"] as? String else { continue }
                    specs.append(ModelSpec(backend: id, model: modelID))
                }
            }
            return .success(specs)
        } catch {
            return .failure(ModelListError(error.localizedDescription))
        }
    }

    // MARK: - Действия

    @objc private func save() {
        guard let base = Settings.normalize(serverField.stringValue) else {
            statusLabel.stringValue = "адрес сервера непонятен: нужен http://хост:порт"
            statusLabel.textColor = .systemRed
            return
        }
        Settings.shared.serverText = base.absoluteString
        serverField.stringValue = base.absoluteString
        if let model = selectedModel {
            Settings.shared.model = model
        }
        let voiceIndex = voicePopup.indexOfSelectedItem
        if VoicePresets.all.indices.contains(voiceIndex) {
            Settings.shared.voice = VoicePresets.all[voiceIndex].id
        }
        Settings.shared.speakReplies = speakCheck.state == .on
        Settings.shared.animateCircle = animateCheck.state == .on
        Settings.shared.startCompact = startCompactCheck.state == .on
        Settings.shared.wakeWord = wakeCheck.state == .on
        Settings.shared.followUpListen = followUpCheck.state == .on
        let pauseIndex = followUpPopup.indexOfSelectedItem
        if Settings.followUpSilenceOptions.indices.contains(pauseIndex) {
            Settings.shared.followUpSilence = Settings.followUpSilenceOptions[pauseIndex]
        }
        statusLabel.stringValue = "сохранено"
        statusLabel.textColor = .secondaryLabelColor
        log("настройки: сервер=\(base.absoluteString) модель=\(Settings.shared.model.storageValue) "
            + "голос=\(Settings.shared.voice) озвучка=\(Settings.shared.speakReplies) "
            + "анимация=\(Settings.shared.animateCircle) "
            + "старт-свёрнутым=\(Settings.shared.startCompact) "
            + "ожидание=\(Settings.shared.wakeWord) "
            + "follow-up=\(Settings.shared.followUpListen)/\(Settings.shared.followUpSilence) с")
    }

    @objc private func check() {
        guard let base = Settings.normalize(serverField.stringValue) else {
            statusLabel.stringValue = "адрес сервера непонятен"
            statusLabel.textColor = .systemRed
            return
        }
        statusLabel.stringValue = "проверяю…"
        statusLabel.textColor = .secondaryLabelColor
        Task { [weak self] in
            var request = URLRequest(url: base.appendingPathComponent("health"))
            request.timeoutInterval = 5
            do {
                let (data, response) = try await URLSession.shared.data(for: request)
                let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                let body = String(decoding: data.prefix(200), as: UTF8.self)
                self?.statusLabel.stringValue = "HTTP \(code): \(body)"
                self?.statusLabel.textColor = code == 200 ? .systemGreen : .systemRed
            } catch {
                self?.statusLabel.stringValue = "нет связи: \(error.localizedDescription)"
                self?.statusLabel.textColor = .systemRed
            }
        }
    }

    @objc private func close() {
        window?.orderOut(nil)
    }

    @objc private func modelChanged() {
        if let model = selectedModel {
            Settings.shared.model = model
        }
    }

    @objc private func voiceChanged() {
        let index = voicePopup.indexOfSelectedItem
        if VoicePresets.all.indices.contains(index) {
            Settings.shared.voice = VoicePresets.all[index].id
        }
    }

    @objc private func speakChanged() {
        Settings.shared.speakReplies = speakCheck.state == .on
    }

    /// Галочка применяется сразу: круг в главном окне перестаёт анимироваться
    /// без перезапуска (главное окно слушает `Settings.didChange`).
    @objc private func animateChanged() {
        Settings.shared.animateCircle = animateCheck.state == .on
        log("настройки: анимация круга=\(Settings.shared.animateCircle ? "включена" : "выключена")")
    }

    /// «Запускаться свёрнутым» (задача 0188): действует при следующем запуске,
    /// текущий вид не меняем — иначе окно закрывалось бы под курсором.
    @objc private func startCompactChanged() {
        Settings.shared.startCompact = startCompactCheck.state == .on
        log("настройки: запускаться свёрнутым="
            + "\(Settings.shared.startCompact ? "включено" : "выключено")")
    }

    /// Ожидание «Привет, Ева» применяется сразу: `AppDelegate` слушает
    /// `Settings.didChange` и поднимает/опускает слушатель без перезапуска.
    @objc private func wakeChanged() {
        Settings.shared.wakeWord = wakeCheck.state == .on
        log("настройки: ожидание „Привет, Ева“=\(Settings.shared.wakeWord ? "включено" : "выключено")")
    }

    /// «Слушать после ответа» (задача 0190) применяется сразу: следующий
    /// ответ уже откроет микрофон сам (или не откроет, если сняли галочку).
    @objc private func followUpChanged() {
        Settings.shared.followUpListen = followUpCheck.state == .on
        log("настройки: слушать после ответа="
            + "\(Settings.shared.followUpListen ? "включено" : "выключено")")
    }

    @objc private func followUpSilenceChanged() {
        let index = followUpPopup.indexOfSelectedItem
        guard Settings.followUpSilenceOptions.indices.contains(index) else { return }
        Settings.shared.followUpSilence = Settings.followUpSilenceOptions[index]
        log("настройки: пауза follow-up=\(Settings.shared.followUpSilence) с")
    }
}
