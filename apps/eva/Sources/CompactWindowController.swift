import AppKit
import Foundation

/// Геометрия компактного окна — без AppKit-состояния, чтобы её можно было
/// проверить в `--selftest-compact` без экрана и без окна.
enum CompactLayout {
    /// Сторона окна. Требование задачи — 120–160 pt; 140 — середина.
    static let size: CGFloat = 140
    /// Отступ от края видимой области для позиции по умолчанию.
    static let inset: CGFloat = 24

    /// Правый нижний угол видимой области (`visibleFrame` уже без строки меню
    /// и без Dock).
    static func defaultOrigin(in visible: NSRect, size: CGFloat = CompactLayout.size) -> NSPoint {
        NSPoint(x: visible.maxX - size - inset, y: visible.minY + inset)
    }

    /// Зажать окно целиком в видимую область. Если окно больше области,
    /// прижимаем к её началу — так хотя бы левый нижний угол остаётся виден.
    static func clampedOrigin(_ origin: NSPoint, size: CGFloat, in visible: NSRect) -> NSPoint {
        let minX = visible.minX
        let maxX = max(minX, visible.maxX - size)
        let minY = visible.minY
        let maxY = max(minY, visible.maxY - size)
        return NSPoint(
            x: min(max(origin.x, minX), maxX),
            y: min(max(origin.y, minY), maxY)
        )
    }

    /// Видимая область экрана, на котором окно провело больше всего места.
    /// Нет пересечений (монитор отключили) — берём основной экран: окно не
    /// должно остаться за краем несуществующего дисплея.
    static func visibleRect(
        for frame: NSRect,
        among visibleFrames: [NSRect],
        fallback: NSRect
    ) -> NSRect {
        var best: NSRect?
        var bestArea: CGFloat = 0
        for candidate in visibleFrames {
            let intersection = candidate.intersection(frame)
            let area = intersection.isNull ? 0 : intersection.width * intersection.height
            if area > bestArea {
                bestArea = area
                best = candidate
            }
        }
        return best ?? fallback
    }

    /// Зажатие с выбором экрана: используется и при показе, и при перетаскивании,
    /// и при смене монитора.
    static func clampedOrigin(
        _ origin: NSPoint,
        size: CGFloat,
        visibleFrames: [NSRect],
        fallback: NSRect
    ) -> NSPoint {
        let frame = NSRect(origin: origin, size: NSSize(width: size, height: size))
        let visible = visibleRect(for: frame, among: visibleFrames, fallback: fallback)
        return clampedOrigin(origin, size: size, in: visible)
    }

    /// Для журнала: «x,y 140×140».
    static func description(_ frame: NSRect) -> String {
        let x = Int(frame.origin.x.rounded())
        let y = Int(frame.origin.y.rounded())
        let width = Int(frame.width.rounded())
        let height = Int(frame.height.rounded())
        return "\(x),\(y) \(width)×\(height)"
    }
}

/// Компактный режим (задача 0188): на экране только круг Евы — безрамочное
/// прозрачное окно поверх всех окон.
///
/// Почему NSPanel, а не NSWindow: `nonactivatingPanel` не забирает фокус у
/// активного приложения. Уровень `floating` держит круг поверх обычных окон,
/// `orderFrontRegardless()` показывает его, не активируя Еву, а `LSUIElement`
/// (Info.plist) не даёт иконке появиться в Dock.
///
/// Окно не больше 160 pt, поэтому каждая деталь поведения — своя:
/// * перетаскивание — вручную по `mouseDragged` (клик, двойной клик и правый
///   клик должны остаться кликами, поэтому `isMovableByWindowBackground`
///   выключен);
/// * одиночный клик ждёт `doubleClickInterval`, чтобы двойной клик не успел
///   развернуть окно;
/// * позиция — в `UserDefaults` (`eva.compactX/Y`), при показе, перетаскивании
///   и смене экрана зажимается в `visibleFrame`.
@MainActor
final class CompactWindowController: NSObject, NSWindowDelegate {
    /// Развернуть главное окно (одиночный клик и пункт меню).
    var onExpand: (() -> Void)?
    /// Начать/закончить слушать (двойной клик, как ⌥Space).
    var onToggleListening: (() -> Void)?
    var onSettings: (() -> Void)?
    var onQuit: (() -> Void)?

    private let circles: CircleHub
    private let defaults: UserDefaults
    private var panel: CirclePanel?
    /// Отложенный разворот по одиночному клику. Именно `Timer` на главном
    /// run loop, а не `Task`: таймер живёт в режиме `.common` и не зависит от
    /// того, чем занят Swift Concurrency, — а двойной клик должен успеть
    /// отменить разворот, даже если между кликами AppKit показывает меню.
    private var clickTimer: Timer?

    // Перетаскивание: где мышь и где окно были в момент нажатия.
    private var dragStartMouse: NSPoint?
    private var dragStartOrigin: NSPoint?

    init(circles: CircleHub, defaults: UserDefaults = .standard) {
        self.circles = circles
        self.defaults = defaults
        super.init()
        // Смена монитора/разрешения: окно могло оказаться за краем.
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(screenParametersChanged),
            name: NSApplication.didChangeScreenParametersNotification,
            object: nil
        )
    }

    var isVisible: Bool { panel?.isVisible == true }

    /// Текущая рамка — для журнала и самопроверки.
    var frame: NSRect? { panel?.frame }

    var windowLevel: NSWindow.Level? { panel?.level }

    var styleMask: NSWindow.StyleMask? { panel?.styleMask }

    var isOpaque: Bool? { panel?.isOpaque }

    var backgroundColor: NSColor? { panel?.backgroundColor }

    // MARK: - Показ

    func show() {
        if panel == nil { build() }
        guard let panel else { return }
        let origin = restoredOrigin()
        panel.setFrameOrigin(origin)
        // orderFrontRegardless, а не makeKeyAndOrderFront: фокус не забираем.
        panel.orderFrontRegardless()
        log("компакт: круг показан — \(CompactLayout.description(panel.frame)) "
            + "level=\(panel.level.rawValue) (floating=\(NSWindow.Level.floating.rawValue))")
    }

    func hide() {
        guard let panel else { return }
        savePosition()
        if panel.isVisible {
            log("компакт: круг скрыт — \(CompactLayout.description(panel.frame))")
        }
        panel.orderOut(nil)
        cancelPendingClick()
    }

    /// Зажать окно в видимую область текущего экрана. Вызывается при смене
    /// монитора, разрешения и в самопроверке.
    func clampIntoVisible() {
        guard let panel else { return }
        let origin = CompactLayout.clampedOrigin(
            panel.frame.origin,
            size: panel.frame.width,
            visibleFrames: visibleFrames,
            fallback: fallbackVisible
        )
        if origin != panel.frame.origin {
            log("компакт: зажал в видимую область — было "
                + "\(CompactLayout.description(panel.frame)), стало "
                + "\(CompactLayout.description(NSRect(origin: origin, size: panel.frame.size)))")
            panel.setFrameOrigin(origin)
        }
    }

    // MARK: - Сборка

    private func build() {
        let size = CompactLayout.size
        let panel = CirclePanel(
            contentRect: NSRect(x: 0, y: 0, width: size, height: size),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.title = "Ева — круг"
        panel.level = .floating
        panel.isFloatingPanel = true
        panel.becomesKeyOnlyIfNeeded = true
        panel.hidesOnDeactivate = false
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.isReleasedWhenClosed = false
        panel.isMovableByWindowBackground = false
        panel.animationBehavior = .none
        // Круг виден на всех рабочих столах и поверх полноэкранных приложений.
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.delegate = self
        self.panel = panel

        let host = CompactCircleHost(
            frame: NSRect(x: 0, y: 0, width: size, height: size)
        )
        host.controller = self
        let circle = CircleView(frame: host.bounds)
        host.addSubview(circle)

        panel.contentView = host
        NSLayoutConstraint.activate([
            circle.leadingAnchor.constraint(equalTo: host.leadingAnchor),
            circle.trailingAnchor.constraint(equalTo: host.trailingAnchor),
            circle.topAnchor.constraint(equalTo: host.topAnchor),
            circle.bottomAnchor.constraint(equalTo: host.bottomAnchor),
        ])

        // Тот же класс, что и большой круг: состояние, уровень и ошибки
        // приходят из общего хаба.
        circles.register(circle)
    }

    /// Для журнала и самопроверки: подвинуть окно так же, как перетаскивание
    /// (через `windowDidMove` позиция сохранится).
    func setFrameOriginForTest(_ origin: NSPoint) {
        panel?.setFrameOrigin(origin)
    }

    // MARK: - Позиция

    private func restoredOrigin() -> NSPoint {
        let visible = fallbackVisible
        let size = CompactLayout.size
        if let x = defaults.object(forKey: Settings.compactXKey) as? Double,
           let y = defaults.object(forKey: Settings.compactYKey) as? Double {
            return CompactLayout.clampedOrigin(
                NSPoint(x: x, y: y),
                size: size,
                visibleFrames: visibleFrames,
                fallback: visible
            )
        }
        return CompactLayout.defaultOrigin(in: visible, size: size)
    }

    private func savePosition() {
        guard let panel else { return }
        defaults.set(Double(panel.frame.origin.x), forKey: Settings.compactXKey)
        defaults.set(Double(panel.frame.origin.y), forKey: Settings.compactYKey)
    }

    private var visibleFrames: [NSRect] {
        NSScreen.screens.map(\.visibleFrame)
    }

    private var fallbackVisible: NSRect {
        NSScreen.main?.visibleFrame
            ?? NSScreen.screens.first?.visibleFrame
            ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
    }

    // MARK: - Мышь

    fileprivate func mouseDown(_ event: NSEvent) {
        dragStartMouse = NSEvent.mouseLocation
        dragStartOrigin = panel?.frame.origin
    }

    fileprivate func mouseDragged(_ event: NSEvent) {
        guard let panel, let startMouse = dragStartMouse, let startOrigin = dragStartOrigin
        else { return }
        let now = NSEvent.mouseLocation
        guard abs(now.x - startMouse.x) > 2 || abs(now.y - startMouse.y) > 2 else { return }
        let target = NSPoint(
            x: startOrigin.x + (now.x - startMouse.x),
            y: startOrigin.y + (now.y - startMouse.y)
        )
        let origin = CompactLayout.clampedOrigin(
            target,
            size: panel.frame.width,
            visibleFrames: visibleFrames,
            fallback: fallbackVisible
        )
        panel.setFrameOrigin(origin)
    }

    fileprivate func mouseUp(_ event: NSEvent) {
        let startMouse = dragStartMouse
        let startOrigin = dragStartOrigin
        dragStartMouse = nil
        dragStartOrigin = nil

        let moved: CGFloat
        if let startMouse, let startOrigin {
            let now = NSEvent.mouseLocation
            let distance = hypot(now.x - startMouse.x, now.y - startMouse.y)
            let sameOrigin = panel?.frame.origin == startOrigin
            moved = sameOrigin ? distance : .greatestFiniteMagnitude
        } else {
            moved = 0
        }
        if moved > 2 {
            // Это перетаскивание, а не клик.
            savePosition()
            return
        }

        if event.clickCount >= 2 {
            // Второй клик двойного: отменяем отложенный разворот.
            cancelPendingClick()
            log("компакт: двойной клик — слушать/остановить")
            onToggleListening?()
            return
        }

        cancelPendingClick()
        let timer = Timer(
            timeInterval: NSEvent.doubleClickInterval,
            target: self,
            selector: #selector(clickTimerFired),
            userInfo: nil,
            repeats: false
        )
        timer.tolerance = 0
        RunLoop.main.add(timer, forMode: .common)
        clickTimer = timer
    }

    private func cancelPendingClick() {
        clickTimer?.invalidate()
        clickTimer = nil
    }

    /// Одиночный клик подтвердился: двойного не было — разворачиваем окно.
    @objc private func clickTimerFired() {
        clickTimer = nil
        log("компакт: клик — развернуть")
        onExpand?()
    }

    /// Контекстное меню круга. Вынесено из `rightMouseDown`, чтобы состав
    /// проверялся без живой мыши (`--selftest-compact`).
    func makeMenu() -> NSMenu {
        let menu = NSMenu()
        let expand = NSMenuItem(
            title: "Развернуть",
            action: #selector(menuExpand),
            keyEquivalent: ""
        )
        expand.target = self
        menu.addItem(expand)
        let settings = NSMenuItem(
            title: "Настройки…",
            action: #selector(menuSettings),
            keyEquivalent: ""
        )
        settings.target = self
        menu.addItem(settings)
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Выход", action: #selector(menuQuit), keyEquivalent: "")
        quit.target = self
        menu.addItem(quit)
        return menu
    }

    /// Самопроверка кликов: прогоняет тот же путь, что и настоящая мышь
    /// (`mouseDown` + `mouseUp`), но без перемещения курсора.
    func simulateClickForTest(clickCount: Int) {
        let now = ProcessInfo.processInfo.systemUptime
        guard let event = NSEvent.mouseEvent(
            with: .leftMouseUp,
            location: .zero,
            modifierFlags: [],
            timestamp: now,
            windowNumber: panel?.windowNumber ?? 0,
            context: nil,
            eventNumber: 0,
            clickCount: clickCount,
            pressure: 1
        ) else { return }
        mouseDown(event)
        mouseUp(event)
    }

    fileprivate func rightMouseDown(_ event: NSEvent) {
        log("компакт: правое меню у \(CompactLayout.description(panel?.frame ?? .zero))")
        if let content = panel?.contentView {
            _ = makeMenu().popUp(positioning: nil, at: event.locationInWindow, in: content)
        }
    }

    @objc private func menuExpand() {
        onExpand?()
    }

    @objc private func menuSettings() {
        onSettings?()
    }

    @objc private func menuQuit() {
        onQuit?()
    }

    // MARK: - NSWindowDelegate

    func windowDidMove(_ notification: Notification) {
        savePosition()
    }

    func windowDidChangeScreen(_ notification: Notification) {
        clampIntoVisible()
    }

    @objc private func screenParametersChanged() {
        guard isVisible else { return }
        log("компакт: сменился экран/разрешение — зажимаю в видимую область")
        clampIntoVisible()
    }
}

/// Безрамочная панель, которая не может стать ключевой: клик по кругу не
/// уводит фокус из приложения, с которым человек работает.
@MainActor
private final class CirclePanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

/// Прозрачный контейнер: круг рисуется в `CircleView`, а мышь (перетаскивание,
/// клики, правое меню) обрабатывает окно.
@MainActor
private final class CompactCircleHost: NSView {
    weak var controller: CompactWindowController?

    /// Первый клик по неактивному окну должен доходить до нас, а не только
    /// активировать приложение (мы всё равно не активируем его — панель
    /// nonactivating).
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override var acceptsFirstResponder: Bool { false }

    /// События ловим сами: дочерний `CircleView` — только картинка.
    override func hitTest(_ point: NSPoint) -> NSView? {
        let local = superview.map { convert(point, from: $0) } ?? point
        return bounds.contains(local) ? self : nil
    }

    override func mouseDown(with event: NSEvent) {
        controller?.mouseDown(event)
    }

    override func mouseDragged(with event: NSEvent) {
        controller?.mouseDragged(event)
    }

    override func mouseUp(with event: NSEvent) {
        controller?.mouseUp(event)
    }

    override func rightMouseDown(with event: NSEvent) {
        controller?.rightMouseDown(event)
    }
}
