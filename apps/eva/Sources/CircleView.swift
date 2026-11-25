import AppKit
import Foundation

/// Состояние большого круга Евы. Одно и то же значение видят окно и журнал:
/// `idle` → `listening` → `thinking` → `speaking` → `idle`, плюс `error`.
enum EvaCircleState: String, CaseIterable, Sendable {
    case idle
    case listening
    case thinking
    case speaking
    case error

    /// Подпись под кругом: короткая, чтобы не спорить с текстовым статусом
    /// в шапке окна (там остаются детали — модель, секунды, текст ошибки).
    var caption: String {
        switch self {
        case .idle: return "готова"
        case .listening: return "слушаю"
        case .thinking: return "думаю"
        case .speaking: return "говорю"
        case .error: return "ошибка"
        }
    }
}

/// Большой круг состояния: рисуется обычным Core Graphics в `draw(_:)`,
/// анимация — таймер 30 Гц (не CVDisplayLink и не Metal: для 580-пиксельного
/// окна этого хватает, а в покое CPU остаётся заметно ниже 2 %).
///
/// Почему так:
/// * `Timer` на главном run loop — единственное место, где трогается AppKit;
/// * уровни приходят из аудио-потока через неизолированную обёртку и только
///   кладутся в `targetLevel` — сам круг сглаживает их при отрисовке;
/// * без галочки «Анимация» таймер не запускается вовсе, круг статичен.
@MainActor
final class CircleView: NSView {
    /// Текущее состояние. Смена цвета — плавная, 0.25 с.
    var visualState: EvaCircleState = .idle {
        didSet {
            guard visualState != oldValue else { return }
            fromColor = currentColor
            toColor = CircleView.colorComponents(for: visualState)
            colorProgress = animationEnabled ? 0 : 1
            if visualState == .error {
                errorFlash = 1
            } else if oldValue == .error {
                errorFlash = 0
            }
            setAccessibilityLabel("Состояние Евы: \(captionText)")
            needsDisplay = true
            syncTimer()
        }
    }

    /// Ожидание фразы «Привет, Ева» (задача 0187): в `idle` круг едва заметно
    /// пульсирует, чтобы было видно, что микрофон слушает. На другие состояния
    /// флаг не влияет — они рисуются как раньше.
    var wakeWaiting = false {
        didSet {
            guard wakeWaiting != oldValue else { return }
            setAccessibilityLabel("Состояние Евы: \(captionText)")
            needsDisplay = true
            syncTimer()
        }
    }

    /// Галочка «Анимация»: выключена — ни таймера, ни волн, только цвет и
    /// подпись состояния (слабые машины).
    var animationEnabled = true {
        didSet {
            guard animationEnabled != oldValue else { return }
            if !animationEnabled {
                stopTimer()
                smoothLevel = 0
                targetLevel = 0
                errorFlash = 0
            }
            needsDisplay = true
            syncTimer()
        }
    }

    /// Текст ошибки под кругом (полный — в статусе шапки, здесь одна строка).
    private var errorText: String?

    private let stateChangeDuration: Double = 0.25
    private let tickInterval: TimeInterval = 1.0 / 30.0

    // Сглаженный уровень 0…1 и цель, которую выставляет аудио.
    private var targetLevel: CGFloat = 0
    private var smoothLevel: CGFloat = 0
    private var phase: Double = 0
    private var lastTick: TimeInterval = 0
    private var timer: Timer?

    // Переход цвета: от предыдущего к новому за 0.25 с.
    private var fromColor: RGBA = CircleView.colorComponents(for: .idle)
    private var toColor: RGBA = CircleView.colorComponents(for: .idle)
    private var colorProgress: CGFloat = 1

    // Красная вспышка ошибки: 1 → 0 примерно за 0.9 с.
    private var errorFlash: CGFloat = 0

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        translatesAutoresizingMaskIntoConstraints = false
        setAccessibilityElement(true)
        setAccessibilityRole(.image)
        setAccessibilityLabel("Состояние Евы: \(captionText)")
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) не используется: вёрстка кодовая")
    }

    // MARK: - Данные извне

    /// Уровень 0…1: RMS микрофона или громкость `AVAudioPlayer`.
    func setLevel(_ value: CGFloat) {
        targetLevel = min(max(value, 0), 1)
    }

    /// Короткая красная вспышка и текст ошибки под кругом. Само состояние
    /// возвращает в `idle` AppDelegate через `visualState`.
    func showError(_ message: String) {
        let line = message
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        errorText = line.isEmpty ? nil : line
        visualState = .error
        errorFlash = 1
        needsDisplay = true
    }

    /// Остановить таймер при закрытии приложения/окна, не дожидаясь deinit.
    func stopAnimation() {
        stopTimer()
    }

    // MARK: - Таймер

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        lastTick = ProcessInfo.processInfo.systemUptime
        syncTimer()
    }

    private func syncTimer() {
        guard animationEnabled, window != nil else {
            stopTimer()
            return
        }
        startTimer()
    }

    private func startTimer() {
        guard timer == nil else { return }
        let timer = Timer(
            timeInterval: tickInterval,
            target: self,
            selector: #selector(tick),
            userInfo: nil,
            repeats: true
        )
        // Допуск даёт системе право сдвигать тик: 30 Гц достаточно для глаза,
        // а лишние пробуждения на слабой машине ни к чему.
        timer.tolerance = tickInterval / 5
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
        lastTick = ProcessInfo.processInfo.systemUptime
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    @objc private func tick() {
        let now = ProcessInfo.processInfo.systemUptime
        let dt = min(max(now - lastTick, 0), 0.1)
        lastTick = now

        // Окно спрятано — не тратим кадры. Таймер остаётся, но тик дешёвый.
        guard window?.isVisible == true else { return }

        phase += dt
        if colorProgress < 1 {
            colorProgress = min(1, colorProgress + CGFloat(dt / stateChangeDuration))
        }

        // Атака быстрее отпускания: голос виден сразу (<100 мс), но круг
        // не дёргается на каждом буфере.
        let tau: CGFloat = targetLevel > smoothLevel ? 0.035 : 0.12
        let alpha = 1 - exp(-CGFloat(dt) / tau)
        smoothLevel += (targetLevel - smoothLevel) * alpha

        if visualState == .error {
            errorFlash = max(0, errorFlash - CGFloat(dt / 0.9))
        } else if errorFlash > 0 {
            errorFlash = max(0, errorFlash - CGFloat(dt / 0.6))
        }

        needsDisplay = true
    }

    // MARK: - Отрисовка

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let bounds = self.bounds
        guard bounds.width > 48, bounds.height > 48 else { return }

        let captionHeight: CGFloat = 15
        let topPad: CGFloat = 4
        let bottomPad: CGFloat = 2
        // Самый дальний элемент (волна слушания) — 1.2 радиуса. Резервируем
        // место сверху и снизу, чтобы волны не обрезались краем вью.
        let maxWave: CGFloat = 1.2
        let vertical = (bounds.height - captionHeight - topPad - bottomPad) / (maxWave + 1)
        let diameter = min(bounds.width - 24, vertical * 2)
        guard diameter > 24 else { return }
        let radius = diameter / 2
        let center = CGPoint(x: bounds.midX, y: bounds.maxY - topPad - maxWave * radius)

        let base = currentColor
        let color = base.nsColor
        let level = animationEnabled
            ? ((visualState == .listening || visualState == .speaking) ? smoothLevel : 0)
            : 0
        // Дыхание: медленная синусоида с периодом ~4.5 с.
        let breath = CGFloat(0.5 + 0.5 * sin(phase * 2 * .pi / 4.5))

        ctx.saveGState()
        ctx.setShouldAntialias(true)
        ctx.setLineCap(.round)

        drawHalo(ctx, center: center, radius: radius, color: color, level: level, breath: breath)

        switch visualState {
        case .listening:
            drawListeningRings(ctx, center: center, radius: radius, color: color, level: level)
        case .speaking:
            drawSpeakingRings(ctx, center: center, radius: radius, color: color, level: level)
        case .thinking:
            drawThinkingPulse(ctx, center: center, radius: radius, color: color)
        case .idle:
            if wakeWaiting {
                drawWakePulse(ctx, center: center, radius: radius, color: color, breath: breath)
            }
        case .error:
            break
        }

        let scale: CGFloat
        switch visualState {
        case .idle:
            scale = 0.94 + 0.06 * breath
        case .listening:
            scale = 0.92 + 0.12 * level
        case .thinking:
            scale = 0.955 + 0.02 * CGFloat(sin(phase * 3.2))
        case .speaking:
            scale = 0.92 + 0.11 * level
        case .error:
            scale = 1.0
        }
        let discRadius = radius * scale
        drawDisc(ctx, center: center, radius: discRadius, color: color)

        switch visualState {
        case .thinking:
            drawThinkingArcs(ctx, center: center, radius: radius, color: color)
        case .listening:
            drawListeningBars(ctx, center: center, radius: radius, color: color, level: level)
        case .speaking:
            drawSpeakingCore(ctx, center: center, radius: radius, color: color, level: level)
        case .error:
            drawErrorFlash(ctx, center: center, radius: discRadius, color: color)
        case .idle:
            break
        }

        // Тонкий контур — круг читается и на светлой, и на тёмной теме.
        ctx.setStrokeColor(color.withAlphaComponent(0.55).cgColor)
        ctx.setLineWidth(1)
        ctx.strokeEllipse(in: CircleView.rect(center: center, radius: discRadius))

        ctx.restoreGState()

        drawCaption(in: bounds)
    }

    private func drawHalo(
        _ ctx: CGContext,
        center: CGPoint,
        radius: CGFloat,
        color: NSColor,
        level: CGFloat,
        breath: CGFloat
    ) {
        let breathe = visualState == .idle ? (0.55 + 0.45 * breath) : 1
        for index in 0..<3 {
            let spread = radius * (1.0 + 0.05 * CGFloat(index + 1))
            let alpha = (0.16 - CGFloat(index) * 0.04) * breathe
            ctx.setStrokeColor(color.withAlphaComponent(alpha).cgColor)
            ctx.setLineWidth(1)
            ctx.strokeEllipse(in: CircleView.rect(center: center, radius: spread))
        }
    }

    private func drawDisc(
        _ ctx: CGContext,
        center: CGPoint,
        radius: CGFloat,
        color: NSColor
    ) {
        let rect = CircleView.rect(center: center, radius: radius)
        ctx.saveGState()
        ctx.addEllipse(in: rect)
        ctx.clip()

        let top = color.blended(withFraction: 0.24, of: .white) ?? color
        let bottom = color.blended(withFraction: 0.28, of: .black) ?? color
        let colors = [top.withAlphaComponent(0.96).cgColor, bottom.withAlphaComponent(0.82).cgColor]
        if let gradient = CGGradient(
            colorsSpace: CGColorSpaceCreateDeviceRGB(),
            colors: colors as CFArray,
            locations: [0, 1]
        ) {
            ctx.drawLinearGradient(
                gradient,
                start: CGPoint(x: center.x, y: center.y + radius),
                end: CGPoint(x: center.x, y: center.y - radius),
                options: []
            )
        } else {
            ctx.setFillColor(color.cgColor)
            ctx.fill(rect)
        }

        // Мягкий блик сверху — круг выглядит объёмным без блюра и Metal.
        ctx.setFillColor(NSColor.white.withAlphaComponent(0.10).cgColor)
        ctx.fillEllipse(in: CGRect(
            x: center.x - radius * 0.46,
            y: center.y + radius * 0.10,
            width: radius * 0.92,
            height: radius * 0.44
        ))
        ctx.restoreGState()
    }

    private func drawListeningRings(
        _ ctx: CGContext,
        center: CGPoint,
        radius: CGFloat,
        color: NSColor,
        level: CGFloat
    ) {
        // Три расходящиеся волны; их яркость и толщина — от реального RMS.
        // На тишине (level ≈ 0) волн практически не видно.
        guard level > 0.01 else { return }
        for index in 0..<3 {
            let offset = CGFloat(index) / 3
            let progress = CGFloat((phase * 0.55 + Double(offset)).truncatingRemainder(dividingBy: 1))
            let waveRadius = radius * (0.98 + 0.22 * progress)
            let alpha = (1 - progress) * (0.10 + 0.60 * level)
            ctx.setStrokeColor(color.withAlphaComponent(alpha).cgColor)
            ctx.setLineWidth(1 + 2.4 * level)
            ctx.strokeEllipse(in: CircleView.rect(center: center, radius: waveRadius))
        }
    }

    /// Едва заметное кольцо-дыхание вокруг диска: alpha ≈0,04–0,09, поэтому
    /// оно читается как «микрофон ждёт», но не спорит с состояниями записи.
    private func drawWakePulse(
        _ ctx: CGContext,
        center: CGPoint,
        radius: CGFloat,
        color: NSColor,
        breath: CGFloat
    ) {
        let spread = radius * (1.02 + 0.03 * breath)
        ctx.setStrokeColor(color.withAlphaComponent(0.04 + 0.05 * breath).cgColor)
        ctx.setLineWidth(1)
        ctx.strokeEllipse(in: CircleView.rect(center: center, radius: spread))
    }

    private func drawListeningBars(
        _ ctx: CGContext,
        center: CGPoint,
        radius: CGFloat,
        color: NSColor,
        level: CGFloat
    ) {
        // Частые насечки вокруг круга: волна «дышит» даже на тихой речи.
        guard level > 0.02 else { return }
        let bars = 56
        let inner = radius * 0.99
        let length = radius * (0.03 + 0.16 * level)
        for index in 0..<bars {
            let angle = CGFloat(index) / CGFloat(bars) * 2 * .pi
            let wobble = 0.45 + 0.55 * CGFloat(0.5 + 0.5 * sin(CGFloat(index) * 0.8 + CGFloat(phase) * 7))
            let cosA = cos(angle)
            let sinA = sin(angle)
            let start = CGPoint(x: center.x + cosA * inner, y: center.y + sinA * inner)
            let end = CGPoint(
                x: center.x + cosA * (inner + length * wobble),
                y: center.y + sinA * (inner + length * wobble)
            )
            ctx.setStrokeColor(color.withAlphaComponent(0.25 + 0.55 * level).cgColor)
            ctx.setLineWidth(1.6)
            ctx.move(to: start)
            ctx.addLine(to: end)
            ctx.strokePath()
        }
    }

    private func drawThinkingPulse(
        _ ctx: CGContext,
        center: CGPoint,
        radius: CGFloat,
        color: NSColor
    ) {
        let pulse = CGFloat(0.5 + 0.5 * sin(phase * 3.0))
        ctx.setFillColor(color.withAlphaComponent(0.05 + 0.05 * pulse).cgColor)
        ctx.fillEllipse(in: CircleView.rect(center: center, radius: radius * 1.04))
    }

    private func drawThinkingArcs(
        _ ctx: CGContext,
        center: CGPoint,
        radius: CGFloat,
        color: NSColor
    ) {
        // Две дуги в разные стороны: «шестерёнки думают».
        ctx.setLineCap(.round)
        for (index, direction) in [1.0, -0.7].enumerated() {
            let start = CGFloat(phase * (1.9 * direction)) + CGFloat(index) * 2.1
            let sweep = CGFloat(index == 0 ? 1.45 : 0.85)
            ctx.setStrokeColor(color.withAlphaComponent(index == 0 ? 0.90 : 0.42).cgColor)
            ctx.setLineWidth(index == 0 ? 4.5 : 3)
            ctx.addArc(
                center: center,
                radius: radius * 0.72,
                startAngle: start,
                endAngle: start + sweep,
                clockwise: false
            )
            ctx.strokePath()
        }
        let innerStart = CGFloat(-phase * 2.1) + 0.8
        ctx.setStrokeColor(color.withAlphaComponent(0.55).cgColor)
        ctx.setLineWidth(2)
        ctx.addArc(
            center: center,
            radius: radius * 0.48,
            startAngle: innerStart,
            endAngle: innerStart + 0.9,
            clockwise: false
        )
        ctx.strokePath()
    }

    private func drawSpeakingRings(
        _ ctx: CGContext,
        center: CGPoint,
        radius: CGFloat,
        color: NSColor,
        level: CGFloat
    ) {
        // Две «жидкие» волны: амплитуда — громкость плеера, фаза — время.
        // Волны идут по внешнему краю диска, чтобы громкость была видна,
        // а не пряталась под заливкой.
        for layer in 0..<2 {
            let path = CGMutablePath()
            let points = 96
            let baseRadius = radius * (0.99 + 0.04 * level + CGFloat(layer) * 0.06)
            let amplitude = radius * (0.012 + 0.10 * level) * (layer == 0 ? 1 : 0.8)
            for index in 0...points {
                let angle = CGFloat(index) / CGFloat(points) * 2 * .pi
                let wobble = sin(angle * 5 + CGFloat(phase) * 7 + CGFloat(layer) * 1.5) * 0.65
                    + sin(angle * 3 - CGFloat(phase) * 5) * 0.35
                let r = baseRadius + amplitude * CGFloat(wobble)
                let point = CGPoint(
                    x: center.x + cos(angle) * r,
                    y: center.y + sin(angle) * r
                )
                if index == 0 {
                    path.move(to: point)
                } else {
                    path.addLine(to: point)
                }
            }
            path.closeSubpath()
            if layer == 0 {
                ctx.addPath(path)
                ctx.setFillColor(color.withAlphaComponent(0.12 + 0.14 * level).cgColor)
                ctx.fillPath()
            } else {
                ctx.addPath(path)
                ctx.setStrokeColor(color.withAlphaComponent(0.46 + 0.38 * level).cgColor)
                ctx.setLineWidth(1.6)
                ctx.strokePath()
            }
        }
    }

    private func drawSpeakingCore(
        _ ctx: CGContext,
        center: CGPoint,
        radius: CGFloat,
        color: NSColor,
        level: CGFloat
    ) {
        // Пульсирующее ядро внутри — маленький индикатор «звук идёт».
        let core = radius * (0.22 + 0.10 * level)
        ctx.setFillColor(color.withAlphaComponent(0.12 + 0.18 * level).cgColor)
        ctx.fillEllipse(in: CircleView.rect(center: center, radius: core))
    }

    private func drawErrorFlash(
        _ ctx: CGContext,
        center: CGPoint,
        radius: CGFloat,
        color: NSColor
    ) {
        guard errorFlash > 0.001 else { return }
        let red = NSColor(srgbRed: 0.98, green: 0.20, blue: 0.22, alpha: 1)
        let rect = CircleView.rect(center: center, radius: radius)
        ctx.saveGState()
        ctx.addEllipse(in: rect)
        ctx.clip()
        ctx.setFillColor(red.withAlphaComponent(0.30 + 0.50 * errorFlash).cgColor)
        ctx.fill(rect)
        ctx.restoreGState()

        // Ударная волна: расходится, пока вспышка гаснет.
        let progress = 1 - errorFlash
        let waveRadius = radius * (1.0 + 0.18 * progress)
        ctx.setStrokeColor(red.withAlphaComponent(0.65 * errorFlash).cgColor)
        ctx.setLineWidth(1.5 + 4 * errorFlash)
        ctx.strokeEllipse(in: CircleView.rect(center: center, radius: waveRadius))
    }

    /// Подпись под кругом; в ожидании wake она говорит, чего именно ждём.
    private var captionText: String {
        if visualState == .idle, wakeWaiting { return "жду „Привет, Ева“" }
        return visualState.caption
    }

    private func drawCaption(in bounds: NSRect) {
        let isError = visualState == .error
        let text: String
        if isError {
            text = errorText ?? visualState.caption
        } else {
            text = captionText
        }
        let color: NSColor = isError
            ? NSColor(srgbRed: 0.92, green: 0.24, blue: 0.26, alpha: 1)
            : .secondaryLabelColor
        let paragraph = NSMutableParagraphStyle()
        paragraph.alignment = .center
        paragraph.lineBreakMode = .byTruncatingTail
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 10.5, weight: .medium),
            .foregroundColor: color,
            .paragraphStyle: paragraph,
        ]
        (text as NSString).draw(
            in: NSRect(x: 6, y: 1, width: bounds.width - 12, height: 14),
            withAttributes: attributes
        )
    }

    // MARK: - Цвета

    private var currentColor: RGBA {
        let t = colorProgress * colorProgress * (3 - 2 * colorProgress)
        return fromColor.mixed(with: toColor, t: t)
    }

    private static func colorComponents(for state: EvaCircleState) -> RGBA {
        switch state {
        case .idle:
            return RGBA(r: 0.34, g: 0.55, b: 0.61, a: 1)
        case .listening:
            return RGBA(r: 0.16, g: 0.62, b: 0.92, a: 1)
        case .thinking:
            return RGBA(r: 0.46, g: 0.42, b: 0.95, a: 1)
        case .speaking:
            return RGBA(r: 0.18, g: 0.75, b: 0.56, a: 1)
        case .error:
            return RGBA(r: 0.92, g: 0.24, b: 0.27, a: 1)
        }
    }

    private static func rect(center: CGPoint, radius: CGFloat) -> CGRect {
        CGRect(
            x: center.x - radius,
            y: center.y - radius,
            width: radius * 2,
            height: radius * 2
        )
    }
}

/// Компоненты цвета в sRGB для плавного перехода между состояниями.
private struct RGBA {
    var r: CGFloat
    var g: CGFloat
    var b: CGFloat
    var a: CGFloat

    var nsColor: NSColor {
        NSColor(srgbRed: r, green: g, blue: b, alpha: a)
    }

    func mixed(with other: RGBA, t: CGFloat) -> RGBA {
        let k = min(max(t, 0), 1)
        return RGBA(
            r: r + (other.r - r) * k,
            g: g + (other.g - g) * k,
            b: b + (other.b - b) * k,
            a: a + (other.a - a) * k
        )
    }
}
