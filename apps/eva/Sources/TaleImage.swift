import AppKit
import Foundation

/// Ссылка на иллюстрацию сказки из SSE-события `image` (задача 0224).
///
/// Сервер присылает путь относительно своего адреса
/// (`/media/fairy-tales/kolobok/01.jpg`), поэтому показывать картинку можно
/// только зная base URL. Всё, что не похоже на http(s), отбрасывается: битый
/// URL не должен ломать ответ.
struct TaleImageRef: Sendable, Equatable {
    let url: String
    let alt: String

    init(url: String, alt: String) {
        self.url = url
        self.alt = alt
    }

    /// Абсолютный адрес картинки; `nil` — пустой/чужой/битый URL.
    nonisolated func resolved(base: URL?) -> URL? {
        guard let base else { return nil }
        return TaleImageURL.resolve(url, base: base)
    }
}

/// Сборка абсолютного URL картинки. Вынесено в отдельный `nonisolated` тип,
/// чтобы правило проверялось без окна и без сети (`--selftest-images`).
enum TaleImageURL {
    nonisolated static func resolve(_ raw: String, base: URL) -> URL? {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        guard let url = URL(string: text, relativeTo: base)?.absoluteURL else { return nil }
        guard let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              let host = url.host, !host.isEmpty
        else {
            return nil
        }
        return url
    }
}

/// Кэш скачанных картинок в памяти: одна и та же иллюстрация при повторном
/// вопросе не ходит в сеть. `NSCache` потокобезопасен и сам сбрасывает память
/// под давлением, поэтому класс помечен `@unchecked Sendable`: кроме кэша в
/// нём ничего нет.
final class TaleImageCache: @unchecked Sendable {
    static let shared = TaleImageCache()

    private let cache = NSCache<NSURL, NSData>()

    private init() {
        cache.countLimit = 32
        cache.totalCostLimit = 64 * 1024 * 1024
    }

    func data(for url: URL) -> Data? {
        cache.object(forKey: url as NSURL) as Data?
    }

    func store(_ data: Data, for url: URL) {
        cache.setObject(data as NSData, forKey: url as NSURL, cost: data.count)
    }

    func removeAll() {
        cache.removeAllObjects()
    }
}

/// Загрузка картинки: `URLSession` в фоне, кэш в памяти, один повтор при
/// ошибке. Возвращает только `Data` (Sendable), поэтому главный актор не
/// блокируется и не пересекает границу изоляции.
enum TaleImageLoader {
    /// Сколько раз пробуем: первая попытка и один повтор (требование 0224).
    static let attempts = 2
    /// Пауза перед повтором: короткая, чтобы картинка успела появиться.
    static let retryDelay: UInt64 = 300_000_000

    nonisolated static func load(_ url: URL) async -> Data? {
        log("картинка: запрос \(url.absoluteString)")
        if let cached = TaleImageCache.shared.data(for: url) {
            log("картинка: из кэша \(url.lastPathComponent)")
            return cached
        }
        for attempt in 0..<attempts {
            if let data = await fetch(url) {
                TaleImageCache.shared.store(data, for: url)
                log("картинка: скачана \(url.lastPathComponent) (\(data.count) байт)")
                return data
            }
            guard attempt + 1 < attempts else { break }
            log("картинка: повтор после ошибки \(url.lastPathComponent)")
            try? await Task.sleep(nanoseconds: retryDelay)
        }
        log("картинка: не загрузилась \(url.absoluteString)")
        return nil
    }

    private nonisolated static func fetch(_ url: URL) async -> Data? {
        var request = URLRequest(url: url)
        request.timeoutInterval = 8
        request.cachePolicy = .useProtocolCachePolicy
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                return nil
            }
            guard !data.isEmpty, looksLikeImage(data) else { return nil }
            return data
        } catch {
            return nil
        }
    }

    /// `/media/...` отдаёт JPEG/PNG; HTML-ошибку или пустое тело не показываем.
    nonisolated static func looksLikeImage(_ data: Data) -> Bool {
        let bytes = [UInt8](data.prefix(4))
        if bytes.count >= 3, bytes[0] == 0xFF, bytes[1] == 0xD8, bytes[2] == 0xFF {
            return true  // JPEG
        }
        if bytes.count >= 4, bytes[0] == 0x89, bytes[1] == 0x50, bytes[2] == 0x4E,
           bytes[3] == 0x47
        {
            return true  // PNG
        }
        if bytes.count >= 3, bytes[0] == 0x47, bytes[1] == 0x49, bytes[2] == 0x46 {
            return true  // GIF
        }
        return false
    }
}

/// Картинка сказки в окне ответа: два слоя-`NSImageView`, новый проявляется,
/// старый гаснет — кроссфейд ~0,35 с. Последняя картинка остаётся на экране до
/// следующего вопроса; на состояние круга (`speaking`/`idle`) показ не влияет.
@MainActor
final class TaleImageView: NSView {
    /// Длительность кроссфейда: заметно, но не задерживает чтение.
    static let fadeDuration = 0.35

    private var layers: [NSImageView] = []
    private var visibleIndex = 0
    private(set) var currentImage: NSImage?
    /// Подпись картинки: подсказка при наведении и озвучка для VoiceOver.
    private(set) var altText = ""
    /// Есть что показывать (последняя картинка ещё на месте).
    var hasImage: Bool { currentImage != nil }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setUp()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setUp()
    }

    private func setUp() {
        wantsLayer = true
        layer?.cornerRadius = 10
        layer?.masksToBounds = true
        layer?.backgroundColor = NSColor.quaternaryLabelColor.withAlphaComponent(0.12).cgColor

        for _ in 0..<2 {
            let view = NSImageView(frame: bounds)
            view.imageScaling = .scaleProportionallyUpOrDown
            view.imageAlignment = .alignCenter
            view.wantsLayer = true
            view.alphaValue = 0
            view.autoresizingMask = [.width, .height]
            addSubview(view)
            layers.append(view)
        }
        isHidden = true
        setAccessibilityRole(.image)
    }

    /// Показать картинку: новый слой проявляется, прежний гаснет — кроссфейд.
    /// `animated: false` — мгновенно (для самопроверки).
    func show(_ image: NSImage, alt: String = "", animated: Bool = true) {
        guard layers.count == 2 else { return }
        let outgoing = layers[visibleIndex]
        visibleIndex = 1 - visibleIndex
        let incoming = layers[visibleIndex]

        incoming.layer?.removeAllAnimations()
        outgoing.layer?.removeAllAnimations()
        incoming.image = image
        isHidden = false
        currentImage = image

        if altText != alt {
            altText = alt
            toolTip = alt.isEmpty ? nil : alt
            setAccessibilityLabel(alt.isEmpty ? "иллюстрация сказки" : alt)
        }

        guard animated else {
            incoming.alphaValue = 1
            outgoing.image = nil
            outgoing.alphaValue = 0
            return
        }
        // Первой картинке тоже нужен плавный вход, поэтому анимируем новый
        // слой всегда; гаснет только тот, на котором было изображение.
        let hadOutgoing = outgoing.image != nil
        incoming.alphaValue = 0
        NSAnimationContext.runAnimationGroup { context in
            context.duration = TaleImageView.fadeDuration
            context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            incoming.animator().alphaValue = 1
            if hadOutgoing { outgoing.animator().alphaValue = 0 }
        } completionHandler: { [weak self] in
            // AppKit зовёт этот блок на главном потоке, но его тип — @Sendable.
            // assumeIsolated (а не Task) сохраняет порядок: следующий кадр
            // анимации не может обогнать очистку слоя.
            MainActor.assumeIsolated {
                guard let self else { return }
                // Пока шёл кроссфейд, могла прийти новая картинка: гасим
                // только тот слой, который так и остался невидимым.
                guard self.layers[self.visibleIndex] === incoming else { return }
                outgoing.image = nil
                outgoing.alphaValue = 0
            }
        }
    }

    /// Убрать картинку (новый вопрос или выключенная настройка).
    func clear() {
        for view in layers {
            view.layer?.removeAllAnimations()
            view.alphaValue = 0
            view.image = nil
        }
        visibleIndex = 0
        currentImage = nil
        altText = ""
        toolTip = nil
        isHidden = true
    }
}
