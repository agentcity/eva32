import AppKit
import Foundation

/// Один источник правды для всех кругов Евы.
///
/// Кругов теперь два: большой в главном окне и маленький в компактном окне
/// (задача 0188). Состояние по-прежнему выбирает `AppDelegate`
/// (`refreshVisualState`), а `CircleHub` только раскладывает уже готовое
/// значение по зарегистрированным `CircleView` — второй копии логики нет.
///
/// Почему хаб, а не общий `CircleView`: одно и то же вью нельзя держать в двух
/// окнах одновременно (анимация и слой привязаны к окну). Рисует оба круга
/// один и тот же класс `CircleView`, а состояние, уровень микрофона, вспышка
/// ошибки, ожидание фразы и follow-up приходят из этого хаба. Пока окно
/// скрыто, его круг всё равно получает значения: при развороте не нужно
/// догонять состояние.
///
/// Ссылки на вью — слабые: окна живут дольше, а вью пересоздаются при сборке
/// окна. Новому вью хаб сразу отдаёт текущий снимок, поэтому порядок
/// «зарегистрировал — показал» не важен.
@MainActor
final class CircleHub {
    private struct WeakCircle {
        weak var view: CircleView?
    }

    private var views: [WeakCircle] = []

    private(set) var state: EvaCircleState = .idle
    private(set) var level: CGFloat = 0
    private(set) var wakeWaiting = false
    private(set) var followUp = false
    private(set) var animationEnabled = true
    private var errorMessage: String?

    /// Подключить круг (главное окно или компактное). Сразу применяет снимок,
    /// чтобы только что созданное окно выглядело как всё остальное.
    func register(_ view: CircleView) {
        views.removeAll { $0.view == nil }
        views.append(WeakCircle(view: view))
        view.animationEnabled = animationEnabled
        view.wakeWaiting = wakeWaiting
        view.followUpListening = followUp
        view.visualState = state
        view.setLevel(level)
        if state == .error, let errorMessage {
            view.showError(errorMessage)
        }
    }

    func setState(_ next: EvaCircleState) {
        state = next
        if next != .error {
            errorMessage = nil
        }
        for view in liveViews {
            view.visualState = next
        }
    }

    func setLevel(_ value: CGFloat) {
        let clamped = min(max(value, 0), 1)
        level = clamped
        for view in liveViews {
            view.setLevel(clamped)
        }
    }

    /// Красная вспышка и текст ошибки: состояние `.error` следом выставит
    /// `AppDelegate.refreshVisualState`, но запомнить текст нужно уже здесь —
    /// компактное окно могло быть создано в момент вспышки.
    func showError(_ message: String) {
        errorMessage = message
        for view in liveViews {
            view.showError(message)
        }
    }

    /// Ожидание «Привет, Ева»: в `idle` оба круга пульсируют одинаково.
    func setWakeWaiting(_ waiting: Bool) {
        guard waiting != wakeWaiting else { return }
        wakeWaiting = waiting
        for view in liveViews {
            view.wakeWaiting = waiting
        }
    }

    /// Идёт follow-up: `listening`, но волны приглушены.
    func setFollowUp(_ active: Bool) {
        guard active != followUp else { return }
        followUp = active
        for view in liveViews {
            view.followUpListening = active
        }
    }

    /// Галочка «Анимация круга»: действует на все круги сразу.
    func setAnimationEnabled(_ enabled: Bool) {
        guard enabled != animationEnabled else { return }
        animationEnabled = enabled
        for view in liveViews {
            view.animationEnabled = enabled
        }
    }

    /// Остановить таймеры при выходе из приложения.
    func stopAnimation() {
        for view in liveViews {
            view.stopAnimation()
        }
    }

    private var liveViews: [CircleView] {
        views.removeAll { $0.view == nil }
        return views.compactMap(\.view)
    }
}
