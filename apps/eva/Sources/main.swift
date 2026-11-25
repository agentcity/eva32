import AppKit
import Foundation

/// Точка входа.
///
/// Обычный запуск (`open EVA.app` или двойной клик): приложение живёт в
/// статус-баре, окно появляется по ⌥Space или из меню.
///
/// Режимы `--selftest-*` печатают результат в stdout и выходят; они не создают
/// ни окна, ни статус-бара, поэтому их можно гонять из терминала.

/// Запустить самопроверку на главном акторе и не возвращаться.
func runSelfTest(_ mode: SelfTest.Mode) -> Never {
    Task { @MainActor in
        let code = await SelfTest.run(mode)
        fflush(stdout)
        exit(code)
    }
    dispatchMain()
}

if let mode = Launch.selftest {
    runSelfTest(mode)
}

// Top-level код исполняется на главном потоке, но в языке 5 не помечен
// @MainActor; assumeIsolated говорит компилятору то, что и так верно.
MainActor.assumeIsolated {
    let application = NSApplication.shared
    let appDelegate = AppDelegate()
    application.delegate = appDelegate
    application.setActivationPolicy(.accessory)
    application.run()
}
