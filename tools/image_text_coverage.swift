// image_text_coverage.swift — доля площади JPEG, занятая распознанным текстом.
//
// Задача 0232: отсеять «иллюстрации», которые на самом деле являются сканами
// книжных страниц с текстом. Для каждого файла Vision (VNRecognizeTextRequest,
// accurate) находит текстовые наблюдения; берётся объединение их boundingBox
// (нормализованные координаты, origin — левый нижний угол), его площадь
// делится на площадь изображения.
//
// Сборка и запуск:
//   swiftc -O tools/image_text_coverage.swift -o /tmp/imgcov
//   /tmp/imgcov plugins/fairy-tales/images/*/*.jpg
//
// Если swiftc не может писать в системный кэш модулей (песочница):
//   TMPDIR=$PWD/.tmp swiftc -O -module-cache-path $PWD/.swift-cache \
//       tools/image_text_coverage.swift -o /tmp/imgcov
//
// Вывод: «путь<TAB>доля в процентах<TAB>число строк». Диагностика — в stderr.

import AppKit
import Foundation
import Vision

/// Объединение прямоугольников в нормализованных координатах Vision.
private func unionArea(_ boxes: [CGRect]) -> Double {
    guard var union = boxes.first else { return 0 }
    for box in boxes.dropFirst() {
        union = union.union(box)
    }
    let area = Double(union.width) * Double(union.height)
    return min(max(area, 0), 1)
}

/// Доля текста в одном файле: (проценты, число наблюдений) или nil.
private func coverage(path: String) -> (Double, Int)? {
    guard let image = NSImage(contentsOfFile: path),
          let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil)
    else { return nil }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.recognitionLanguages = ["ru-RU", "en-US"]

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    do {
        try handler.perform([request])
    } catch {
        FileHandle.standardError.write(
            Data("warning: \(path): \(error)\n".utf8)
        )
        return nil
    }

    let observations = (request.results ?? []) as [VNRecognizedTextObservation]
    let boxes = observations.compactMap { observation -> CGRect? in
        // Пустые/вырожденные боксы не считаем: они не занимают площадь.
        let box = observation.boundingBox
        guard box.width > 0, box.height > 0 else { return nil }
        return box
    }
    guard !boxes.isEmpty else { return (0.0, 0) }
    return (unionArea(boxes) * 100.0, observations.count)
}

let paths = Array(CommandLine.arguments.dropFirst())
if paths.isEmpty {
    FileHandle.standardError.write(
        Data("usage: imgcov <image.jpg> [image.jpg ...]\n".utf8)
    )
    exit(2)
}

for path in paths {
    autoreleasepool {
        guard let (percent, lines) = coverage(path: path) else {
            FileHandle.standardError.write(
                Data("skip: \(path): не изображение или Vision не справился\n".utf8)
            )
            return
        }
        print(String(format: "%@\t%.2f\t%d", path, percent, lines))
    }
}
