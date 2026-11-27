# Иллюстрации к сказкам

Классические иллюстрации к русским народным сказкам из Wikimedia Commons. Все изображения — **общественное достояние** (Public domain); для каждого файла указаны автор, страница файла на Commons и лицензия. Файлы лежат в репозитории: `plugins/fairy-tales/images/<slug>/NN.jpg`, ширина до 1024 px, JPEG ≤ 300 КБ.

Доля текста измерена утилитой [`tools/image_text_coverage.swift`](../../tools/image_text_coverage.swift) (Vision, `VNRecognizeTextRequest`, `.accurate`): это отношение площади объединения текстовых `boundingBox` к площади кадра. Порог **12 %**: выше — скан страницы, а не иллюстрация. У всех 65 оставшихся файлов доля ≤ 12 %.

Подбор к абзацам — по `tags.json` (кто / что / где) и словам абзаца; см. `images_for` в `plugins/fairy-tales/plugin.py`. Файлы `NN` идут по сюжету, `after_chars` ставится на границе абзаца, к которому подобрана картинка.

| Сказка | Файлов | Макс. доля текста, % | Объём |
|---|---|---|---|
| Колобок | 8 | 0.26 | 2.0 МБ |
| Репка | 7 | 1.72 | 0.8 МБ |
| Теремок | 6 | 0.26 | 1.2 МБ |
| Лиса и журавль | 7 | 1.93 | 1.7 МБ |
| Три медведя | 8 | 3.13 | 1.7 МБ |
| Гуси-лебеди | 6 | 0.00 | 0.7 МБ |
| Царевна-лягушка | 7 | 1.86 | 1.9 МБ |
| Баба-яга | 8 | 1.57 | 2.0 МБ |
| Сивка-бурка | 8 | 10.19 | 1.9 МБ |
| **Всего** | **65** | **10.19** | **13.9 МБ** |

## Удалённые и заменённые сканы

Проверено 72 исходных файла; 13 сканов с долей текста > 12 % удалены или заменены. Дополнительно удалён `tri-medvedya/08.jpg` (5.18 %, книжная реклама). Итог: 14 удалено, 7 добавлено, осталось 65 файлов.

| Было | Доля текста, % | Что сделано | Почему |
|---|---|---|---|
| `kolobok/08.jpg` | 44.96 | удалён | скан открытки с текстом; в наборе осталось 8 иллюстраций |
| `repka/01.jpg` | 94.87 | удалён | лубочный лист — почти весь текст; осталось 7 иллюстраций |
| `teremok/01.jpg` | 67.34 | удалён | обложка с названием и выходными данными |
| `teremok/02.jpg` | 34.81 | удалён | страница с текстом |
| `lisa-i-zhuravl/01.jpg` | 38.95 | удалён | английская страница с подписью и текстом |
| `tri-medvedya/01.jpg` | 50.20 | удалён | титульная страница (с. 5); заменён на с. 6 |
| `tri-medvedya/03.jpg` | 31.99 | удалён | текстовая страница (с. 11); заменён на с. 10 |
| `tri-medvedya/04.jpg` | 53.25 | удалён | текстовая страница (с. 15); заменён на с. 17 |
| `tri-medvedya/08.jpg` | 5.18 | удалён | с. 32 — книжная реклама без сюжета (ниже порога, но не иллюстрация) |
| `gusi-lebedi/02.jpg` | 62.03 | удалён | марка с крупным текстом |
| `tsarevna-lyagushka/01.jpg` | 63.01 | удалён | обложка с названием |
| `sivka-burka/02.jpg` | 22.25 | удалён | титульный скан 1906 года; заменён на иллюстрацию Сытина |
| `sivka-burka/04.jpg` | 13.24 | удалён | титульный скан 1916 года; заменён на иллюстрацию Сытина |
| `sivka-burka/06.jpg` | 53.02 | удалён | титульный скан 1915 года; заменён на иллюстрацию Сытина |

Замены: `tri-medvedya` — страницы 6, 10, 17, 20 вместо титульных/текстовых 5, 11, 15; `sivka-burka` — иллюстрации Сытина 1915 года вместо трёх титульных сканов.

## Колобок — `images/kolobok/` (8 файлов)

Семь рисунков К. В. Кузнецова из книги «Колобок» (Детиздат ЦК ВЛКСМ, 1941) идут по сюжету, `08.jpg` — фрагмент рисунка А. Медведева (изд. В. Майстраха, 1913) с финальной сценой у лисы. Открытка Е. М. Бём удалена как скан с текстом.

| Файл | Автор | Источник | Лицензия | Доля текста (OCR), % | Примечание |
|---|---|---|---|---|---|
| `01.jpg` | Константин Кузнецов | [Кузнецов КВ Колобок 1.jpg](https://commons.wikimedia.org/wiki/File:%D0%9A%D1%83%D0%B7%D0%BD%D0%B5%D1%86%D0%BE%D0%B2%20%D0%9A%D0%92%20%D0%9A%D0%BE%D0%BB%D0%BE%D0%B1%D0%BE%D0%BA%201.jpg) | Public domain | 0.00 | 280 КБ |
| `02.jpg` | Константин Кузнецов | [Кузнецов КВ Колобок 2.jpg](https://commons.wikimedia.org/wiki/File:%D0%9A%D1%83%D0%B7%D0%BD%D0%B5%D1%86%D0%BE%D0%B2%20%D0%9A%D0%92%20%D0%9A%D0%BE%D0%BB%D0%BE%D0%B1%D0%BE%D0%BA%202.jpg) | Public domain | 0.00 | 276 КБ |
| `03.jpg` | Константин Кузнецов | [Кузнецов КВ Колобок 3.jpg](https://commons.wikimedia.org/wiki/File:%D0%9A%D1%83%D0%B7%D0%BD%D0%B5%D1%86%D0%BE%D0%B2%20%D0%9A%D0%92%20%D0%9A%D0%BE%D0%BB%D0%BE%D0%B1%D0%BE%D0%BA%203.jpg) | Public domain | 0.00 | 136 КБ |
| `04.jpg` | Константин Кузнецов | [Кузнецов КВ Колобок 4.jpg](https://commons.wikimedia.org/wiki/File:%D0%9A%D1%83%D0%B7%D0%BD%D0%B5%D1%86%D0%BE%D0%B2%20%D0%9A%D0%92%20%D0%9A%D0%BE%D0%BB%D0%BE%D0%B1%D0%BE%D0%BA%204.jpg) | Public domain | 0.00 | 287 КБ |
| `05.jpg` | Константин Кузнецов | [Кузнецов КВ Колобок 5.jpg](https://commons.wikimedia.org/wiki/File:%D0%9A%D1%83%D0%B7%D0%BD%D0%B5%D1%86%D0%BE%D0%B2%20%D0%9A%D0%92%20%D0%9A%D0%BE%D0%BB%D0%BE%D0%B1%D0%BE%D0%BA%205.jpg) | Public domain | 0.26 | 285 КБ |
| `06.jpg` | Константин Кузнецов | [Кузнецов КВ Колобок 6.jpg](https://commons.wikimedia.org/wiki/File:%D0%9A%D1%83%D0%B7%D0%BD%D0%B5%D1%86%D0%BE%D0%B2%20%D0%9A%D0%92%20%D0%9A%D0%BE%D0%BB%D0%BE%D0%B1%D0%BE%D0%BA%206.jpg) | Public domain | 0.00 | 270 КБ |
| `07.jpg` | Константин Кузнецов | [Кузнецов КВ Колобок 7.jpg](https://commons.wikimedia.org/wiki/File:%D0%9A%D1%83%D0%B7%D0%BD%D0%B5%D1%86%D0%BE%D0%B2%20%D0%9A%D0%92%20%D0%9A%D0%BE%D0%BB%D0%BE%D0%B1%D0%BE%D0%BA%207.jpg) | Public domain | 0.00 | 278 КБ |
| `08.jpg` | А. Медведев | [Колобок (Майстрах) 19 crop.png](https://commons.wikimedia.org/wiki/File:%D0%9A%D0%BE%D0%BB%D0%BE%D0%B1%D0%BE%D0%BA%20%28%D0%9C%D0%B0%D0%B9%D1%81%D1%82%D1%80%D0%B0%D1%85%29%2019%20crop.png) | Public domain | 0.00 | 257 КБ |

## Репка — `images/repka/` (7 файлов)

Лубочный лист с текстом удалён. Остались страницы книги В. В. Каррика «Репка; Рукавичка» (Москва, 1916) и кадры мультфильма «Репка» (Мосфильм, реж. А. Птушко, 1936): тянут репку, зовут бабку, внучку, Жучку, кошку, мышку.

| Файл | Автор | Источник | Лицензия | Доля текста (OCR), % | Примечание |
|---|---|---|---|---|---|
| `01.jpg` | Валерий Каррик | [Каррик В. В. Репка; Рукавичка 1916.pdf](https://commons.wikimedia.org/wiki/File:%D0%9A%D0%B0%D1%80%D1%80%D0%B8%D0%BA%20%D0%92.%20%D0%92.%20%D0%A0%D0%B5%D0%BF%D0%BA%D0%B0%3B%20%D0%A0%D1%83%D0%BA%D0%B0%D0%B2%D0%B8%D1%87%D0%BA%D0%B0%201916.pdf) | Public domain | 1.72 | с. 2; 152 КБ |
| `02.jpg` | Валерий Каррик | [Каррик В. В. Репка; Рукавичка 1916.pdf](https://commons.wikimedia.org/wiki/File:%D0%9A%D0%B0%D1%80%D1%80%D0%B8%D0%BA%20%D0%92.%20%D0%92.%20%D0%A0%D0%B5%D0%BF%D0%BA%D0%B0%3B%20%D0%A0%D1%83%D0%BA%D0%B0%D0%B2%D0%B8%D1%87%D0%BA%D0%B0%201916.pdf) | Public domain | 0.00 | с. 3; 142 КБ |
| `03.jpg` | Валерий Каррик | [Каррик В. В. Репка; Рукавичка 1916.pdf](https://commons.wikimedia.org/wiki/File:%D0%9A%D0%B0%D1%80%D1%80%D0%B8%D0%BA%20%D0%92.%20%D0%92.%20%D0%A0%D0%B5%D0%BF%D0%BA%D0%B0%3B%20%D0%A0%D1%83%D0%BA%D0%B0%D0%B2%D0%B8%D1%87%D0%BA%D0%B0%201916.pdf) | Public domain | 0.00 | с. 4; 131 КБ |
| `04.jpg` | Валерий Матвеев. Детское время | [Репка. Мультфильм. 1936 год.webm](https://commons.wikimedia.org/wiki/File:%D0%A0%D0%B5%D0%BF%D0%BA%D0%B0.%20%D0%9C%D1%83%D0%BB%D1%8C%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D0%BC.%201936%20%D0%B3%D0%BE%D0%B4.webm) | Public domain | 0.00 | кадр 3:30; 94 КБ |
| `05.jpg` | Валерий Матвеев. Детское время | [Репка. Мультфильм. 1936 год.webm](https://commons.wikimedia.org/wiki/File:%D0%A0%D0%B5%D0%BF%D0%BA%D0%B0.%20%D0%9C%D1%83%D0%BB%D1%8C%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D0%BC.%201936%20%D0%B3%D0%BE%D0%B4.webm) | Public domain | 0.00 | кадр 7:00; 79 КБ |
| `06.jpg` | Валерий Матвеев. Детское время | [Репка. Мультфильм. 1936 год.webm](https://commons.wikimedia.org/wiki/File:%D0%A0%D0%B5%D0%BF%D0%BA%D0%B0.%20%D0%9C%D1%83%D0%BB%D1%8C%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D0%BC.%201936%20%D0%B3%D0%BE%D0%B4.webm) | Public domain | 0.00 | кадр 9:20; 75 КБ |
| `07.jpg` | Валерий Матвеев. Детское время | [Репка. Мультфильм. 1936 год.webm](https://commons.wikimedia.org/wiki/File:%D0%A0%D0%B5%D0%BF%D0%BA%D0%B0.%20%D0%9C%D1%83%D0%BB%D1%8C%D1%82%D1%84%D0%B8%D0%BB%D1%8C%D0%BC.%201936%20%D0%B3%D0%BE%D0%B4.webm) | Public domain | 0.00 | кадр 10:30; 106 КБ |

## Теремок — `images/teremok/` (6 файлов)

Г. И. Нарбут — рисунки к книге «Теремок. Мизгирь» (изд. И. Кнебель, 1910): насекомые, лиса, заяц, медведь; `03.jpg` — кадр мультфильма «Теремок» (Ленфильм, 1937), `06.jpg` — «Теремок» (Союзмультфильм, 1945). Обложка и страница с текстом удалены.

| Файл | Автор | Источник | Лицензия | Доля текста (OCR), % | Примечание |
|---|---|---|---|---|---|
| `01.jpg` | Георгий Нарбут | [Теремок. Мизгирь 06.jpg](https://commons.wikimedia.org/wiki/File:%D0%A2%D0%B5%D1%80%D0%B5%D0%BC%D0%BE%D0%BA.%20%D0%9C%D0%B8%D0%B7%D0%B3%D0%B8%D1%80%D1%8C%2006.jpg) | Public domain | 0.20 | 274 КБ |
| `02.jpg` | Георгий Нарбут | [Теремок. Мизгирь 02.jpg](https://commons.wikimedia.org/wiki/File:%D0%A2%D0%B5%D1%80%D0%B5%D0%BC%D0%BE%D0%BA.%20%D0%9C%D0%B8%D0%B7%D0%B3%D0%B8%D1%80%D1%8C%2002.jpg) | Public domain | 0.00 | 265 КБ |
| `03.jpg` | Ленфильм | [Теремок (1937).jpg](https://commons.wikimedia.org/wiki/File:%D0%A2%D0%B5%D1%80%D0%B5%D0%BC%D0%BE%D0%BA%20%281937%29.jpg) | Public domain | 0.00 | 103 КБ |
| `04.jpg` | Георгий Нарбут | [Теремок Заяц - Нарбут.jpg](https://commons.wikimedia.org/wiki/File:%D0%A2%D0%B5%D1%80%D0%B5%D0%BC%D0%BE%D0%BA%20%D0%97%D0%B0%D1%8F%D1%86%20-%20%D0%9D%D0%B0%D1%80%D0%B1%D1%83%D1%82.jpg) | Public domain | 0.08 | 267 КБ |
| `05.jpg` | Георгий Нарбут | [Теремок Медведь - Нарбут.jpg](https://commons.wikimedia.org/wiki/File:%D0%A2%D0%B5%D1%80%D0%B5%D0%BC%D0%BE%D0%BA%20%D0%9C%D0%B5%D0%B4%D0%B2%D0%B5%D0%B4%D1%8C%20-%20%D0%9D%D0%B0%D1%80%D0%B1%D1%83%D1%82.jpg) | Public domain | 0.26 | 251 КБ |
| `06.jpg` | Союзмультфильм | [Rooster in Teremok (1945).jpg](https://commons.wikimedia.org/wiki/File:Rooster%20in%20Teremok%20%281945%29.jpg) | Public domain | 0.00 | 83 КБ |

## Лиса и журавль — `images/lisa-i-zhuravl/` (7 файлов)

Русских иллюстраций к этой сказке в свободных источниках почти нет, поэтому собраны классические западноевропейские иллюстрации того же сюжета — басни «Лис и журавль» (Le Renard et la cigogne): гравюры XVI–XIX веков и рисунки к Лафонтену. Английская страница с текстом удалена.

| Файл | Автор | Источник | Лицензия | Доля текста (OCR), % | Примечание |
|---|---|---|---|---|---|
| `01.jpg` | Jean Jacques Flipart / After Jean-Baptiste Oudry | [Baquoy-Oudry-La Fontaine-Le renard et la cigogne.jpg](https://commons.wikimedia.org/wiki/File:Baquoy-Oudry-La%20Fontaine-Le%20renard%20et%20la%20cigogne.jpg) | Public domain | 1.00 | 284 КБ |
| `02.jpg` | Jean-Baptiste Oudry | [Jean-Baptiste Oudry - Renard et la cigogne (1747).jpg](https://commons.wikimedia.org/wiki/File:Jean-Baptiste%20Oudry%20-%20Renard%20et%20la%20cigogne%20%281747%29.jpg) | Public domain | 0.00 | 167 КБ |
| `03.jpg` | Jean Ignace Isidore Gérard Grandville | [1-18-Le-renard-et-la-cigogne.jpg](https://commons.wikimedia.org/wiki/File:1-18-Le-renard-et-la-cigogne.jpg) | Public domain | 0.00 | 254 КБ |
| `04.jpg` | Marcus Gheeraerts the Elder | [Marcus Gheeraerts I - The fox and the stork.jpg](https://commons.wikimedia.org/wiki/File:Marcus%20Gheeraerts%20I%20-%20The%20fox%20and%20the%20stork.jpg) | Public domain | 0.00 | 245 КБ |
| `05.jpg` | Огюст Дельер (по басне Ж. де Лафонтена) | [Fables de La Fontaine - Auguste Delierre - Le Renard et la Cigogne.jpg](https://commons.wikimedia.org/wiki/File:Fables%20de%20La%20Fontaine%20-%20Auguste%20Delierre%20-%20Le%20Renard%20et%20la%20Cigogne.jpg) | Public domain | 1.60 | 292 КБ |
| `06.jpg` | Michel Aubert / After Jean-Baptiste Oudry | [Aubert-Oudry-La Fontaine-Le renard et la cigogne.jpg](https://commons.wikimedia.org/wiki/File:Aubert-Oudry-La%20Fontaine-Le%20renard%20et%20la%20cigogne.jpg) | Public domain | 0.00 | 290 КБ |
| `07.jpg` | неизвестен | [The Fox and The Stork Aesopus moralisatus 1487.png](https://commons.wikimedia.org/wiki/File:The%20Fox%20and%20The%20Stork%20Aesopus%20moralisatus%201487.png) | Public domain | 1.93 | 241 КБ |

## Три медведя — `images/tri-medvedya/` (8 файлов)

Л. Лесли Брук, «The Story of the Three Bears» (1905), страницы 6, 8, 10, 17, 19, 20, 23, 27 по порядку: девочка в лесу, дом, похлёбка, стулья, кровати, возвращение медведей, побег. Титульные и текстовые страницы (с. 5, 11, 15), а также книжная реклама (с. 32) удалены.

| Файл | Автор | Источник | Лицензия | Доля текста (OCR), % | Примечание |
|---|---|---|---|---|---|
| `01.jpg` | Леонард Лесли Брук | [The Story of the Three Bears pg 6.jpg](https://commons.wikimedia.org/wiki/File:The%20Story%20of%20the%20Three%20Bears%20pg%206.jpg) | Public domain | 0.71 | с. 6; замена скана с. 5; 188 КБ |
| `02.jpg` | Леонард Лесли Брук | [The Story of the Three Bears pg 8.jpg](https://commons.wikimedia.org/wiki/File:The%20Story%20of%20the%20Three%20Bears%20pg%208.jpg) | Public domain | 3.13 | с. 8; 274 КБ |
| `03.jpg` | Леонард Лесли Брук | [The Story of the Three Bears pg 10.jpg](https://commons.wikimedia.org/wiki/File:The%20Story%20of%20the%20Three%20Bears%20pg%2010.jpg) | Public domain | 0.00 | с. 10; замена скана с. 11; 247 КБ |
| `04.jpg` | Леонард Лесли Брук | [The Story of the Three Bears pg 17.jpg](https://commons.wikimedia.org/wiki/File:The%20Story%20of%20the%20Three%20Bears%20pg%2017.jpg) | Public domain | 0.00 | с. 17; замена скана с. 15; 218 КБ |
| `05.jpg` | Леонард Лесли Брук | [The Story of the Three Bears pg 19.jpg](https://commons.wikimedia.org/wiki/File:The%20Story%20of%20the%20Three%20Bears%20pg%2019.jpg) | Public domain | 0.00 | с. 19; 219 КБ |
| `06.jpg` | Леонард Лесли Брук | [The Story of the Three Bears pg 20.jpg](https://commons.wikimedia.org/wiki/File:The%20Story%20of%20the%20Three%20Bears%20pg%2020.jpg) | Public domain | 0.00 | с. 20; дополнительная иллюстрация; 257 КБ |
| `07.jpg` | Леонард Лесли Брук | [The Story of the Three Bears pg 23.jpg](https://commons.wikimedia.org/wiki/File:The%20Story%20of%20the%20Three%20Bears%20pg%2023.jpg) | Public domain | 0.14 | с. 23; 243 КБ |
| `08.jpg` | Леонард Лесли Брук | [The Story of the Three Bears pg 27.jpg](https://commons.wikimedia.org/wiki/File:The%20Story%20of%20the%20Three%20Bears%20pg%2027.jpg) | Public domain | 0.39 | с. 27; 87 КБ |

## Гуси-лебеди — `images/gusi-lebedi/` (6 файлов)

Открытка Ф. Тейхеля и кадры мультфильма «Гуси-лебеди» (Союзмультфильм, реж. И. Иванов-Вано, 1949) — от похищения братца до возвращения домой. Марка СССР с крупным текстом удалена.

| Файл | Автор | Источник | Лицензия | Доля текста (OCR), % | Примечание |
|---|---|---|---|---|---|
| `01.jpg` | Франц Тейхель | [Гуси-лебеди - Тейхель.jpg](https://commons.wikimedia.org/wiki/File:%D0%93%D1%83%D1%81%D0%B8-%D0%BB%D0%B5%D0%B1%D0%B5%D0%B4%D0%B8%20-%20%D0%A2%D0%B5%D0%B9%D1%85%D0%B5%D0%BB%D1%8C.jpg) | Public domain | 0.00 | 289 КБ |
| `02.jpg` | «Союзмультфильм» | [1949. Гуси-лебеди.webm](https://commons.wikimedia.org/wiki/File:1949.%20%D0%93%D1%83%D1%81%D0%B8-%D0%BB%D0%B5%D0%B1%D0%B5%D0%B4%D0%B8.webm) | Public domain | 0.00 | кадр 3:00; 86 КБ |
| `03.jpg` | «Союзмультфильм» | [1949. Гуси-лебеди.webm](https://commons.wikimedia.org/wiki/File:1949.%20%D0%93%D1%83%D1%81%D0%B8-%D0%BB%D0%B5%D0%B1%D0%B5%D0%B4%D0%B8.webm) | Public domain | 0.00 | кадр 6:00; 104 КБ |
| `04.jpg` | «Союзмультфильм» | [1949. Гуси-лебеди.webm](https://commons.wikimedia.org/wiki/File:1949.%20%D0%93%D1%83%D1%81%D0%B8-%D0%BB%D0%B5%D0%B1%D0%B5%D0%B4%D0%B8.webm) | Public domain | 0.00 | кадр 8:00; 75 КБ |
| `05.jpg` | «Союзмультфильм» | [1949. Гуси-лебеди.webm](https://commons.wikimedia.org/wiki/File:1949.%20%D0%93%D1%83%D1%81%D0%B8-%D0%BB%D0%B5%D0%B1%D0%B5%D0%B4%D0%B8.webm) | Public domain | 0.00 | кадр 12:00; 58 КБ |
| `06.jpg` | «Союзмультфильм» | [1949. Гуси-лебеди.webm](https://commons.wikimedia.org/wiki/File:1949.%20%D0%93%D1%83%D1%81%D0%B8-%D0%BB%D0%B5%D0%B1%D0%B5%D0%B4%D0%B8.webm) | Public domain | 0.00 | кадр 16:30; 81 КБ |

## Царевна-лягушка — `images/tsarevna-lyagushka/` (7 файлов)

И. Я. Билибин, серия «Царевна-лягушка» (1899–1901), страницы 1–6 по порядку; `07.jpg` — картина В. М. Васнецова «Царевна-лягушка» (1918). Обложка с названием удалена.

| Файл | Автор | Источник | Лицензия | Доля текста (OCR), % | Примечание |
|---|---|---|---|---|---|
| `01.jpg` | Иван Билибин | [Царевна лягушка 1.jpg](https://commons.wikimedia.org/wiki/File:%D0%A6%D0%B0%D1%80%D0%B5%D0%B2%D0%BD%D0%B0%20%D0%BB%D1%8F%D0%B3%D1%83%D1%88%D0%BA%D0%B0%201.jpg) | Public domain | 0.00 | с. 1; 279 КБ |
| `02.jpg` | Иван Билибин | [Царевна лягушка 2.jpg](https://commons.wikimedia.org/wiki/File:%D0%A6%D0%B0%D1%80%D0%B5%D0%B2%D0%BD%D0%B0%20%D0%BB%D1%8F%D0%B3%D1%83%D1%88%D0%BA%D0%B0%202.jpg) | Public domain | 0.13 | с. 2; 278 КБ |
| `03.jpg` | Иван Билибин | [Царевна лягушка 3.jpg](https://commons.wikimedia.org/wiki/File:%D0%A6%D0%B0%D1%80%D0%B5%D0%B2%D0%BD%D0%B0%20%D0%BB%D1%8F%D0%B3%D1%83%D1%88%D0%BA%D0%B0%203.jpg) | Public domain | 1.86 | с. 3; 292 КБ |
| `04.jpg` | Иван Билибин | [Царевна лягушка 4.jpg](https://commons.wikimedia.org/wiki/File:%D0%A6%D0%B0%D1%80%D0%B5%D0%B2%D0%BD%D0%B0%20%D0%BB%D1%8F%D0%B3%D1%83%D1%88%D0%BA%D0%B0%204.jpg) | Public domain | 0.24 | с. 4; 278 КБ |
| `05.jpg` | Иван Билибин | [Царевна лягушка 5.jpg](https://commons.wikimedia.org/wiki/File:%D0%A6%D0%B0%D1%80%D0%B5%D0%B2%D0%BD%D0%B0%20%D0%BB%D1%8F%D0%B3%D1%83%D1%88%D0%BA%D0%B0%205.jpg) | Public domain | 0.36 | с. 5; 281 КБ |
| `06.jpg` | Иван Билибин | [Царевна лягушка 6.jpg](https://commons.wikimedia.org/wiki/File:%D0%A6%D0%B0%D1%80%D0%B5%D0%B2%D0%BD%D0%B0%20%D0%BB%D1%8F%D0%B3%D1%83%D1%88%D0%BA%D0%B0%206.jpg) | Public domain | 0.28 | с. 6; 266 КБ |
| `07.jpg` | Виктор Васнецов | [Vasnetsov Frog Princess.jpg](https://commons.wikimedia.org/wiki/File:Vasnetsov%20Frog%20Princess.jpg) | Public domain | 0.58 | 246 КБ |

## Баба-яга — `images/baba-yaga/` (8 файлов)

Н. Н. Каразин, И. Я. Билибин (в том числе «Баба-яга и девицы-птицы», 1902), В. М. Васнецов, Н. А. Богатов, А. Л. Обер, Б. В. Зворыкин, А. Н. Бенуа. Все изображения прошли OCR-проверку.

| Файл | Автор | Источник | Лицензия | Доля текста (OCR), % | Примечание |
|---|---|---|---|---|---|
| `01.jpg` | Николай Каразин | [Baba Yaga by N.Karazin.jpg](https://commons.wikimedia.org/wiki/File:Baba%20Yaga%20by%20N.Karazin.jpg) | Public domain | 0.00 | 286 КБ |
| `02.jpg` | Иван Билибин | [Bilibin. Baba Yaga.jpg](https://commons.wikimedia.org/wiki/File:Bilibin.%20Baba%20Yaga.jpg) | Public domain | 0.37 | 281 КБ |
| `03.jpg` | Виктор Васнецов | [Баба Яга.jpg](https://commons.wikimedia.org/wiki/File:%D0%91%D0%B0%D0%B1%D0%B0%20%D0%AF%D0%B3%D0%B0.jpg) | Public domain | 0.00 | 266 КБ |
| `04.jpg` | Николай Богатов | [Сказка Баба-яга 3.jpg](https://commons.wikimedia.org/wiki/File:%D0%A1%D0%BA%D0%B0%D0%B7%D0%BA%D0%B0%20%D0%91%D0%B0%D0%B1%D0%B0-%D1%8F%D0%B3%D0%B0%203.jpg) | Public domain | 1.16 | 285 КБ |
| `05.jpg` | Артемий Обер | [Баба-Яга. А. Л. Обер.JPG](https://commons.wikimedia.org/wiki/File:%D0%91%D0%B0%D0%B1%D0%B0-%D0%AF%D0%B3%D0%B0.%20%D0%90.%20%D0%9B.%20%D0%9E%D0%B1%D0%B5%D1%80.JPG) | Public domain | 0.00 | 203 КБ |
| `06.jpg` | Борис Зворыкин | [Baba-Yaga by B.Zvorykin.jpg](https://commons.wikimedia.org/wiki/File:Baba-Yaga%20by%20B.Zvorykin.jpg) | Public domain | 1.57 | 279 КБ |
| `07.jpg` | Александр Бенуа | [Baba Yaga by Alexandre Benois.jpg](https://commons.wikimedia.org/wiki/File:Baba%20Yaga%20by%20Alexandre%20Benois.jpg) | Public domain | 1.35 | 281 КБ |
| `08.jpg` | Иван Билибин | [Baba Yaga and maiden-birds by I.Bilibin (1902).jpg](https://commons.wikimedia.org/wiki/File:Baba%20Yaga%20and%20maiden-birds%20by%20I.Bilibin%20%281902%29.jpg) | Public domain | 0.41 | 177 КБ |

## Сивка-бурка — `images/sivka-burka/` (8 файлов)

В. М. Васнецов, издание И. Д. Сытина 1914–1916 годов, Б. В. Зворыкин и открытка В. В. Владимирова. Титульные сканы заменены иллюстрациями Сытина 1915 года.

| Файл | Автор | Источник | Лицензия | Доля текста (OCR), % | Примечание |
|---|---|---|---|---|---|
| `01.jpg` | Виктор Васнецов | [Сивка-Бурка.jpg](https://commons.wikimedia.org/wiki/File:%D0%A1%D0%B8%D0%B2%D0%BA%D0%B0-%D0%91%D1%83%D1%80%D0%BA%D0%B0.jpg) | Public domain | 0.00 | 288 КБ |
| `02.jpg` | неизвестен | [Сивка-бурка, вещая каурка 1916 2.jpg](https://commons.wikimedia.org/wiki/File:%D0%A1%D0%B8%D0%B2%D0%BA%D0%B0-%D0%B1%D1%83%D1%80%D0%BA%D0%B0%2C%20%D0%B2%D0%B5%D1%89%D0%B0%D1%8F%20%D0%BA%D0%B0%D1%83%D1%80%D0%BA%D0%B0%201916%202.jpg) | Public domain | 10.19 | 276 КБ |
| `03.jpg` | неизвестен | [Сивка-бурка, вещая каурка 1914.jpg](https://commons.wikimedia.org/wiki/File:%D0%A1%D0%B8%D0%B2%D0%BA%D0%B0-%D0%B1%D1%83%D1%80%D0%BA%D0%B0%2C%20%D0%B2%D0%B5%D1%89%D0%B0%D1%8F%20%D0%BA%D0%B0%D1%83%D1%80%D0%BA%D0%B0%201914.jpg) | Public domain | 6.74 | 262 КБ |
| `04.jpg` | неизвестен (типография И. Д. Сытина) | [Сивка-бурка Издание Сытина 3.jpg](https://commons.wikimedia.org/wiki/File:%D0%A1%D0%B8%D0%B2%D0%BA%D0%B0-%D0%B1%D1%83%D1%80%D0%BA%D0%B0%20%D0%98%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%A1%D1%8B%D1%82%D0%B8%D0%BD%D0%B0%203.jpg) | Public domain | 0.75 | 1915; замена титульного скана; 184 КБ |
| `05.jpg` | неизвестен (типография И. Д. Сытина) | [Сивка-бурка Издание Сытина 2.jpg](https://commons.wikimedia.org/wiki/File:%D0%A1%D0%B8%D0%B2%D0%BA%D0%B0-%D0%B1%D1%83%D1%80%D0%BA%D0%B0%20%D0%98%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%A1%D1%8B%D1%82%D0%B8%D0%BD%D0%B0%202.jpg) | Public domain | 0.00 | 1915; замена титульного скана; 244 КБ |
| `06.jpg` | неизвестен (типография И. Д. Сытина) | [Сивка-бурка Издание Сытина 4.jpg](https://commons.wikimedia.org/wiki/File:%D0%A1%D0%B8%D0%B2%D0%BA%D0%B0-%D0%B1%D1%83%D1%80%D0%BA%D0%B0%20%D0%98%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%A1%D1%8B%D1%82%D0%B8%D0%BD%D0%B0%204.jpg) | Public domain | 0.18 | 1915; замена титульного скана; 230 КБ |
| `07.jpg` | Борис Зворыкин | [Sivka-Burka by B.Zvorykin.jpg](https://commons.wikimedia.org/wiki/File:Sivka-Burka%20by%20B.Zvorykin.jpg) | Public domain | 1.13 | 290 КБ |
| `08.jpg` | Vasily Vladimirov | [Василий Владимиров. Сивка Бурка.jpg](https://commons.wikimedia.org/wiki/File:%D0%92%D0%B0%D1%81%D0%B8%D0%BB%D0%B8%D0%B9%20%D0%92%D0%BB%D0%B0%D0%B4%D0%B8%D0%BC%D0%B8%D1%80%D0%BE%D0%B2.%20%D0%A1%D0%B8%D0%B2%D0%BA%D0%B0%20%D0%91%D1%83%D1%80%D0%BA%D0%B0.jpg) | Public domain | 5.38 | 163 КБ |

## Замечания

- Для «Лисы и журавля» свободных русских иллюстраций почти нет: взяты западноевропейские гравюры и рисунки к той же басне Лафонтена (сюжет совпадает: лиса угощает журавля с тарелки, журавль лису — из кувшина).
- Кадры из мультфильмов («Теремок», 1937 и 1945; «Гуси-лебеди», 1949; «Репка», 1936) — общественное достояние по данным Commons.
- Иллюстрации не генерировались: только сканы и репродукции из Commons. Скачивание — через `commons.wikimedia.org/w/thumb.php?f=<file>&w=1024`.

Источник: https://commons.wikimedia.org/ (API `action=query&prop=imageinfo` и `w/thumb.php`).
