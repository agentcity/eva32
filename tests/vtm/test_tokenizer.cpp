// Тесты BPE-токенизатора VTM (C++17, без внешних зависимостей).
//
// Запуск:  ./test_tokenizer ../../tests/vtm/fixtures/tokenizer.json
// (Makefile.tok вызывает именно так). Возврат != 0 при любом провале.
#include "tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

int g_checks = 0;
int g_fails = 0;
std::string g_fixture_path;

std::string escape_bytes(const std::string &s) {
    std::string out;
    for (unsigned char c : s) {
        if (c >= 0x20 && c < 0x7F && c != '\\') {
            out.push_back(static_cast<char>(c));
        } else if (c == '\\') {
            out += "\\\\";
        } else {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\x%02X", c);
            out += buf;
        }
    }
    return out;
}

std::string read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::string();
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

void fail(const char *file, int line, const std::string &what) {
    ++g_fails;
    std::printf("FAIL %s:%d: %s\n", file, line, what.c_str());
}

#define CHECK(cond)                                                        \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(cond)) fail(__FILE__, __LINE__, #cond);                      \
    } while (0)

#define CHECK_MSG(cond, msg)                                               \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(cond)) fail(__FILE__, __LINE__, std::string(#cond) + ": " + (msg)); \
    } while (0)

void check_text(const std::string &got, const std::string &want,
                const std::string &what, int line) {
    ++g_checks;
    if (got != want) {
        fail(__FILE__, line,
             what + "\n  получили: \"" + escape_bytes(got) +
                 "\"\n  ожидали:  \"" + escape_bytes(want) + "\"");
    }
}

void check_ids(const std::vector<int32_t> &got,
               std::initializer_list<int32_t> want, const std::string &what,
               int line) {
    ++g_checks;
    const std::vector<int32_t> expected(want);
    if (got != expected) {
        std::string g;
        std::string w;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (i) g += ",";
            g += std::to_string(got[i]);
        }
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (i) w += ",";
            w += std::to_string(expected[i]);
        }
        fail(__FILE__, line,
             what + "\n  получили ids: [" + g + "]\n  ожидали ids:  [" + w + "]");
    }
}

void roundtrip(const vtm::Tokenizer &tok, const std::string &text,
               const std::string &label, int line) {
    const std::vector<int32_t> ids = tok.encode(text);
    const std::string back = tok.decode(ids);
    ++g_checks;
    if (back != text) {
        fail(__FILE__, line,
             "round-trip " + label + " (" + std::to_string(ids.size()) +
                 " ids)\n  вход:    \"" + escape_bytes(text) +
                 "\"\n  обратно: \"" + escape_bytes(back) + "\"");
    }
}

#define ROUNDTRIP(tok, text, label) roundtrip(tok, text, label, __LINE__)

bool load_throws(const std::string &json, const std::string &source,
                 std::string *message) {
    try {
        const vtm::Tokenizer tok = vtm::Tokenizer::from_json(json, source);
        (void)tok;
    } catch (const vtm::TokenizerError &e) {
        if (message != nullptr) *message = e.what();
        return true;
    } catch (const std::exception &e) {
        if (message != nullptr) *message = e.what();
        return true;
    }
    return false;
}

bool load_file_throws(const std::string &path, std::string *message) {
    try {
        const vtm::Tokenizer tok = vtm::Tokenizer::load(path);
        (void)tok;
    } catch (const vtm::TokenizerError &e) {
        if (message != nullptr) *message = e.what();
        return true;
    } catch (const std::exception &e) {
        if (message != nullptr) *message = e.what();
        return true;
    }
    return false;
}

// Мини-JSON: словарь без полного byte-алфавита, но достаточный для
// проверки разбора merges/added_tokens/normalizer.
std::string mini_tokenizer_json(const std::string &extra_root,
                                const std::string &normalizer,
                                const std::string &added) {
    return std::string(
               "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":1,"
               "\"ab\":2,\"c\":3},\"merges\":[[\"a\",\"b\"]],\"unk_token\":null,"
               "\"eos_token\":\"<e>\"},\"normalizer\":" ) +
           normalizer +
           ",\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"use_regex\":true,"
           "\"add_prefix_space\":false},\"decoder\":{\"type\":\"ByteLevel\"},"
           "\"added_tokens\":" + added + extra_root + "}";
}

void test_info(const vtm::Tokenizer &tok) {
    const vtm::TokenizerInfo &info = tok.info();
    std::printf("== info ==\n");
    CHECK(info.vocab_size == 287);
    CHECK(info.merge_count == 31);
    CHECK(info.merges_missing_from_vocab == 0);
    CHECK(info.added_token_count == 4);
    CHECK(info.special_token_count == 4);
    CHECK(info.byte_alphabet_size == 256);
    CHECK(info.byte_level);
    CHECK(info.use_regex);
    CHECK(!info.add_prefix_space);
    CHECK(info.model_type == "BPE");
    CHECK(info.normalizer_type == "null");
    CHECK(info.pre_tokenizer_type == "ByteLevel");
    CHECK(info.decoder_type == "ByteLevel");
    CHECK(info.unk_id == -1);
    CHECK(info.eos_id == 501);
    CHECK(info.eos_token == "<|im_end|>");
    CHECK(info.warnings.empty());
    CHECK_MSG(info.source == g_fixture_path,
              "source=\"" + info.source + "\", fixture=\"" + g_fixture_path + "\"");
}

void test_known_merges(const vtm::Tokenizer &tok) {
    std::printf("== известные мержи (фикстура) ==\n");
    int32_t hello_id = -1;
    CHECK(tok.token_to_id("hello", &hello_id));
    check_ids(tok.encode("hello"), {hello_id}, "encode(hello)", __LINE__);
    check_text(tok.id_to_token(hello_id), "hello", "id_to_token(hello)", __LINE__);

    const std::vector<int32_t> hello_world = tok.encode("hello world");
    ++g_checks;
    if (hello_world.size() != 2) {
        fail(__FILE__, __LINE__,
             "encode(\"hello world\") должен быть 2 токена, получили " +
                 std::to_string(hello_world.size()));
    } else {
        CHECK(hello_world[0] == hello_id);
        check_text(tok.decode(hello_world), "hello world",
                   "decode(encode(hello world))", __LINE__);
    }

    // Кириллица и эмодзи — по одному токену на слово (цепочка мержей).
    const std::vector<int32_t> privet = tok.encode("Привет");
    ++g_checks;
    if (privet.size() != 1) {
        fail(__FILE__, __LINE__,
             "encode(Привет) должен быть 1 токен, получили " +
                 std::to_string(privet.size()));
    }
    check_text(tok.decode(tok.encode("Привет")), "Привет", "decode(Привет)",
               __LINE__);
    check_text(tok.decode(tok.encode("мир")), "мир", "decode(мир)", __LINE__);

    const std::vector<int32_t> smile = tok.encode("\xF0\x9F\x99\x82");  // 🙂
    ++g_checks;
    if (smile.size() != 1) {
        fail(__FILE__, __LINE__,
             "encode(🙂) должен быть 1 токен, получили " +
                 std::to_string(smile.size()));
    }
    check_text(tok.decode(smile), "\xF0\x9F\x99\x82", "decode(🙂)", __LINE__);

    // 'мир' отдельно и ' мир' с ведущим пробелом — разные токены.
    CHECK(tok.encode("мир") != tok.encode(" мир"));
}

void test_specials(const vtm::Tokenizer &tok) {
    std::printf("== added_tokens / special ==\n");
    int32_t start = -1;
    int32_t end = -1;
    int32_t im = -1;
    int32_t eot = -1;
    CHECK(tok.token_to_id("<|im_start|>", &start));
    CHECK(tok.token_to_id("<|im_end|>", &end));
    CHECK(tok.token_to_id("<|im|>", &im));
    CHECK(tok.token_to_id("<|endoftext|>", &eot));
    CHECK(start == 500);
    CHECK(end == 501);
    CHECK(im == 503);
    CHECK(eot == 502);
    int32_t hello = -1;
    CHECK(tok.token_to_id("hello", &hello));

    const std::string text = "<|im_start|>hello<|im_end|>";
    const std::vector<int32_t> ids = tok.encode(text);
    check_ids(ids, {start, hello, end}, "encode со special", __LINE__);
    check_text(tok.decode(ids), text, "decode со special", __LINE__);

    // Спецтокен в окружении обычного текста не разбивается BPE.
    const std::vector<int32_t> mixed = tok.encode("a<|im_start|>b");
    ++g_checks;
    if (mixed.size() != 3 || mixed[1] != start) {
        fail(__FILE__, __LINE__,
             "спецтокен в середине текста не опознан как отдельный id");
    }
    check_text(tok.decode(mixed), "a<|im_start|>b", "decode mid-special",
               __LINE__);
    check_text(tok.decode({start}), "<|im_start|>", "decode({start})",
               __LINE__);
    check_ids(tok.encode("<|im|>"), {im}, "encode(<|im|>)", __LINE__);

    // Без special-режима те же символы идут через обычный BPE, но
    // round-trip обязан сохраниться.
    vtm::EncodeOptions opts;
    opts.allow_special_tokens = false;
    const std::vector<int32_t> plain = tok.encode("<|im_start|>", opts);
    ++g_checks;
    if (plain.size() <= 1) {
        fail(__FILE__, __LINE__,
             "при allow_special_tokens=false спецтокен должен разбираться BPE");
    }
    check_text(tok.decode(plain), "<|im_start|>", "decode без special",
               __LINE__);
}

void test_roundtrips(const vtm::Tokenizer &tok) {
    std::printf("== round-trip ==\n");
    const char *const ascii[] = {
        "",
        "hello world",
        "The quick brown fox jumps over 13 lazy dogs!",
        "  double  spaces  ",
        "line\nbreak\ttab\r\n",
        "quotes: \"double\" and 'single', apostrophe's",
        "back\\slash / slash \\n not a newline",
        "hello, world!",
        "0123456789",
        "()[]{}<>!?.,;:#@$%^&*_-+=~`|",
    };
    for (const char *s : ascii) ROUNDTRIP(tok, s, "ASCII");

    const char *const cyrillic[] = {
        "Привет, мир!",
        "Ёжик в тумане — тест",
        "Съешь же ещё этих мягких французских булок, да выпей чаю.",
        "Привет  мир",
        "МОСКВА",
        "Яя Ёё Йй Ъъ Ыы Ьь Ээ Юю",
    };
    for (const char *s : cyrillic) ROUNDTRIP(tok, s, "кириллица");

    const char *const emoji[] = {
        "🙂",
        "🙂🚀",
        "Привет 🙂 мир 🚀!",
        "👨‍👩‍👧‍👦",
        "❤️",
        "🇷🇺",
        "🙂🙂🙂",
    };
    for (const char *s : emoji) ROUNDTRIP(tok, s, "эмодзи");

    const char *const other[] = {
        "Hello, мир! 🙂 123",
        "test\nПривет\t🙂",
        "naïve café",
        "«ёлки» — тире",
        "日本語テスト",
        "한국어 테스트",
        "العربية",
        "עברית",
        "हिन्दी",
        "a<|im_start|>b<|im_end|>c",
        "<|im_start|>Привет 🙂<|im_end|>",
    };
    for (const char *s : other) ROUNDTRIP(tok, s, "смесь");

    // Многобайтовые символы, разрезанные на байтовые токены, склеиваются
    // обратно в целый символ (encode не рвёт UTF-8 посимвольно).
    const std::vector<int32_t> ids = tok.encode("é");
    ++g_checks;
    if (ids.size() < 2) {
        fail(__FILE__, __LINE__,
             "encode(é) должен дать минимум 2 байтовых токена");
    }
    check_text(tok.decode(ids), "é", "decode(é)", __LINE__);
}

void test_decode_errors(const vtm::Tokenizer &tok) {
    std::printf("== decode: ошибки ==\n");
    check_text(tok.decode({}), "", "decode({})", __LINE__);

    bool threw = false;
    try {
        (void)tok.decode({999999});
    } catch (const vtm::TokenizerError &e) {
        threw = !std::string(e.what()).empty();
    }
    CHECK_MSG(threw, "decode(999999) должен бросить TokenizerError");

    threw = false;
    try {
        (void)tok.decode({-1});
    } catch (const vtm::TokenizerError &e) {
        threw = !std::string(e.what()).empty();
    }
    CHECK_MSG(threw, "decode(-1) должен бросить TokenizerError");

    threw = false;
    try {
        (void)tok.id_to_token(424242);
    } catch (const vtm::TokenizerError &e) {
        threw = !std::string(e.what()).empty();
    }
    CHECK_MSG(threw, "id_to_token(424242) должен бросить TokenizerError");
}

void test_broken_json() {
    std::printf("== битый JSON ==\n");
    std::string message;

    CHECK(load_throws("{", "broken.json", &message));
    CHECK_MSG(!message.empty(), "сообщение об ошибке пустое");

    CHECK(load_throws("", "empty.json", &message));
    CHECK(!message.empty());

    CHECK(load_throws("{\"model\": ", "truncated.json", &message));
    CHECK(!message.empty());

    CHECK(load_throws("{\"model\":{\"type\":\"Unigram\",\"vocab\":{}}}",
                      "unigram.json", &message));
    CHECK(message.find("Unigram") != std::string::npos ||
          message.find("BPE") != std::string::npos);

    CHECK(load_throws("{\"model\":{\"type\":\"BPE\"}}", "no_vocab.json",
                      &message));
    CHECK(!message.empty());

    CHECK(load_throws(
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[\"a\"]}}",
        "bad_merge.json", &message));
    CHECK(!message.empty());

    CHECK(load_throws(
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]},"
        "\"added_tokens\":[{\"id\":5}]}",
        "bad_added.json", &message));
    CHECK(!message.empty());

    // Защита от гигантских id (иначе resize по испорченному файлу).
    CHECK(load_throws(
        R"json({"model":{"type":"BPE","vocab":{"a":99999999}}})json",
        "huge_id.json", &message));
    CHECK(!message.empty());
    CHECK(load_throws(
        R"json({"model":{"type":"BPE","vocab":{"a":0}},"added_tokens":[{"id":99999999,"content":"<x>","special":true}]})json",
        "huge_added_id.json", &message));
    CHECK(!message.empty());

    // Нет файла — тоже понятная ошибка, а не падение.
    CHECK(load_file_throws("/tmp/vtm-tokenizer-no-such-file-42.json", &message));
    CHECK(!message.empty());

    // Файл с мусором и файл, обрезанный посередине.
    const std::string garbage_path =
        "test_tokenizer_garbage_" + std::to_string(::getpid()) + ".json";
    {
        std::ofstream out(garbage_path, std::ios::binary);
        out << "not json at all\n";
    }
    CHECK(load_file_throws(garbage_path, &message));
    CHECK(!message.empty());
    std::remove(garbage_path.c_str());

    const std::string fixture = read_file(g_fixture_path);
    CHECK_MSG(!fixture.empty(), "фикстура не прочитана: " + g_fixture_path);
    const std::string truncated_path =
        "test_tokenizer_truncated_" + std::to_string(::getpid()) + ".json";
    {
        std::ofstream out(truncated_path, std::ios::binary);
        out << fixture.substr(0, fixture.size() / 2);
    }
    CHECK(load_file_throws(truncated_path, &message));
    CHECK_MSG(!message.empty(), "нет сообщения для обрезанного файла");
    std::printf("  обрезанный tokenizer.json -> %s\n", message.c_str());
    std::remove(truncated_path.c_str());
}

void test_mini_json() {
    std::printf("== разбор вариантов tokenizer.json ==\n");
    // merges массивом пар, added_tokens_decoder, Lowercase + Replace.
    const std::string normalizer =
        "{\"type\":\"Sequence\",\"normalizers\":[{\"type\":\"Lowercase\"},"
        "{\"type\":\"Replace\",\"pattern\":{\"String\":\"xx\"},"
        "\"content\":\"ab\"}]}";
    const std::string added =
        "[{\"id\":10,\"content\":\"<e>\",\"special\":true}]";
    const std::string json = mini_tokenizer_json(
        ",\"added_tokens_decoder\":{\"11\":{\"content\":\"<f>\","
        "\"special\":true}}",
        normalizer, added);
    vtm::Tokenizer tok = vtm::Tokenizer::from_json(json, "mini.json");
    CHECK(tok.info().vocab_size == 4);
    CHECK(tok.info().merge_count == 1);
    CHECK(tok.info().added_token_count == 2);
    CHECK(tok.info().special_token_count == 2);
    CHECK(tok.token_to_id("<f>", nullptr));

    // Lowercase приводит AB -> ab, Replace xx -> ab, затем мерж [a,b].
    check_ids(tok.encode("AB"), {2}, "mini: Lowercase+merge", __LINE__);
    check_ids(tok.encode("xx"), {2}, "mini: Replace+merge", __LINE__);
    check_text(tok.decode({2}), "ab", "mini: decode ab", __LINE__);
    // <e> и <f> пришли соответственно из added_tokens и *_decoder.
    check_ids(tok.encode("<e><f>"), {10, 11}, "mini: specials", __LINE__);
    check_text(tok.decode({10, 11}), "<e><f>", "mini: decode specials",
               __LINE__);

    // Нормализатор NFC не применяется, но токенизатор загружается и
    // честно сообщает об этом в warnings.
    const std::string nfc_json = mini_tokenizer_json(
        "", "{\"type\":\"NFKC\"}", added);
    vtm::Tokenizer nfc = vtm::Tokenizer::from_json(nfc_json, "nfc.json");
    bool noted = false;
    for (const std::string &w : nfc.info().warnings) {
        if (w.find("NFKC") != std::string::npos) noted = true;
    }
    CHECK_MSG(noted, "предупреждение о NFKC отсутствует");
    check_text(nfc.decode(nfc.encode("ab")), "ab", "nfc: round-trip ab",
               __LINE__);

    // Спецтокен не должен нормализоваться: Lowercase не превращает <E> в <e>,
    // иначе added_token перестал бы опознаваться.
    const std::string upper_added =
        "[{\"id\":10,\"content\":\"<E>\",\"special\":true}]";
    const std::string upper_json = mini_tokenizer_json(
        "", "{\"type\":\"Lowercase\"}", upper_added);
    vtm::Tokenizer upper = vtm::Tokenizer::from_json(upper_json, "upper.json");
    int32_t upper_id = -1;
    CHECK(upper.token_to_id("<E>", &upper_id));
    CHECK(upper_id == 10);
    check_ids(upper.encode("<E>"), {10}, "special не нормализуется", __LINE__);
    check_text(upper.decode({10}), "<E>", "decode special <E>", __LINE__);
    check_text(upper.decode(upper.encode("AB")), "ab",
               "Lowercase на обычном тексте", __LINE__);

    // Сломанное правило мержа.
    std::string message;
    CHECK(load_throws(
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},"
        "\"merges\":[123]}}",
        "bad_merge_type.json", &message));
    CHECK(!message.empty());
}

} // namespace

int main(int argc, char **argv) {
    std::string path;
    if (argc > 1) {
        path = argv[1];
    } else {
        const char *const candidates[] = {
            "../../tests/vtm/fixtures/tokenizer.json",
            "tests/vtm/fixtures/tokenizer.json",
        };
        for (const char *c : candidates) {
            std::ifstream in(c, std::ios::binary);
            if (in.good()) {
                path = c;
                break;
            }
        }
    }
    if (path.empty()) {
        std::fprintf(stderr, "не найден tokenizer.json (аргумент не задан)\n");
        return 2;
    }
    g_fixture_path = path;

    std::printf("BPE tokenizer tests\nfixture: %s\n\n", path.c_str());
    vtm::Tokenizer tok;
    try {
        tok = vtm::Tokenizer::load(path);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "не удалось загрузить фикстуру: %s\n", e.what());
        return 2;
    }

    test_info(tok);
    test_known_merges(tok);
    test_specials(tok);
    test_roundtrips(tok);
    test_decode_errors(tok);
    test_broken_json();
    test_mini_json();

    std::printf("\n%s: %d проверок, %d провалов\n",
                g_fails == 0 ? "OK" : "FAILED", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
