// VTM: BPE-токенизатор (C++17, без внешних зависимостей).
//
// Внутри — свой маленький JSON-парсер (tokenizer.json бывает на десятки МБ),
// таблица byte-level GPT-2 (байт <-> unicode-символ), GPT-2 regex-разбиение
// перед BPE, поиск added_tokens по подстроке и BPE по рангам мержей.
#include "tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string_view>
#include <utility>

#include <unistd.h>

namespace vtm {
namespace {

// ---------------------------------------------------------------------------
// UTF-8
// ---------------------------------------------------------------------------

constexpr uint32_t kReplacement = 0xFFFD;

// Потолок id: защита от гигантских resize на испорченном файле.
constexpr std::size_t kMaxVocabId = 1u << 20;  // 1 048 576: хватает любым BPE-словарям

void utf8_append(std::string &out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// Строгий разбор одного UTF-8 символа. false — байт не начинает корректную
// последовательность; вызывающий обязан сдвинуться на 1 байт.
bool utf8_decode(const std::string &s, std::size_t i, uint32_t *cp, std::size_t *len) {
    const std::size_t n = s.size();
    if (i >= n) return false;
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        *cp = c;
        *len = 1;
        return true;
    }
    std::size_t need;
    uint32_t v;
    if ((c & 0xE0) == 0xC0) {
        need = 1;
        v = c & 0x1Fu;
        if (v < 2) return false;  // overlong (C0/C1)
    } else if ((c & 0xF0) == 0xE0) {
        need = 2;
        v = c & 0x0Fu;
    } else if ((c & 0xF8) == 0xF0) {
        need = 3;
        v = c & 0x07u;
    } else {
        return false;
    }
    if (i + need >= n) return false;
    for (std::size_t k = 1; k <= need; ++k) {
        const unsigned char b = static_cast<unsigned char>(s[i + k]);
        if ((b & 0xC0) != 0x80) return false;
        v = (v << 6) | (b & 0x3Fu);
    }
    if (need == 2 && v < 0x800) return false;
    if (need == 3 && v < 0x10000) return false;
    if (v > 0x10FFFF) return false;
    if (v >= 0xD800 && v <= 0xDFFF) return false;
    *cp = v;
    *len = need + 1;
    return true;
}

// Заменить некорректные последовательности на U+FFFD (decode не должен
// возвращать битый UTF-8 даже на испорченном словаре).
std::string utf8_sanitize(const std::string &bytes) {
    std::string out;
    out.reserve(bytes.size());
    std::size_t i = 0;
    while (i < bytes.size()) {
        uint32_t cp = 0;
        std::size_t len = 0;
        if (utf8_decode(bytes, i, &cp, &len)) {
            out.append(bytes, i, len);
            i += len;
        } else {
            utf8_append(out, kReplacement);
            ++i;
        }
    }
    return out;
}

// Выпустить из bytes завершённые UTF-8 символы в out, оставив в bytes
// недописанный хвост (не больше 3 байт). at_end == true: хвост больше не
// допишется (конец генерации или сменился тип токена), поэтому битые байты
// заменяются на U+FFFD — так ничего не теряется и не рвётся UTF-8.
void utf8_emit_complete(std::string &bytes, std::string &out, bool at_end) {
    std::size_t i = 0;
    while (i < bytes.size()) {
        const unsigned char c = static_cast<unsigned char>(bytes[i]);
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
            ++i;
            continue;
        }
        uint32_t cp = 0;
        std::size_t len = 0;
        if (utf8_decode(bytes, i, &cp, &len)) {
            out.append(bytes, i, len);
            i += len;
            continue;
        }
        if (!at_end) {
            // Сколько байт нужно символу с таким ведущим байтом (0 — байт
            // не может начинать UTF-8 последовательность).
            std::size_t need = 0;
            if (c >= 0xC2 && c <= 0xDF) need = 2;
            else if (c >= 0xE0 && c <= 0xEF) need = 3;
            else if (c >= 0xF0 && c <= 0xF4) need = 4;
            if (need != 0 && bytes.size() - i < need) {
                // Не хватает байт: если все имеющиеся — продолжения, это
                // недописанный символ, ждём следующий токен. Если нет —
                // последовательность уже точно битая, не задерживаем.
                bool plausible = true;
                for (std::size_t k = 1; i + k < bytes.size(); ++k) {
                    if ((static_cast<unsigned char>(bytes[i + k]) & 0xC0) !=
                        0x80) {
                        plausible = false;
                        break;
                    }
                }
                if (plausible) break;
            }
        }
        utf8_append(out, kReplacement);
        ++i;
    }
    if (i != 0) bytes.erase(0, i);
}

// ---------------------------------------------------------------------------
// Классы символов для GPT-2 regex (приближение \p{L}, \p{N}, \s).
// Полные Unicode-таблицы в рантайм не тянем; покрыты основные письменности.
// ---------------------------------------------------------------------------

bool is_space_cp(uint32_t c) {
    return c == 0x09 || c == 0x0A || c == 0x0B || c == 0x0C || c == 0x0D ||
           c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}

bool is_letter_cp(uint32_t c) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return true;
    if (c == 0xAA || c == 0xB5 || c == 0xBA) return true;
    if (c >= 0xC0 && c <= 0x2AF) return true;   // Latin-1 Supplement / Extended
    if (c >= 0x370 && c <= 0x3FF) return true;   // Greek
    if (c >= 0x400 && c <= 0x52F) return true;   // Cyrillic
    if (c >= 0x531 && c <= 0x58F) return true;   // Armenian
    if (c >= 0x5D0 && c <= 0x5EA) return true;   // Hebrew
    if (c >= 0x620 && c <= 0x64A) return true;   // Arabic
    if (c >= 0x900 && c <= 0x97F) return true;   // Devanagari
    if (c >= 0xE01 && c <= 0xE5B) return true;   // Thai
    if (c >= 0x10A0 && c <= 0x10FF) return true; // Georgian
    if (c >= 0x1E00 && c <= 0x1FFF) return true; // Latin/Greek Extended
    if (c >= 0x2C60 && c <= 0x2C7F) return true; // Latin Extended-C
    if (c >= 0x3040 && c <= 0x30FF) return true; // Hiragana/Katakana
    if (c >= 0x3400 && c <= 0x4DBF) return true; // CJK Ext-A
    if (c >= 0x4E00 && c <= 0x9FFF) return true; // CJK
    if (c >= 0xA640 && c <= 0xA69F) return true; // Cyrillic Extended-B
    if (c >= 0xAC00 && c <= 0xD7AF) return true; // Hangul
    if (c >= 0xF900 && c <= 0xFAFF) return true; // CJK Compatibility
    if (c >= 0xFF21 && c <= 0xFF3A) return true; // Fullwidth Latin
    if (c >= 0xFF41 && c <= 0xFF5A) return true;
    if (c >= 0x20000 && c <= 0x2FA1F) return true; // CJK Ext-B+
    return false;
}

bool is_digit_cp(uint32_t c) {
    if (c >= '0' && c <= '9') return true;
    if (c >= 0x660 && c <= 0x669) return true;   // Arabic-Indic
    if (c >= 0x6F0 && c <= 0x6F9) return true;
    if (c >= 0x966 && c <= 0x96F) return true;   // Devanagari
    if (c >= 0xFF10 && c <= 0xFF19) return true; // Fullwidth
    return false;
}

// ---------------------------------------------------------------------------
// Byte-level таблица GPT-2: байт -> unicode-символ и обратно.
// ---------------------------------------------------------------------------

struct ByteTables {
    std::array<std::string, 256> enc;
    std::unordered_map<uint32_t, unsigned char> dec;

    ByteTables() {
        std::array<bool, 256> used{};
        std::vector<int> bs;
        for (int b = 0x21; b <= 0x7E; ++b) { bs.push_back(b); used[b] = true; }
        for (int b = 0xA1; b <= 0xAC; ++b) { bs.push_back(b); used[b] = true; }
        for (int b = 0xAE; b <= 0xFF; ++b) { bs.push_back(b); used[b] = true; }
        std::vector<int> cs = bs;
        int extra = 0;
        for (int b = 0; b < 256; ++b) {
            if (!used[b]) {
                bs.push_back(b);
                cs.push_back(256 + extra);
                ++extra;
            }
        }
        for (std::size_t k = 0; k < bs.size(); ++k) {
            utf8_append(enc[static_cast<std::size_t>(bs[k])],
                        static_cast<uint32_t>(cs[k]));
            dec[static_cast<uint32_t>(cs[k])] =
                static_cast<unsigned char>(bs[k]);
        }
    }
};

const ByteTables &byte_tables() {
    static const ByteTables tables;
    return tables;
}

std::string merge_key(const std::string &a, const std::string &b) {
    std::string key;
    key.reserve(a.size() + 1 + b.size());
    key.append(a);
    key.push_back('\0');
    key.append(b);
    return key;
}

// ---------------------------------------------------------------------------
// Минимальный JSON-парсер (объекты, массивы, строки, числа, bool, null).
// ---------------------------------------------------------------------------

struct Json {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ };
    Type type = NUL;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    const Json *find(std::string_view key) const {
        if (type != OBJ) return nullptr;
        for (const auto &kv : obj) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
};

class JsonParser {
public:
    JsonParser(const std::string &text, std::string source)
        : s_(text), source_(std::move(source)) {}

    Json parse() {
        skip_ws();
        Json v = value(0);
        skip_ws();
        if (i_ != s_.size()) error("лишние данные после JSON");
        return v;
    }

private:
    const std::string &s_;
    std::string source_;
    std::size_t i_ = 0;

    [[noreturn]] void error(const std::string &msg) const {
        std::size_t line = 1;
        std::size_t col = 1;
        const std::size_t upto = std::min(i_, s_.size());
        for (std::size_t k = 0; k < upto; ++k) {
            if (s_[k] == '\n') {
                ++line;
                col = 1;
            } else {
                ++col;
            }
        }
        throw TokenizerError(source_ + ":" + std::to_string(line) + ":" +
                             std::to_string(col) + ": " + msg);
    }

    void skip_ws() {
        while (i_ < s_.size()) {
            const char c = s_[i_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++i_;
            } else {
                break;
            }
        }
    }

    bool consume(char c) {
        if (i_ < s_.size() && s_[i_] == c) {
            ++i_;
            return true;
        }
        return false;
    }

    void expect(char c) {
        if (!consume(c)) {
            error(std::string("ожидался символ '") + c + "'");
        }
    }

    static int hex_digit(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    uint32_t parse_hex4() {
        if (i_ + 4 > s_.size()) error("обрыв \\u-escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            const int d = hex_digit(s_[i_ + static_cast<std::size_t>(k)]);
            if (d < 0) error("не шестнадцатеричная цифра в \\u-escape");
            v = (v << 4) | static_cast<uint32_t>(d);
        }
        i_ += 4;
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (i_ < s_.size()) {
            const unsigned char c = static_cast<unsigned char>(s_[i_++]);
            if (c == '"') return out;
            if (c == '\\') {
                if (i_ >= s_.size()) error("обрыв escape-последовательности");
                const char e = s_[i_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = parse_hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (i_ + 1 >= s_.size() || s_[i_] != '\\' ||
                                s_[i_ + 1] != 'u') {
                                error("высокий суррогат без пары \\uXXXX");
                            }
                            i_ += 2;
                            const uint32_t lo = parse_hex4();
                            if (lo < 0xDC00 || lo > 0xDFFF) {
                                error("некорректная суррогатная пара");
                            }
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                            error("одиночный низкий суррогат");
                        }
                        utf8_append(out, cp);
                        break;
                    }
                    default:
                        error(std::string("неизвестный escape \\") + e);
                }
            } else if (c < 0x20) {
                error("управляющий символ внутри строки");
            } else {
                out.push_back(static_cast<char>(c));
            }
        }
        error("незакрытая строка");
    }

    Json parse_number() {
        const std::size_t start = i_;
        consume('-');
        if (i_ >= s_.size()) error("обрыв числа");
        if (s_[i_] == '0') {
            ++i_;
        } else if (s_[i_] >= '1' && s_[i_] <= '9') {
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        } else {
            error("ожидалась цифра");
        }
        if (i_ < s_.size() && s_[i_] == '.') {
            ++i_;
            if (i_ >= s_.size() || s_[i_] < '0' || s_[i_] > '9') {
                error("ожидалась цифра после '.'");
            }
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
            if (i_ >= s_.size() || s_[i_] < '0' || s_[i_] > '9') {
                error("ожидалась цифра в экспоненте");
            }
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        }
        const std::string text = s_.substr(start, i_ - start);
        errno = 0;
        char *end = nullptr;
        const double v = std::strtod(text.c_str(), &end);
        if (end == text.c_str() || *end != '\0' || !std::isfinite(v)) {
            error("некорректное число");
        }
        Json j;
        j.type = Json::NUM;
        j.number = v;
        return j;
    }

    void parse_literal(const char *lit) {
        const std::size_t n = std::strlen(lit);
        if (s_.compare(i_, n, lit) != 0) error("некорректный литерал");
        i_ += n;
    }

    Json value(int depth) {
        if (depth > 200) error("слишком глубокая вложенность JSON (> 200)");
        skip_ws();
        if (i_ >= s_.size()) error("неожиданный конец JSON");
        const char c = s_[i_];
        if (c == '{') return object(depth);
        if (c == '[') return array(depth);
        if (c == '"') {
            Json j;
            j.type = Json::STR;
            j.str = parse_string();
            return j;
        }
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        if (c == 't') {
            parse_literal("true");
            Json j;
            j.type = Json::BOOL;
            j.boolean = true;
            return j;
        }
        if (c == 'f') {
            parse_literal("false");
            Json j;
            j.type = Json::BOOL;
            j.boolean = false;
            return j;
        }
        if (c == 'n') {
            parse_literal("null");
            return Json();
        }
        error(std::string("неожиданный символ '") + c + "'");
    }

    Json object(int depth) {
        expect('{');
        Json j;
        j.type = Json::OBJ;
        skip_ws();
        if (consume('}')) return j;
        for (;;) {
            skip_ws();
            if (i_ >= s_.size() || s_[i_] != '"') {
                error("ожидался ключ-строка");
            }
            std::string key = parse_string();
            skip_ws();
            expect(':');
            Json v = value(depth + 1);
            j.obj.emplace_back(std::move(key), std::move(v));
            skip_ws();
            if (consume('}')) return j;
            expect(',');
        }
    }

    Json array(int depth) {
        expect('[');
        Json j;
        j.type = Json::ARR;
        skip_ws();
        if (consume(']')) return j;
        for (;;) {
            j.arr.push_back(value(depth + 1));
            skip_ws();
            if (consume(']')) return j;
            expect(',');
        }
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Сборка Tokenizer из разобранного JSON.
// ---------------------------------------------------------------------------

class TokenizerBuilder {
public:
    static Tokenizer build(const Json &root, const std::string &source) {
        Tokenizer t;
        t.info_.source = source;
        if (root.type != Json::OBJ) fail(source, "корень tokenizer.json — не объект");

        const Json *model = root.find("model");
        if (model == nullptr || model->type != Json::OBJ) {
            fail(source, "нет объекта model");
        }
        const Json *type = model->find("type");
        if (type != nullptr && type->type == Json::STR) {
            t.info_.model_type = type->str;
            if (type->str != "BPE") {
                fail(source, "model.type = \"" + type->str +
                                 "\" не поддерживается (нужен BPE)");
            }
        } else {
            t.info_.model_type = "BPE";
            warn(t, "model.type отсутствует — считаем BPE");
        }

        build_vocab(t, *model, source);
        build_merges(t, *model, source);
        build_added(t, root, source);
        build_model_tokens(t, *model);
        build_normalizer(t, root.find("normalizer"));
        build_pre_tokenizer(t, root.find("pre_tokenizer"));
        build_decoder(t, root.find("decoder"));
        t.info_.byte_level = t.info_.byte_level || t.decoder_byte_level_;

        const Json *ignore_merges = model->find("ignore_merges");
        if (ignore_merges != nullptr && ignore_merges->type == Json::BOOL &&
            ignore_merges->boolean) {
            warn(t, "model.ignore_merges = true — мержи не применяются");
            t.merge_rank_.clear();
            t.info_.merge_count = 0;
        }
        check_byte_alphabet(t);
        return t;
    }

private:
    static void fail(const std::string &source, const std::string &msg) {
        throw TokenizerError(source + ": " + msg);
    }

    static void warn(Tokenizer &t, const std::string &msg) {
        t.info_.warnings.push_back(msg);
    }

    static bool as_int(const Json &j, int64_t *out) {
        if (j.type != Json::NUM) return false;
        if (!std::isfinite(j.number)) return false;
        if (j.number < static_cast<double>(std::numeric_limits<int32_t>::min()) ||
            j.number > static_cast<double>(std::numeric_limits<int32_t>::max())) {
            return false;
        }
        const double rounded = std::floor(j.number);
        if (rounded != j.number) return false;
        *out = static_cast<int64_t>(j.number);
        return true;
    }

    static bool as_bool(const Json *j, bool fallback) {
        if (j == nullptr || j->type != Json::BOOL) return fallback;
        return j->boolean;
    }

    static std::string type_of(const Json *j) {
        if (j == nullptr || j->type == Json::NUL) return "null";
        if (j->type != Json::OBJ) return "?";
        const Json *t = j->find("type");
        if (t == nullptr || t->type != Json::STR) return "?";
        return t->str;
    }

    static void build_vocab(Tokenizer &t, const Json &model, const std::string &source) {
        const Json *vocab = model.find("vocab");
        if (vocab == nullptr || vocab->type != Json::OBJ) {
            fail(source, "model.vocab отсутствует или не объект");
        }
        std::vector<char> seen;
        std::size_t max_id = 0;
        for (const auto &kv : vocab->obj) {
            int64_t id = 0;
            if (!as_int(kv.second, &id) || id < 0) {
                fail(source, "model.vocab[\"" + kv.first +
                                 "\"] должен быть целым >= 0");
            }
            if (static_cast<std::size_t>(id) >= kMaxVocabId) {
                fail(source, "model.vocab[\"" + kv.first + "\"]: id " +
                                 std::to_string(id) + " слишком большой");
            }
            if (kv.first.empty()) {
                fail(source, "model.vocab: пустой токен не поддерживается");
            }
            if (kv.first.find('\0') != std::string::npos) {
                fail(source, "model.vocab: токен с нулевым байтом не поддерживается");
            }
            if (t.token_to_id_.find(kv.first) != t.token_to_id_.end()) {
                fail(source, "дублирующийся токен в model.vocab: \"" + kv.first + "\"");
            }
            const std::size_t uid = static_cast<std::size_t>(id);
            if (uid >= seen.size()) {
                seen.resize(uid + 1, 0);
                t.id_to_token_.resize(uid + 1);
            }
            if (seen[uid]) {
                fail(source, "дублирующийся id " + std::to_string(id) +
                                 " в model.vocab (токен \"" + kv.first + "\")");
            }
            seen[uid] = 1;
            t.id_to_token_[uid] = kv.first;
            t.token_to_id_.emplace(kv.first, static_cast<int32_t>(id));
            max_id = std::max(max_id, uid);
        }
        if (vocab->obj.empty()) fail(source, "model.vocab пуст");
        t.info_.vocab_size = vocab->obj.size();
        if (t.id_to_token_.size() != max_id + 1) t.id_to_token_.resize(max_id + 1);
    }

    static void build_merges(Tokenizer &t, const Json &model, const std::string &source) {
        const Json *merges = model.find("merges");
        if (merges == nullptr) return;
        if (merges->type != Json::ARR) {
            fail(source, "model.merges должен быть массивом");
        }
        std::size_t index = 0;
        for (const Json &entry : merges->arr) {
            std::string a;
            std::string b;
            if (entry.type == Json::STR) {
                const std::string &s = entry.str;
                const std::size_t sp = s.find(' ');
                if (sp == std::string::npos || sp == 0 || sp + 1 >= s.size() ||
                    s.find(' ', sp + 1) != std::string::npos) {
                    fail(source, "model.merges[" + std::to_string(index) +
                                     "]: ожидалось \"a b\"");
                }
                a = s.substr(0, sp);
                b = s.substr(sp + 1);
            } else if (entry.type == Json::ARR && entry.arr.size() == 2 &&
                       entry.arr[0].type == Json::STR && entry.arr[1].type == Json::STR) {
                a = entry.arr[0].str;
                b = entry.arr[1].str;
            } else {
                fail(source, "model.merges[" + std::to_string(index) +
                                 "]: ожидалась строка \"a b\" или пара [a, b]");
            }
            if (a.empty() || b.empty()) {
                fail(source, "model.merges[" + std::to_string(index) +
                                 "]: пустой символ");
            }
            if (a.find('\0') != std::string::npos ||
                b.find('\0') != std::string::npos) {
                fail(source, "model.merges[" + std::to_string(index) +
                                 "]: нулевой байт в символе не поддерживается");
            }
            const std::string merged = a + b;
            if (t.token_to_id_.find(merged) == t.token_to_id_.end()) {
                ++t.info_.merges_missing_from_vocab;
                continue;  // такой мерж всё равно дал бы неизвестный токен
            }
            const std::string key = merge_key(a, b);
            if (t.merge_rank_.find(key) != t.merge_rank_.end()) {
                continue;  // дубликат: первый ранг и есть ранг
            }
            t.merge_rank_.emplace(key, static_cast<int32_t>(index));
            ++t.info_.merge_count;
            ++index;
        }
        if (t.info_.merges_missing_from_vocab != 0) {
            warn(t, "отброшено мержей без результата в vocab: " +
                        std::to_string(t.info_.merges_missing_from_vocab));
        }
    }

    static void add_added_token(Tokenizer &t, const std::string &content,
                                int32_t id, bool special, const std::string &source) {
        if (content.empty()) {
            fail(source, "added_token с пустым content");
        }
        if (id < 0) fail(source, "added_token \"" + content + "\" с id < 0");
        if (static_cast<std::size_t>(id) >= kMaxVocabId) {
            fail(source, "added_token \"" + content + "\": id " +
                             std::to_string(id) + " слишком большой");
        }
        if (content.find('\0') != std::string::npos) {
            fail(source, "added_token с нулевым байтом не поддерживается");
        }
        const std::size_t uid = static_cast<std::size_t>(id);
        if (uid >= t.id_to_token_.size()) {
            t.id_to_token_.resize(uid + 1);
            t.is_added_.resize(uid + 1, 0);
        } else if (t.is_added_.size() < t.id_to_token_.size()) {
            t.is_added_.resize(t.id_to_token_.size(), 0);
        }
        if (!t.id_to_token_[uid].empty() && t.id_to_token_[uid] != content &&
            !t.is_added_[uid]) {
            fail(source, "added_token id=" + std::to_string(id) +
                             " конфликтует с model.vocab (там \"" +
                             t.id_to_token_[uid] + "\")");
        }
        // Один и тот же контент может прийти из added_tokens и
        // added_tokens_decoder — второй раз не добавляем.
        for (const auto &a : t.added_) {
            if (a.id == id && a.content == content) return;
        }
        t.id_to_token_[uid] = content;
        t.is_added_[uid] = 1;
        const auto it = t.token_to_id_.find(content);
        if (it == t.token_to_id_.end()) {
            t.token_to_id_.emplace(content, id);
        } else if (it->second != id) {
            warn(t, "два id для токена \"" + content + "\": " +
                        std::to_string(it->second) + " и " + std::to_string(id));
        }
        t.added_.push_back(Tokenizer::AddedToken{content, id, special});
        ++t.info_.added_token_count;
        if (special) ++t.info_.special_token_count;
    }

    static void build_added(Tokenizer &t, const Json &root, const std::string &source) {
        if (const Json *arr = root.find("added_tokens")) {
            if (arr->type != Json::ARR) {
                fail(source, "added_tokens должен быть массивом");
            }
            std::size_t i = 0;
            for (const Json &entry : arr->arr) {
                if (entry.type != Json::OBJ) {
                    fail(source, "added_tokens[" + std::to_string(i) +
                                     "] должен быть объектом");
                }
                const Json *content = entry.find("content");
                const Json *idj = entry.find("id");
                if (content == nullptr || content->type != Json::STR) {
                    fail(source, "added_tokens[" + std::to_string(i) +
                                     "].content отсутствует или не строка");
                }
                int64_t id = 0;
                if (idj == nullptr || !as_int(*idj, &id)) {
                    fail(source, "added_tokens[" + std::to_string(i) +
                                     "].id отсутствует или не целое");
                }
                add_added_token(t, content->str, static_cast<int32_t>(id),
                                as_bool(entry.find("special"), false), source);
                ++i;
            }
        }
        // Старый/альтернативный вариант: added_tokens_decoder {"id": {...}}.
        if (const Json *dec = root.find("added_tokens_decoder")) {
            if (dec->type != Json::OBJ) {
                fail(source, "added_tokens_decoder должен быть объектом");
            }
            for (const auto &kv : dec->obj) {
                if (kv.second.type != Json::OBJ) {
                    fail(source, "added_tokens_decoder[\"" + kv.first +
                                     "\"] должен быть объектом");
                }
                const Json *content = kv.second.find("content");
                if (content == nullptr || content->type != Json::STR) {
                    fail(source, "added_tokens_decoder[\"" + kv.first +
                                     "\"].content отсутствует или не строка");
                }
                const char *begin = kv.first.c_str();
                char *end = nullptr;
                errno = 0;
                const long long parsed = std::strtoll(begin, &end, 10);
                if (end == begin || *end != '\0' || errno != 0 ||
                    parsed < 0 ||
                    parsed > std::numeric_limits<int32_t>::max()) {
                    fail(source, "added_tokens_decoder: ключ \"" + kv.first +
                                     "\" не целое >= 0");
                }
                add_added_token(t, content->str, static_cast<int32_t>(parsed),
                                as_bool(kv.second.find("special"), false), source);
            }
        }
    }

    static int32_t resolve_model_token(Tokenizer &t, const Json *j,
                                       const std::string &field) {
        if (j == nullptr || j->type == Json::NUL) return -1;
        if (j->type == Json::STR) {
            const auto it = t.token_to_id_.find(j->str);
            if (it == t.token_to_id_.end()) {
                warn(t, "model." + field + " = \"" + j->str +
                            "\" отсутствует в словаре");
                return -1;
            }
            return it->second;
        }
        if (j->type == Json::NUM) {
            int64_t id = 0;
            if (as_int(*j, &id) && id >= 0 &&
                static_cast<std::size_t>(id) < t.id_to_token_.size()) {
                return static_cast<int32_t>(id);
            }
            warn(t, "model." + field + " — некорректный id");
            return -1;
        }
        warn(t, "model." + field + " — не строка и не число");
        return -1;
    }

    static void build_model_tokens(Tokenizer &t, const Json &model) {
        const Json *unk = model.find("unk_token");
        const Json *bos = model.find("bos_token");
        const Json *eos = model.find("eos_token");
        const Json *pad = model.find("pad_token");
        if (unk != nullptr && unk->type == Json::STR) t.info_.unk_token = unk->str;
        if (bos != nullptr && bos->type == Json::STR) t.info_.bos_token = bos->str;
        if (eos != nullptr && eos->type == Json::STR) t.info_.eos_token = eos->str;
        if (pad != nullptr && pad->type == Json::STR) t.info_.pad_token = pad->str;
        t.info_.unk_id = resolve_model_token(t, unk, "unk_token");
        t.info_.bos_id = resolve_model_token(t, bos, "bos_token");
        t.info_.eos_id = resolve_model_token(t, eos, "eos_token");
        t.info_.pad_id = resolve_model_token(t, pad, "pad_token");
    }

    // --- normalizer -------------------------------------------------------

    struct NormalizerParse {
        std::string description;
        bool lossy = false;
    };

    static NormalizerParse parse_normalizer(Tokenizer &t, const Json *j,
                                            bool topLevel) {
        NormalizerParse res;
        if (j == nullptr || j->type == Json::NUL) {
            res.description = "null";
            return res;
        }
        if (j->type != Json::OBJ) {
            res.description = "?";
            warn(t, "normalizer: ожидался объект — игнорируем");
            res.lossy = true;
            return res;
        }
        const std::string type = type_of(j);
        if (type == "Sequence") {
            const Json *children = j->find("normalizers");
            std::string desc = "Sequence(";
            if (children != nullptr && children->type == Json::ARR) {
                bool first = true;
                for (const Json &child : children->arr) {
                    const NormalizerParse sub = parse_normalizer(t, &child, false);
                    if (!first) desc += ",";
                    desc += sub.description;
                    first = false;
                    res.lossy = res.lossy || sub.lossy;
                }
            }
            desc += ")";
            res.description = desc;
            return res;
        }
        if (type == "Replace") {
            const Json *pattern = j->find("pattern");
            const Json *content = j->find("content");
            if (pattern != nullptr && pattern->type == Json::OBJ &&
                pattern->find("String") != nullptr &&
                pattern->find("String")->type == Json::STR) {
                Tokenizer::NormalizerOp op;
                op.kind = Tokenizer::NormalizerOp::Replace;
                op.a = pattern->find("String")->str;
                if (content != nullptr && content->type == Json::STR) {
                    op.b = content->str;
                }
                if (op.a.empty()) {
                    warn(t, "normalizer Replace с пустым pattern — пропущен");
                } else {
                    t.normalizer_ops_.push_back(std::move(op));
                }
                res.description = "Replace";
                return res;
            }
            warn(t, "normalizer Replace поддерживается только с pattern.String");
            res.description = "Replace(regex?)";
            res.lossy = true;
            return res;
        }
        if (type == "Prepend") {
            std::string text;
            if (const Json *v = j->find("prepend")) {
                if (v->type == Json::STR) text = v->str;
            }
            if (!text.empty()) {
                Tokenizer::NormalizerOp op;
                op.kind = Tokenizer::NormalizerOp::Prepend;
                op.b = text;
                t.normalizer_ops_.push_back(std::move(op));
            }
            res.description = "Prepend";
            return res;
        }
        if (type == "Strip") {
            Tokenizer::NormalizerOp op;
            op.kind = Tokenizer::NormalizerOp::Strip;
            t.normalizer_ops_.push_back(op);
            res.description = "Strip";
            return res;
        }
        if (type == "Lowercase") {
            Tokenizer::NormalizerOp op;
            op.kind = Tokenizer::NormalizerOp::Lowercase;
            t.normalizer_ops_.push_back(op);
            warn(t, "normalizer Lowercase реализован частично: ASCII, Latin-1, "
                    "кириллица, греческий; остальные письменности без изменений");
            res.description = "Lowercase";
            return res;
        }
        if (type == "NFC" || type == "NFD" || type == "NFKC" || type == "NFKD") {
            warn(t, "normalizer " + type +
                        " не применяется: Unicode-таблиц нормализации в рантайме нет");
            res.description = type + "(не применяется)";
            res.lossy = true;
            return res;
        }
        warn(t, "normalizer \"" + type + "\" не поддерживается — игнорируем");
        res.description = type;
        res.lossy = true;
        (void)topLevel;
        return res;
    }

    static void build_normalizer(Tokenizer &t, const Json *j) {
        const NormalizerParse res = parse_normalizer(t, j, true);
        t.info_.normalizer_type = res.description;
    }

    // --- pre_tokenizer / decoder -----------------------------------------

    struct PreParse {
        std::string description;
        bool byte_level = false;
        bool use_regex = false;  // нет pre_tokenizer — нет и regex-разбиения
        bool add_prefix_space = false;
        bool unsupported = false;
    };

    static PreParse parse_pre_tokenizer(Tokenizer &t, const Json *j) {
        PreParse res;
        if (j == nullptr || j->type == Json::NUL) {
            res.description = "null";
            return res;
        }
        if (j->type != Json::OBJ) {
            res.description = "?";
            warn(t, "pre_tokenizer: ожидался объект — игнорируем");
            res.unsupported = true;
            return res;
        }
        const std::string type = type_of(j);
        if (type == "ByteLevel") {
            res.description = "ByteLevel";
            res.byte_level = true;
            res.use_regex = as_bool(j->find("use_regex"), true);
            res.add_prefix_space = as_bool(j->find("add_prefix_space"), false);
            return res;
        }
        if (type == "Sequence") {
            res.description = "Sequence(";
            const Json *children = j->find("pretokenizers");
            bool first = true;
            if (children != nullptr && children->type == Json::ARR) {
                for (const Json &child : children->arr) {
                    const PreParse sub = parse_pre_tokenizer(t, &child);
                    if (!first) res.description += ",";
                    res.description += sub.description;
                    first = false;
                    res.byte_level = res.byte_level || sub.byte_level;
                    if (sub.byte_level) {
                        res.use_regex = sub.use_regex;
                        res.add_prefix_space = sub.add_prefix_space;
                    }
                    res.unsupported = res.unsupported || sub.unsupported;
                }
            }
            res.description += ")";
            return res;
        }
        if (type == "Split") {
            warn(t, "pre_tokenizer Split не поддерживается: будет применён "
                    "стандартный GPT-2 regex ByteLevel");
            res.description = "Split(не поддерживается)";
            res.unsupported = true;
            return res;
        }
        warn(t, "pre_tokenizer \"" + type + "\" не поддерживается");
        res.description = type;
        res.unsupported = true;
        return res;
    }

    static void build_pre_tokenizer(Tokenizer &t, const Json *j) {
        const PreParse res = parse_pre_tokenizer(t, j);
        t.info_.pre_tokenizer_type = res.description;
        t.info_.byte_level = res.byte_level;
        t.info_.use_regex = res.use_regex;
        t.info_.add_prefix_space = res.add_prefix_space;
        if (j != nullptr && j->type != Json::NUL && !res.byte_level) {
            warn(t, "pre_tokenizer без ByteLevel: токены могут не быть "
                    "byte-level (GPT-2/Qwen)");
        }
    }

    static void build_decoder(Tokenizer &t, const Json *j) {
        if (j == nullptr || j->type == Json::NUL) {
            t.info_.decoder_type = "null";
            if (t.info_.byte_level) {
                warn(t, "decoder отсутствует: decode вернёт byte-level символы "
                        "без обратного отображения в байты");
            }
            return;
        }
        if (j->type != Json::OBJ) {
            t.info_.decoder_type = "?";
            warn(t, "decoder: ожидался объект — игнорируем");
            return;
        }
        const std::string type = type_of(j);
        if (type == "ByteLevel") {
            t.info_.decoder_type = "ByteLevel";
            t.decoder_byte_level_ = true;
            return;
        }
        if (type == "Sequence") {
            t.info_.decoder_type = "Sequence(";
            const Json *children = j->find("decoders");
            bool first = true;
            if (children != nullptr && children->type == Json::ARR) {
                for (const Json &child : children->arr) {
                    const std::string sub = type_of(&child);
                    if (!first) t.info_.decoder_type += ",";
                    t.info_.decoder_type += sub;
                    first = false;
                    if (sub == "ByteLevel") t.decoder_byte_level_ = true;
                    else warn(t, "decoder \"" + sub + "\" не поддерживается");
                }
            }
            t.info_.decoder_type += ")";
            if (!t.decoder_byte_level_) {
                warn(t, "в decoder нет ByteLevel: decode склеит токены как есть");
            }
            return;
        }
        warn(t, "decoder \"" + type + "\" не поддерживается: decode склеит "
                "токены как есть");
        t.info_.decoder_type = type;
    }

    static void check_byte_alphabet(Tokenizer &t) {
        const ByteTables &tables = byte_tables();
        std::size_t have = 0;
        for (std::size_t b = 0; b < 256; ++b) {
            if (t.token_to_id_.find(tables.enc[b]) != t.token_to_id_.end()) ++have;
        }
        t.info_.byte_alphabet_size = have;
        if (have != 256) {
            warn(t, "byte-level словарь неполный: " + std::to_string(have) +
                        "/256 базовых байтовых токенов");
        }
    }
};

// ---------------------------------------------------------------------------
// Tokenizer: загрузка
// ---------------------------------------------------------------------------

bool file_exists(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    return in.good();
}

std::string read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw TokenizerError("не удалось открыть tokenizer.json: " + path +
                             " (" + std::strerror(errno) + ")");
    }
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0) {
        throw TokenizerError("не удалось определить размер файла: " + path);
    }
    constexpr std::streamoff kMax = static_cast<std::streamoff>(1) << 30;  // 1 ГиБ
    if (size > kMax) {
        throw TokenizerError("tokenizer.json слишком большой (> 1 ГиБ): " + path);
    }
    in.seekg(0, std::ios::beg);
    std::string data(static_cast<std::size_t>(size), '\0');
    if (size > 0) {
        in.read(&data[0], size);
        if (!in) {
            throw TokenizerError("ошибка чтения tokenizer.json: " + path);
        }
    }
    return data;
}

Tokenizer Tokenizer::from_json(const std::string &json, const std::string &source) {
    JsonParser parser(json, source);
    const Json root = parser.parse();
    return TokenizerBuilder::build(root, source);
}

Tokenizer Tokenizer::load(const std::string &path) {
    return from_json(read_file(path), path);
}

std::string Tokenizer::default_path() {
    const char *env = std::getenv("VTM_TOKENIZER");
    if (env != nullptr && *env != '\0' && file_exists(env)) return env;

    char cwd_buf[4096];
    std::string cwd;
    if (::getcwd(cwd_buf, sizeof(cwd_buf)) != nullptr) cwd = cwd_buf;

    // От текущего каталога и вверх: <dir>/tests/vtm/fixtures/tokenizer.json.
    if (!cwd.empty()) {
        std::string dir = cwd;
        for (;;) {
            const std::string candidate =
                dir + "/tests/vtm/fixtures/tokenizer.json";
            if (file_exists(candidate)) return candidate;
            if (dir == "/" || dir.empty()) break;
            const std::size_t slash = dir.rfind('/');
            if (slash == std::string::npos) break;
            if (slash == 0) {
                dir = "/";
            } else {
                dir.resize(slash);
            }
        }
        const std::string local = cwd + "/tokenizer.json";
        if (file_exists(local)) return local;
    }
    if (file_exists("tokenizer.json")) return "tokenizer.json";
    if (file_exists("tests/vtm/fixtures/tokenizer.json")) {
        return "tests/vtm/fixtures/tokenizer.json";
    }
    return std::string();
}

// ---------------------------------------------------------------------------
// Tokenizer: normalizer
// ---------------------------------------------------------------------------

bool ascii_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
}

uint32_t lower_cp(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return c + 0x20;
    if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return c + 0x20;
    if (c == 0x178) return 0xFF;
    if (c >= 0x400 && c <= 0x40F) return c + 0x50;
    if (c >= 0x410 && c <= 0x42F) return c + 0x20;
    if (c >= 0x391 && c <= 0x3A9 && c != 0x3A2) return c + 0x20;
    if (c == 0x386) return 0x3AC;
    return c;
}

std::string trim_spaces(const std::string &text) {
    std::size_t begin = 0;
    while (begin < text.size() && ascii_space(text[begin])) ++begin;
    std::size_t end = text.size();
    while (end > begin && ascii_space(text[end - 1])) --end;
    return text.substr(begin, end - begin);
}

std::string replace_all(const std::string &text, const std::string &from,
                        const std::string &to) {
    if (from.empty()) return text;
    std::string out;
    std::size_t pos = 0;
    for (;;) {
        const std::size_t hit = text.find(from, pos);
        if (hit == std::string::npos) {
            out.append(text, pos, std::string::npos);
            break;
        }
        out.append(text, pos, hit - pos);
        out.append(to);
        pos = hit + from.size();
    }
    return out;
}

std::string lowercase_utf8(const std::string &text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        uint32_t cp = 0;
        std::size_t len = 0;
        if (!utf8_decode(text, i, &cp, &len)) {
            out.push_back(text[i]);
            ++i;
            continue;
        }
        utf8_append(out, lower_cp(cp));
        i += len;
    }
    return out;
}

std::string Tokenizer::normalize(const std::string &text) const {
    std::string out = text;
    for (const NormalizerOp &op : normalizer_ops_) {
        switch (op.kind) {
            case NormalizerOp::Replace:
                out = replace_all(out, op.a, op.b);
                break;
            case NormalizerOp::Prepend:
                out = op.b + out;
                break;
            case NormalizerOp::Strip:
                out = trim_spaces(out);
                break;
            case NormalizerOp::Lowercase:
                out = lowercase_utf8(out);
                break;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// GPT-2 regex-разбиение (ByteLevel pre_tokenizer)
// ---------------------------------------------------------------------------

std::vector<std::string> Tokenizer::pre_tokenize(const std::string &text) const {
    std::vector<std::string> pieces;
    const std::size_t n = text.size();
    std::size_t i = 0;

    auto decode_at = [&](std::size_t pos, uint32_t *cp, std::size_t *len) {
        return utf8_decode(text, pos, cp, len);
    };

    while (i < n) {
        uint32_t cp = 0;
        std::size_t len = 0;
        const bool ok = decode_at(i, &cp, &len);

        // 's|'t|'re|'ve|'m|'ll|'d
        if (ok && cp == '\'') {
            static const char *const kSuffixes[] = {"s", "t", "re", "ve", "m",
                                                    "ll", "d"};
            bool matched = false;
            for (const char *suffix : kSuffixes) {
                const std::size_t sl = std::strlen(suffix);
                if (i + 1 + sl <= n && text.compare(i + 1, sl, suffix) == 0) {
                    pieces.push_back(text.substr(i, 1 + sl));
                    i += 1 + sl;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }

        const bool lead_space = ok && cp == ' ';
        const std::size_t start_after_space = lead_space ? i + len : i;

        auto consume_run = [&](std::size_t pos,
                               bool (*pred)(uint32_t)) -> std::size_t {
            while (pos < n) {
                uint32_t c = 0;
                std::size_t l = 0;
                if (!decode_at(pos, &c, &l) || !pred(c)) break;
                pos += l;
            }
            return pos;
        };

        if (start_after_space < n) {
            uint32_t nc = 0;
            std::size_t nl = 0;
            const bool next_ok = decode_at(start_after_space, &nc, &nl);
            const bool next_letter = next_ok && is_letter_cp(nc);
            const bool next_digit = next_ok && is_digit_cp(nc);
            const bool next_other = next_ok && !is_space_cp(nc) &&
                                    !is_letter_cp(nc) && !is_digit_cp(nc);
            if (next_letter || next_digit || next_other) {
                std::size_t end;
                if (next_letter) {
                    end = consume_run(start_after_space, is_letter_cp);
                } else if (next_digit) {
                    end = consume_run(start_after_space, is_digit_cp);
                } else {
                    end = consume_run(start_after_space, [](uint32_t c) {
                        return !is_space_cp(c) && !is_letter_cp(c) &&
                               !is_digit_cp(c);
                    });
                }
                pieces.push_back(text.substr(i, end - i));
                i = end;
                continue;
            }
        }

        if (!lead_space && ok) {
            if (is_letter_cp(cp)) {
                const std::size_t end = consume_run(i, is_letter_cp);
                pieces.push_back(text.substr(i, end - i));
                i = end;
                continue;
            }
            if (is_digit_cp(cp)) {
                const std::size_t end = consume_run(i, is_digit_cp);
                pieces.push_back(text.substr(i, end - i));
                i = end;
                continue;
            }
            if (!is_space_cp(cp)) {
                const std::size_t end = consume_run(i, [](uint32_t c) {
                    return !is_space_cp(c) && !is_letter_cp(c) &&
                           !is_digit_cp(c);
                });
                pieces.push_back(text.substr(i, end - i));
                i = end;
                continue;
            }
        }

        if (ok && is_space_cp(cp)) {
            std::size_t k = i;
            std::size_t last_start = i;
            while (k < n) {
                uint32_t c = 0;
                std::size_t l = 0;
                if (!decode_at(k, &c, &l) || !is_space_cp(c)) break;
                last_start = k;
                k += l;
            }
            if (k < n && last_start > i) {
                // \s+(?!\S): всё, кроме последнего пробела; он уйдёт
                // к следующему слову через " ?" альтернативу.
                pieces.push_back(text.substr(i, last_start - i));
                i = last_start;
            } else {
                pieces.push_back(text.substr(i, k - i));
                i = k;
            }
            continue;
        }

        // Некорректный UTF-8: трактуем байт как «прочее».
        {
            std::size_t k = i;
            while (k < n) {
                uint32_t c = 0;
                std::size_t l = 0;
                if (decode_at(k, &c, &l) &&
                    (is_space_cp(c) || is_letter_cp(c) || is_digit_cp(c))) {
                    break;
                }
                if (!decode_at(k, &c, &l)) {
                    ++k;
                } else {
                    k += l;
                }
            }
            pieces.push_back(text.substr(i, k - i));
            i = k;
        }
    }
    return pieces;
}

// ---------------------------------------------------------------------------
// Tokenizer: BPE и encode
// ---------------------------------------------------------------------------

std::vector<int32_t> Tokenizer::bpe(const std::string &piece) const {
    const ByteTables &tables = byte_tables();
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (const unsigned char b : piece) {
        symbols.push_back(tables.enc[b]);
    }

    while (symbols.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        std::size_t best_i = symbols.size();
        std::string best_a;
        std::string best_b;
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it = merge_rank_.find(merge_key(symbols[i], symbols[i + 1]));
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
                best_a = symbols[i];
                best_b = symbols[i + 1];
            }
        }
        if (best_i == symbols.size()) break;

        std::vector<std::string> next;
        next.reserve(symbols.size());
        for (std::size_t i = 0; i < symbols.size();) {
            if (i + 1 < symbols.size() && symbols[i] == best_a &&
                symbols[i + 1] == best_b) {
                next.push_back(best_a + best_b);
                i += 2;
            } else {
                next.push_back(symbols[i]);
                ++i;
            }
        }
        symbols.swap(next);
    }

    std::vector<int32_t> ids;
    ids.reserve(symbols.size());
    for (const std::string &sym : symbols) {
        const auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
            continue;
        }
        if (info_.unk_id >= 0) {
            ids.push_back(info_.unk_id);
            continue;
        }
        std::string bytes;
        for (unsigned char c : sym) {
            if (c >= 0x20 && c < 0x7F) {
                bytes.push_back(static_cast<char>(c));
            } else {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\x%02X", c);
                bytes += buf;
            }
        }
        throw TokenizerError("BPE: символа нет в словаре (нет unk_token): \"" +
                             bytes + "\"; словарь неполный?");
    }
    return ids;
}

std::vector<int32_t> Tokenizer::encode_ordinary(const std::string &text) const {
    std::vector<int32_t> ids;
    if (text.empty()) return ids;
    if (info_.use_regex) {
        for (const std::string &piece : pre_tokenize(text)) {
            const std::vector<int32_t> part = bpe(piece);
            ids.insert(ids.end(), part.begin(), part.end());
        }
    } else {
        ids = bpe(text);
    }
    return ids;
}

bool Tokenizer::match_added(const std::string &text, std::size_t pos,
                            bool allow_special, std::size_t *len,
                            int32_t *id) const {
    std::size_t best_len = 0;
    int32_t best_id = -1;
    for (const AddedToken &tok : added_) {
        if (tok.special && !allow_special) continue;
        if (tok.content.size() <= best_len) continue;
        if (pos + tok.content.size() > text.size()) continue;
        if (text.compare(pos, tok.content.size(), tok.content) == 0) {
            best_len = tok.content.size();
            best_id = tok.id;
        }
    }
    if (best_len == 0) return false;
    *len = best_len;
    *id = best_id;
    return true;
}

std::vector<int32_t> Tokenizer::encode(const std::string &text,
                                       const EncodeOptions &opts) const {
    // added_tokens ищем в исходном тексте: спецтокены не нормализуются
    // (как в HF: normalizer применяется только к обычным фрагментам).
    const std::string &source = text;

    auto encode_segment = [&](std::size_t begin, std::size_t end) {
        std::string segment = source.substr(begin, end - begin);
        if (info_.add_prefix_space && !segment.empty() && segment[0] != ' ') {
            segment.insert(segment.begin(), ' ');
        }
        return encode_ordinary(normalize(segment));
    };

    std::vector<int32_t> out;
    std::size_t start = 0;
    std::size_t pos = 0;
    while (pos < source.size()) {
        std::size_t len = 0;
        int32_t id = -1;
        if (match_added(source, pos, opts.allow_special_tokens, &len, &id)) {
            if (pos > start) {
                const std::vector<int32_t> part = encode_segment(start, pos);
                out.insert(out.end(), part.begin(), part.end());
            }
            out.push_back(id);
            pos += len;
            start = pos;
        } else {
            ++pos;
        }
    }
    if (start < source.size()) {
        const std::vector<int32_t> part = encode_segment(start, source.size());
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

// ---------------------------------------------------------------------------
// Tokenizer: decode
// ---------------------------------------------------------------------------

std::string Tokenizer::decode(const std::vector<int32_t> &ids) const {
    const ByteTables &tables = byte_tables();
    std::string out;
    std::string pending;  // байты обычных токенов (могут резать UTF-8 символ)

    auto flush_pending = [&]() {
        if (!pending.empty()) {
            out += utf8_sanitize(pending);
            pending.clear();
        }
    };

    for (const int32_t id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size() ||
            id_to_token_[static_cast<std::size_t>(id)].empty()) {
            throw TokenizerError("decode: неизвестный token id = " +
                                 std::to_string(id));
        }
        const std::string &tok = id_to_token_[static_cast<std::size_t>(id)];
        const bool is_added = static_cast<std::size_t>(id) < is_added_.size() &&
                              is_added_[static_cast<std::size_t>(id)] != 0;
        if (is_added) {
            flush_pending();
            out += tok;  // спецтокен выводится как есть, без byte-level
            continue;
        }
        if (!decoder_byte_level_) {
            flush_pending();
            out += tok;
            continue;
        }
        std::size_t i = 0;
        while (i < tok.size()) {
            uint32_t cp = 0;
            std::size_t len = 0;
            if (!utf8_decode(tok, i, &cp, &len)) {
                pending.push_back(tok[i]);
                ++i;
                continue;
            }
            const auto it = tables.dec.find(cp);
            if (it != tables.dec.end()) {
                pending.push_back(static_cast<char>(it->second));
            } else {
                pending.append(tok, i, len);
            }
            i += len;
        }
    }
    flush_pending();
    return out;
}

// ---------------------------------------------------------------------------
// Tokenizer: инкрементальный decode (стриминг по одному токену)
// ---------------------------------------------------------------------------

std::string DecoderStream::push(int32_t id, bool *stop) {
    const Tokenizer &t = *tok_;
    if (stop != nullptr) *stop = false;
    if (id < 0 || static_cast<std::size_t>(id) >= t.id_to_token_.size() ||
        t.id_to_token_[static_cast<std::size_t>(id)].empty()) {
        throw TokenizerError("decode: неизвестный token id = " +
                             std::to_string(id));
    }
    const std::size_t uid = static_cast<std::size_t>(id);
    std::string out;

    // Стоп-токен: текст не выдаём, генерацию пора заканчивать. Недописанный
    // хвост отдаст finish() — не теряем возможный последний символ.
    if (t.is_stop_token(id)) {
        if (stop != nullptr) *stop = true;
        return out;
    }

    const std::string &tok = t.id_to_token_[uid];
    const bool is_added = uid < t.is_added_.size() && t.is_added_[uid] != 0;
    if (is_added) {
        // Управляющий добавочный токен (<|im_end|> и пр.) в текст не идёт.
        if (t.is_special_token(id)) return out;
        // Обычный добавочный токен — готовый текст, но до него надо
        // доизвлечь всё, что успело накопиться.
        utf8_emit_complete(pending_, out, true);
        out += tok;
        return out;
    }
    if (!t.decoder_byte_level_) {
        utf8_emit_complete(pending_, out, true);
        out += tok;
        return out;
    }

    // byte-level: символы токена — это байты UTF-8 текста. Копим их; отдаём
    // только целые символы, хвост (до 3 байт) ждёт следующий токен.
    const ByteTables &tables = byte_tables();
    std::size_t i = 0;
    while (i < tok.size()) {
        uint32_t cp = 0;
        std::size_t len = 0;
        if (!utf8_decode(tok, i, &cp, &len)) {
            pending_.push_back(tok[i]);
            ++i;
            continue;
        }
        const auto it = tables.dec.find(cp);
        if (it != tables.dec.end()) {
            pending_.push_back(static_cast<char>(it->second));
        } else {
            pending_.append(tok, i, len);
        }
        i += len;
    }
    utf8_emit_complete(pending_, out, false);
    return out;
}

std::string DecoderStream::finish() {
    std::string out;
    // Конец генерации: недописанный символ показать уже нечем, но и терять
    // его нельзя — отдаём U+FFFD вместо сырых байт, чтобы stdout оставался
    // валидным UTF-8.
    utf8_emit_complete(pending_, out, true);
    return out;
}

bool Tokenizer::token_to_id(const std::string &token, int32_t *out) const {
    const auto it = token_to_id_.find(token);
    if (it == token_to_id_.end()) return false;
    if (out != nullptr) *out = it->second;
    return true;
}

std::string Tokenizer::id_to_token(int32_t id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size() ||
        id_to_token_[static_cast<std::size_t>(id)].empty()) {
        throw TokenizerError("неизвестный token id = " + std::to_string(id));
    }
    return id_to_token_[static_cast<std::size_t>(id)];
}

} // namespace vtm
