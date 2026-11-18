// VTM: BPE-токенизатор (C++17, без внешних зависимостей).
//
// Читает tokenizer.json формата HuggingFace (model.type == "BPE"),
// поддерживает byte-level отображение (GPT-2/Qwen), added_tokens
// (поиск подстроки до BPE), encode()/decode().
//
// Пример:
//   vtm::Tokenizer tok = vtm::Tokenizer::load("tokenizer.json");
//   std::vector<int32_t> ids = tok.encode("Привет, мир!");
//   std::string text = tok.decode(ids);          // text == "Привет, мир!"
//
// Ошибки чтения/разбора — исключение vtm::TokenizerError (std::runtime_error),
// сообщение содержит путь/позицию. Исключения не покидают библиотеку молча.
#ifndef VTM_TOKENIZER_H
#define VTM_TOKENIZER_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vtm {

class TokenizerError : public std::runtime_error {
public:
    explicit TokenizerError(const std::string &message)
        : std::runtime_error(message) {}
};

// Сведения о загруженном токенизаторе (для vtm-tok info и диагностики).
struct TokenizerInfo {
    std::string source;              // путь/метка источника
    std::string model_type;          // model.type ("BPE")
    std::string normalizer_type;     // "null", "NFC", "Sequence(Replace)" ...
    std::string pre_tokenizer_type;  // "ByteLevel", "Sequence(...)", "null"
    std::string decoder_type;        // "ByteLevel", "Sequence(...)", "null"

    std::size_t vocab_size = 0;             // число обычных токенов в model.vocab
    std::size_t merge_count = 0;            // принятых правил BPE
    std::size_t merges_missing_from_vocab = 0;  // отброшенных (нет результата в vocab)
    std::size_t added_token_count = 0;      // added_tokens (все)
    std::size_t special_token_count = 0;    // из них special == true
    std::size_t byte_alphabet_size = 0;     // сколько из 256 byte-level символов есть в vocab

    bool byte_level = false;        // найден ByteLevel в pre_tokenizer/decoder
    bool use_regex = true;          // ByteLevel.use_regex (GPT-2 regex-разбиение)
    bool add_prefix_space = false;  // ByteLevel.add_prefix_space

    int32_t unk_id = -1;
    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t pad_id = -1;
    std::string unk_token;
    std::string bos_token;
    std::string eos_token;
    std::string pad_token;

    // Что распознано, но не поддержано полностью (например, нормализатор NFC).
    std::vector<std::string> warnings;
};

struct EncodeOptions {
    // false — не выдёргивать special == true added_tokens (обычный текст).
    bool allow_special_tokens = true;
};

// Публичное описание added_tokens (порядок — как в tokenizer.json).
// Нужно хосту, чтобы строить чат-шаблон из спецтокенов самого словаря,
// а не из захардкоженных id: content, id и special берутся из файла.
struct AddedTokenInfo {
    std::string content;
    int32_t id = -1;
    bool special = false;
};

class Tokenizer {
public:
    // Загрузить tokenizer.json с диска. Бросает TokenizerError.
    static Tokenizer load(const std::string &path);
    // Разобрать JSON из памяти (source — для сообщений об ошибках).
    static Tokenizer from_json(const std::string &json,
                               const std::string &source = "<memory>");
    // Путь по умолчанию: $VTM_TOKENIZER, затем ./tokenizer.json,
    // tests/vtm/fixtures/tokenizer.json (с подъёмом по родительским каталогам).
    // Пустая строка — ничего не найдено.
    static std::string default_path();

    std::vector<int32_t> encode(const std::string &text,
                                const EncodeOptions &opts = EncodeOptions()) const;
    std::string decode(const std::vector<int32_t> &ids) const;

    // Найти id токена (включая added_tokens). false — нет такого токена.
    bool token_to_id(const std::string &token, int32_t *out) const;
    // Текст токена по id; бросает TokenizerError на неизвестном id.
    std::string id_to_token(int32_t id) const;

    const TokenizerInfo &info() const { return info_; }
    // Все токены по id (для отладки/CLI); added_tokens тоже входят.
    const std::vector<std::string> &id_to_token_table() const { return id_to_token_; }

    // added_tokens в порядке tokenizer.json. Определён здесь inline, чтобы
    // не менять tokenizer.cpp: чат-шаблону нужен content спецтокенов
    // (<|im_start|> и т. п.) и их id из файла, а не из кода.
    std::vector<AddedTokenInfo> added_tokens() const {
        std::vector<AddedTokenInfo> out;
        out.reserve(added_.size());
        for (const AddedToken &t : added_) {
            out.push_back(AddedTokenInfo{t.content, t.id, t.special});
        }
        return out;
    }

private:
    friend class TokenizerBuilder;

    struct AddedToken {
        std::string content;
        int32_t id = -1;
        bool special = false;
    };

    // model.vocab: токен -> id; id -> токен.
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<char> is_added_;  // id принадлежит added_tokens

    // BPE: пара символов -> ранг (меньше — раньше). Пара хранится как
    // конкатенация через '\0' (в токенах byte-level нет нулевого байта).
    std::unordered_map<std::string, int32_t> merge_rank_;

    std::vector<AddedToken> added_;  // в порядке added_tokens (приоритет при равной длине)
    bool decoder_byte_level_ = false;

    // Поддерживаемое подмножество normalizer (последовательность простых операций).
    struct NormalizerOp {
        enum Kind { Replace, Prepend, Strip, Lowercase } kind = Replace;
        std::string a;  // Replace: что ищем
        std::string b;  // Replace/Prepend: на что/чем заменяем
    };
    std::vector<NormalizerOp> normalizer_ops_;

    TokenizerInfo info_;

    // UTF-8 нормализация и GPT-2-разбиение.
    std::string normalize(const std::string &text) const;
    std::vector<std::string> pre_tokenize(const std::string &text) const;
    std::vector<int32_t> encode_ordinary(const std::string &text) const;
    std::vector<int32_t> bpe(const std::string &piece) const;
    bool match_added(const std::string &text, std::size_t pos,
                     bool allow_special, std::size_t *len, int32_t *id) const;
};

} // namespace vtm

#endif // VTM_TOKENIZER_H
