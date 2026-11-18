// vtm-tok — CLI к BPE-токенизатору VTM.
//
//   vtm-tok [опции] encode "текст"      -> id через запятую
//   vtm-tok [опции] decode 1,2,3        -> текст (или id из stdin)
//   vtm-tok [опции] info [tokenizer.json]
//
// Опции:
//   -t, --tokenizer PATH   путь к tokenizer.json (иначе $VTM_TOKENIZER,
//                          затем поиск tests/vtm/fixtures/tokenizer.json вверх)
//       --no-special       encode: не выдёргивать special == true added_tokens
//   -h, --help
//       --version
#include "tokenizer.hpp"

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

const char *kUsage =
    "использование:\n"
    "  vtm-tok [опции] encode \"текст\"     id через запятую\n"
    "  vtm-tok [опции] decode 1,2,3       текст (или id из stdin)\n"
    "  vtm-tok [опции] info [tokenizer.json]\n"
    "\n"
    "опции:\n"
    "  -t, --tokenizer PATH  путь к tokenizer.json\n"
    "                        (по умолчанию $VTM_TOKENIZER, затем\n"
    "                         tests/vtm/fixtures/tokenizer.json вверх по дереву)\n"
    "      --no-special      encode: не разбирать special == true added_tokens\n"
    "  -h, --help\n"
    "      --version\n";

bool file_exists(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    return in.good();
}

// Путь из аргумента ищем и относительно cwd, и относительно родителей cwd:
// так `vtm-tok info tests/vtm/fixtures/tokenizer.json` работает и из src/vtm.
std::string resolve_input_path(const std::string &given) {
    if (given.empty() || file_exists(given)) return given;
    char buf[4096];
    if (::getcwd(buf, sizeof(buf)) == nullptr) return given;
    std::string dir = buf;
    for (;;) {
        const std::string candidate = dir + "/" + given;
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
    return given;
}

std::string pick_tokenizer(const std::string &opt) {
    if (!opt.empty()) return resolve_input_path(opt);
    const std::string def = vtm::Tokenizer::default_path();
    if (def.empty()) {
        throw vtm::TokenizerError(
            "tokenizer.json не найден: укажите -t PATH или $VTM_TOKENIZER");
    }
    return def;
}

std::string read_stdin() {
    std::ostringstream out;
    out << std::cin.rdbuf();
    return out.str();
}

std::vector<int32_t> parse_ids(const std::string &input) {
    std::vector<int32_t> ids;
    std::size_t i = 0;
    const std::size_t n = input.size();
    while (i < n) {
        while (i < n) {
            const unsigned char c = static_cast<unsigned char>(input[i]);
            if (std::isspace(c) || c == ',' || c == '[' || c == ']') {
                ++i;
            } else {
                break;
            }
        }
        if (i >= n) break;
        const char *begin = input.c_str() + i;
        char *end = nullptr;
        errno = 0;
        const long long v = std::strtoll(begin, &end, 10);
        if (end == begin || errno != 0 || v < 0 ||
            v > std::numeric_limits<int32_t>::max()) {
            throw vtm::TokenizerError(
                "некорректный id токена в позиции " + std::to_string(i) +
                " (\"" + input.substr(i, 16) + "\")");
        }
        ids.push_back(static_cast<int32_t>(v));
        i += static_cast<std::size_t>(end - begin);
        if (i < n) {
            const unsigned char c = static_cast<unsigned char>(input[i]);
            if (!std::isspace(c) && c != ',' && c != ']') {
                throw vtm::TokenizerError(
                    "ожидались id через запятую в позиции " + std::to_string(i));
            }
        }
    }
    return ids;
}

std::string format_ids(const std::vector<int32_t> &ids) {
    std::string out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) out.push_back(',');
        out += std::to_string(ids[i]);
    }
    return out;
}

std::string describe_id(const vtm::Tokenizer &tok, int32_t id) {
    if (id < 0) return "none";
    std::string text;
    try {
        text = tok.id_to_token(id);
    } catch (const std::exception &) {
        return std::to_string(id) + " (нет в словаре)";
    }
    std::string shown;
    for (unsigned char c : text) {
        if (c >= 0x20 && c < 0x7F) {
            shown.push_back(static_cast<char>(c));
        } else {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\x%02X", c);
            shown += buf;
        }
    }
    return std::to_string(id) + " (\"" + shown + "\")";
}

void print_info(const vtm::Tokenizer &tok) {
    const vtm::TokenizerInfo &info = tok.info();
    std::printf("file: %s\n", info.source.c_str());
    std::printf("model: %s\n", info.model_type.c_str());
    std::printf("vocab: %zu\n", info.vocab_size);
    std::printf("merges: %zu (missing_in_vocab: %zu)\n", info.merge_count,
                info.merges_missing_from_vocab);
    std::printf("added_tokens: %zu (special: %zu)\n", info.added_token_count,
                info.special_token_count);
    std::printf("byte_alphabet: %zu/256\n", info.byte_alphabet_size);
    std::printf("byte_level: %s\n", info.byte_level ? "yes" : "no");
    std::printf("pre_tokenizer: %s (use_regex=%s, add_prefix_space=%s)\n",
                info.pre_tokenizer_type.c_str(),
                info.use_regex ? "yes" : "no",
                info.add_prefix_space ? "yes" : "no");
    std::printf("normalizer: %s\n", info.normalizer_type.c_str());
    std::printf("decoder: %s\n", info.decoder_type.c_str());
    std::printf("unk: %s\n", describe_id(tok, info.unk_id).c_str());
    std::printf("bos: %s\n", describe_id(tok, info.bos_id).c_str());
    std::printf("eos: %s\n", describe_id(tok, info.eos_id).c_str());
    std::printf("pad: %s\n", describe_id(tok, info.pad_id).c_str());
    if (info.warnings.empty()) {
        std::printf("warnings: none\n");
    } else {
        std::printf("warnings (%zu):\n", info.warnings.size());
        for (const std::string &w : info.warnings) {
            std::printf("  - %s\n", w.c_str());
        }
    }
}

// Текст команды: один аргумент как есть; несколько — через пробел.
std::string join_args(int argc, char **argv, int from) {
    std::string out;
    for (int i = from; i < argc; ++i) {
        if (i != from) out.push_back(' ');
        out += argv[i];
    }
    return out;
}

} // namespace

int main(int argc, char **argv) {
    std::ios::sync_with_stdio(false);

    std::string tokenizer_path;
    bool allow_special = true;
    int i = 1;
    for (; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::printf("%s", kUsage);
            return 0;
        }
        if (arg == "--version") {
            std::printf("vtm-tok 0.1\n");
            return 0;
        }
        if (arg == "-t" || arg == "--tokenizer") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "vtm-tok: ошибка: %s требует путь\n",
                             arg.c_str());
                return 2;
            }
            tokenizer_path = argv[++i];
            continue;
        }
        const std::string prefix = "--tokenizer=";
        if (arg.compare(0, prefix.size(), prefix) == 0) {
            tokenizer_path = arg.substr(prefix.size());
            continue;
        }
        if (arg == "--no-special") {
            allow_special = false;
            continue;
        }
        break;
    }

    if (i >= argc) {
        std::fprintf(stderr, "vtm-tok: ошибка: не указана команда\n\n%s",
                     kUsage);
        return 2;
    }
    const std::string command = argv[i++];

    try {
        if (command == "encode") {
            const std::string text =
                (i < argc) ? join_args(argc, argv, i) : read_stdin();
            const std::string path = pick_tokenizer(tokenizer_path);
            const vtm::Tokenizer tok = vtm::Tokenizer::load(path);
            vtm::EncodeOptions opts;
            opts.allow_special_tokens = allow_special;
            const std::vector<int32_t> ids = tok.encode(text, opts);
            std::printf("%s\n", format_ids(ids).c_str());
            return 0;
        }
        if (command == "decode") {
            const std::string text =
                (i < argc) ? join_args(argc, argv, i) : read_stdin();
            const std::vector<int32_t> ids = parse_ids(text);
            const std::string path = pick_tokenizer(tokenizer_path);
            const vtm::Tokenizer tok = vtm::Tokenizer::load(path);
            std::printf("%s\n", tok.decode(ids).c_str());
            return 0;
        }
        if (command == "info") {
            std::string path = tokenizer_path;
            if (i < argc) {
                path = argv[i];
            } else if (path.empty()) {
                path = vtm::Tokenizer::default_path();
                if (path.empty()) {
                    throw vtm::TokenizerError(
                        "tokenizer.json не найден: укажите путь, -t PATH или "
                        "$VTM_TOKENIZER");
                }
            }
            path = resolve_input_path(path);
            const vtm::Tokenizer tok = vtm::Tokenizer::load(path);
            print_info(tok);
            return 0;
        }
        std::fprintf(stderr, "vtm-tok: ошибка: неизвестная команда \"%s\"\n\n%s",
                     command.c_str(), kUsage);
        return 2;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "vtm-tok: ошибка: %s\n", e.what());
        return 1;
    }
}
