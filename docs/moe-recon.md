# Разведка: Qwen3.5-35B-A3B (2026-11-25)

Источник: `Qwen/Qwen3.5-35B-A3B` config.json (HF). Актуально на 2026-11-25.

## Что это

- `architectures: Qwen3_5MoeForConditionalGeneration`, `model_type: qwen3_5_moe`.
- **Мультимодальная** (vision_config: 27 слоёв, patch 16) — текст отдельно от картинок.
- 40 слоёв, hidden 2048, vocab **248 320**, `tie_word_embeddings: false`.
- **MoE: 256 экспертов, 8 активных** на токен, `moe_intermediate_size: 512`,
  плюс общий эксперт 512. Счёт как у ~3B — вот почему «летает» на CPU.
- **Гибридное внимание:** `layer_types` = 30 × `linear_attention` + 10 × `full_attention`
  (каждый 4-й слой полный). `linear_conv_kernel_dim: 4`, head_dim 128/128,
  32 value-головы.
- mRoPE: `mrope_interleaved`, section [11,11,10], theta 1e7, `partial_rotary_factor 0.25`,
  `head_dim: 256` у полного внимания.
- Контекст 262 144.

## Что это значит для ВТМ

| Блок | Есть у нас | Нужно |
|---|---|---|
| Dense attention (GQA) | ✅ | — |
| MoE-роутер + top-8 из 256 | ❌ | операция роутера, разреженные эксперты, кэш |
| **Linear attention (gated delta)** | ❌ | новая механика: состояние + conv-kernel 4 |
| mRoPE (частичный, interleaved) | частично | новая раскладка частот |
| vocab 248320, untied lm_head | ✅ (int8-таблица) | — |
| Vision tower | ❌ | не нужен для текста |

## Вывод

Это **не «тот же принцип», что Qwen3-30B-A3B**: там чистый MoE на обычном внимании,
здесь — гибрид с gated-delta linear attention. Портировать в ВТМ можно, но это:
1) MoE (роутер, эксперты, кэш) — 1 спринт;
2) linear attention — ещё 1 спринт (новая механика состояния, не только матрицы);
3) mRoPE — небольшая правка.

Рекомендация: начать с **Qwen3-30B-A3B** (проверенный MoE без новой attention), а
Qwen3.5-35b брать после MoE-базы. Иначе рискуем построить импортёр под архитектуру,
которую интерпретатор не исполняет.
