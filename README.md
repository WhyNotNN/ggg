# SDXL generation with JAX on 2× AMD MI50 (gfx906)

Запуск Stable Diffusion XL через Diffusers Flax/JAX на AMD Instinct MI50 16GB.
MI50 — это `gfx906`, поэтому ставим `gfx906`-сборки ROCm и JAX из ночного
индекса AMD:

```text
https://rocm.nightlies.amd.com/v2-staging/gfx906/
```

> AMD больше не выпускает поддержку MI50 в стабильном ROCm. `v2-staging/gfx906`
> — единственный рабочий путь получить JAX/PJRT под `gfx906` сегодня. Колёса
> `jaxlib`, `jax-rocm7-plugin`, `jax-rocm7-pjrt` и `rocm-sdk-libraries-gfx906`
> ставятся одной датой, иначе ABI PJRT-плагина не сойдётся с ROCm-рантаймом.

## Требования к хосту

- Linux x86_64 с подгруженным `amdgpu`, доступными `/dev/kfd` и `/dev/dri`,
  пользователь в группе `video` (или `render`).
- Python **3.11 / 3.12 / 3.13 / 3.14** (для более старых интерпретаторов
  колёс на индексе нет). Рекомендуется 3.12.
- `pip >= 24`.
- Достаточно свободного места под кэш моделей HF (SDXL base ~ 13 GB).

Сам ROCm 7 userspace притаскивается колесом `rocm-sdk-libraries-gfx906` —
ставить системный `/opt/rocm` не обязательно.

## Установка

```bash
python3.12 -m venv .venv && source .venv/bin/activate

# Свежие nightly:
PYTHON_BIN=python scripts/install_rocm_jax.sh

# Или приколотить всё к одной ночной дате (рекомендуется для повторяемости):
PYTHON_BIN=python scripts/install_rocm_jax.sh 20260428
```

Для приватных/гейтированных моделей HF — один раз залогиниться:

```bash
huggingface-cli login
```

## 1. Проверить, что JAX видит обе MI50

```bash
scripts/run_sdxl_jax.sh \
  --check-only \
  --compile-smoke-test \
  --devices 0,1 \
  --require-devices 2
```

Должны увидеть два устройства `rocm:0` и `rocm:1` и `compile smoke test ok`.
Если smoke-тест валится — поставь более старую дату через
`scripts/install_rocm_jax.sh <YYYYMMDD>`.

## 2. Сгенерировать картинки

По одной 1024×1024 на каждую MI50:

```bash
scripts/run_sdxl_jax.sh \
  --devices 0,1 \
  --require-devices 2 \
  --prompt "cinematic photo of a futuristic research station on Mars, detailed, sharp focus" \
  --negative-prompt "low quality, blurry, distorted" \
  --steps 30 \
  --guidance-scale 5.0 \
  --images-per-device 1 \
  --dtype fp16 \
  --output-dir outputs/sdxl-mi50
```

На одной карте:

```bash
HIP_VISIBLE_DEVICES=0 scripts/run_sdxl_jax.sh \
  --require-devices 1 \
  --prompt "studio portrait of a snow leopard, detailed fur" \
  --steps 30 --dtype fp16 \
  --output-dir outputs/sdxl-mi50
```

Первый запуск долгий: компилируется JAX-программа и скачиваются веса. Повторные
запуски с теми же формами тензоров — быстрые.

## Что делает `scripts/run_sdxl_jax.sh`

Прокидывает переменные, которые важны на `gfx906` 16GB, и зовёт
`generate_sdxl_jax.py`:

- `JAX_PLATFORMS=rocm`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — не выгребать сразу всю VRAM.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` — мягкий потолок на процесс, чтобы
  attention в SDXL не влетал в OOM из-за фрагментации.
- `MIOPEN_FIND_MODE=FAST` — без долгих автотюнов MIOpen, которые на Vega20
  любят зависать.
- `HIP_VISIBLE_DEVICES=0,1` — обе MI50 по умолчанию.

Любую переменную можно перебить, просто экспортнув её до запуска скрипта.

## Замечания по MI50 16GB

- Держи `--images-per-device 1` для 1024×1024 SDXL.
- `--dtype fp16` — fp32 SDXL не влезет в 16GB.
- Параметры планировщика остаются в fp32 даже при fp16-весах: иначе у SDXL
  встречается рассинхрон сигм.
- Если новейший nightly валится с ошибками Triton/`gfx906`-компилятора —
  переустановись на более раннюю дату через
  `scripts/install_rocm_jax.sh <YYYYMMDD>`. Скрипт зафиксирует все четыре
  колеса (`jaxlib`, `jax-rocm7-plugin`, `jax-rocm7-pjrt`,
  `rocm-sdk-libraries-gfx906`) на одну дату.
- Если ловишь OOM — снижай `--mem-fraction 0.85` (или `0.80`).
- `--no-jit` — только для отладки; обычная генерация должна идти с JIT.
