#!/usr/bin/env python3
"""
Gera include/minicpm_metadata.h a partir de um arquivo GGUF.

Uso:
    python tools/generate_header.py <path_to_gguf> [output_h]

Exemplo:
    python tools/generate_header.py ../MiniCPM5-2B-Q4_K_M.gguf
    python tools/generate_header.py ../MiniCPM5-2B-Q4_K_M.gguf include/minicpm_metadata.h
"""
import sys
from pathlib import Path

# Resolve gguf-py do sibling llama.cpp
LLAMA_CPP = Path(__file__).resolve().parent.parent.parent / "llama.cpp"
sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))

try:
    from gguf.gguf_reader import GGUFReader
except ImportError:
    print(f"ERROR: gguf-py não encontrado em {LLAMA_CPP / 'gguf-py'}")
    print("Certifique-se de que llama.cpp está clonado na pasta pai.")
    sys.exit(1)


def get_scalar_value(reader, key, default):
    if key not in reader.fields:
        return default
    field = reader.fields[key]
    val = field.parts[field.data[0]]
    if hasattr(val, '__len__') and not isinstance(val, (str, bytes)):
        return val[0]
    return val


def extract_metadata_and_generate_h(gguf_path, output_h_path):
    print(f"[gen_header] Lendo GGUF: {gguf_path}")
    reader = GGUFReader(gguf_path)

    dim = int(get_scalar_value(reader, "llama.embedding_length", 2048))
    ffn_dim = int(get_scalar_value(reader, "llama.feed_forward_length", 6144))
    layers = int(get_scalar_value(reader, "llama.block_count", 42))
    heads = int(get_scalar_value(reader, "llama.attention.head_count", 16))
    kv_heads = int(get_scalar_value(reader, "llama.attention.head_count_kv", 2))
    vocab_size = int(get_scalar_value(reader, "llama.vocab_size", 130560))
    rope_base = float(get_scalar_value(reader, "llama.rope.freq_base", 5000000.0))
    alignment = int(get_scalar_value(reader, "general.alignment", 32))

    tensors_list = []
    for tensor in reader.tensors:
        shape = list(tensor.shape)
        dims = shape + [0] * (4 - len(shape))
        tensors_list.append({
            "name": tensor.name,
            "offset": tensor.data_offset,
            "size_bytes": tensor.n_bytes,
            "n_dims": len(shape),
            "dims": dims,
            "type": tensor.tensor_type.name
        })

    out = Path(output_h_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[gen_header] Escrevendo: {out}")

    with open(out, "w") as f:
        f.write("// ==========================================================\n")
        f.write("// ARQUIVO GERADO AUTOMATICAMENTE - NAO EDITE DIRETAMENTE\n")
        f.write(f"// Gerado por: tools/generate_header.py\n")
        f.write(f"// Fonte GGUF: {Path(gguf_path).name}\n")
        f.write("// ==========================================================\n\n")
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n\n")

        f.write("// Constantes da Arquitetura do Modelo\n")
        f.write(f"const int MODEL_DIM = {dim};\n")
        f.write(f"const int MODEL_FFN_DIM = {ffn_dim};\n")
        f.write(f"const int MODEL_LAYERS = {layers};\n")
        f.write(f"const int MODEL_HEADS = {heads};\n")
        f.write(f"const int MODEL_KV_HEADS = {kv_heads};\n")
        f.write(f"const int MODEL_VOCAB_SIZE = {vocab_size};\n")
        f.write(f"const float MODEL_ROPE_BASE = {rope_base}f;\n")
        f.write(f"const uint64_t GGUF_ALIGNMENT = {alignment};\n\n")

        f.write("struct TensorMetadata {\n")
        f.write("    const char* name;\n")
        f.write("    uint64_t absolute_offset;\n")
        f.write("    uint64_t size_bytes;\n")
        f.write("    uint32_t n_dims;\n")
        f.write("    uint64_t dims[4];\n")
        f.write("    const char* type_str;\n")
        f.write("};\n\n")

        f.write(f"const int NUM_TENSORS = {len(tensors_list)};\n")
        f.write("const TensorMetadata TENSORS_INFO[NUM_TENSORS] = {\n")
        for i, t in enumerate(tensors_list):
            dims_str = ", ".join(map(str, t["dims"]))
            comma = "," if i < len(tensors_list) - 1 else ""
            f.write(f'    {{ "{t["name"]}", {t["offset"]}, {t["size_bytes"]}, {t["n_dims"]}, {{ {dims_str} }}, "{t["type"]}" }}{comma}\n')
        f.write("};\n")

    print(f"[gen_header] Concluído! ({len(tensors_list)} tensores)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    gguf = sys.argv[1]
    default_out = str(Path(__file__).resolve().parent.parent / "include" / "minicpm_metadata.h")
    output = sys.argv[2] if len(sys.argv) >= 3 else default_out
    extract_metadata_and_generate_h(gguf, output)
