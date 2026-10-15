# AOSP Integration — README

## What is this folder?

This folder contains the **Task 7: AOSP C++ Integration** deliverable for Phase 2
of the Adaptive Memory Management (AMM) project.

It proves that the Python ML pipeline can be ported to native Android OS code.

---

## Folder Structure

```
aosp_integration/
├── cpp_prototype/
│   └── inference.cpp          Full C++ inference engine (compiles standalone)
├── docs/
│   └── AOSP_ARCHITECTURE.md   Hook points in ActivityManagerService & lmkd
├── onnx_export/
│   └── export_onnx.py         Export XGBoost model to ONNX for C++ runtime
└── README.md                  This file
```

---

## What Does Each File Do?

### `cpp_prototype/inference.cpp`
- Standalone C++ implementation of the **full AMM inference pipeline**
- Feature extraction: 46-dim vector (19 base + 16 App2Vec + 11 PrefixSpan)
- 6-Factor Cache Score: `S = 0.25R + 0.20F + 0.30P + 0.10I + 0.10C - 0.30M`
- Preload Gate: 5-condition safety check
- Mock ONNX inference (replace with `Ort::Session::Run()` for real model)
- Compiles on any platform with a C++17 compiler

### `docs/AOSP_ARCHITECTURE.md`
- Exact line-by-line description of AOSP hook points
- Modified `ActivityManagerService.java` snippet
- Modified `lmkd.cpp` snippet
- New `AMMDecisionEngine.java` class design
- JNI bridge (`amm_jni.cpp`) design
- Android.bp build config
- Complete data flow diagram

### `onnx_export/export_onnx.py`
- Exports trained XGBoost Phase 2 model to ONNX format
- ONNX model can then be loaded by ONNX Runtime in C++/Java/Android

---

## How to Compile the C++ Prototype

### Option 1: Linux / macOS (no ONNX Runtime needed for mock version)
```bash
# Comment out the ONNX Runtime #include line in inference.cpp, then:
g++ -O2 -std=c++17 -lm aosp_integration/cpp_prototype/inference.cpp -o amm_inference
./amm_inference
```

### Option 2: With real ONNX Runtime (Linux)
```bash
# Install ONNX Runtime:
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz
tar xf onnxruntime-linux-x64-1.17.0.tgz

# Compile:
g++ -O2 -std=c++17 aosp_integration/cpp_prototype/inference.cpp \
    -I onnxruntime-linux-x64-1.17.0/include \
    -L onnxruntime-linux-x64-1.17.0/lib -lonnxruntime \
    -o amm_inference

# Run:
./amm_inference aosp_integration/onnx_export/xgboost_phase2.onnx
```

### Option 3: Android NDK (cross-compile for ARM64)
```bash
export NDK=$HOME/Android/Sdk/ndk/26.1.10909125
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++ \
    -O2 -std=c++17 aosp_integration/cpp_prototype/inference.cpp \
    -I ${ONNXRT_ANDROID}/headers \
    -L ${ONNXRT_ANDROID}/jni/arm64-v8a -lonnxruntime \
    -o amm_inference_arm64

# Push to phone:
adb push amm_inference_arm64 /data/local/tmp/
adb shell chmod +x /data/local/tmp/amm_inference_arm64
adb shell /data/local/tmp/amm_inference_arm64
```

---

## Export the ONNX Model

```bash
# First run the Phase 2 pipeline to train the model:
python scripts/run_phase2.py

# Then export to ONNX:
pip install onnxmltools skl2onnx onnx
python aosp_integration/onnx_export/export_onnx.py
```

---

## Why Not Full AOSP Build?

Full AOSP build requires:
- Ubuntu 20.04+ Linux machine
- 200+ GB disk space (AOSP source is ~150 GB)
- 16+ GB RAM
- 2-3 days of compilation time

Instead, we provide:
1. **C++ prototype** — proves the algorithm runs in C++ with correct outputs
2. **Architecture document** — proves we know exactly where to hook into AOSP
3. **ONNX export** — proves the model format is compatible with Android runtime

This is the standard approach used in academic and industrial research papers
when proposing OS-level ML integration (see: [B1] LMKD paper, [B2] MobiSys paper
in the literature survey).
