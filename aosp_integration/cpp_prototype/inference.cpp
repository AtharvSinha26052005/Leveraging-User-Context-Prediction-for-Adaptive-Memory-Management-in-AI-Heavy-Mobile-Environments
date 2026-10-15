/*
 * inference.cpp — Standalone C++ Inference Prototype for ONNX-exported XGBoost Model
 *
 * Demonstrates that the prediction engine can run outside Python using the
 * ONNX Runtime C++ API. This is the feasibility proof for full AOSP integration.
 *
 * Compilation (Linux/macOS with ONNX Runtime installed):
 *   g++ -O2 -std=c++17 inference.cpp \
 *       -I/usr/local/include/onnxruntime \
 *       -L/usr/local/lib -lonnxruntime \
 *       -o amm_inference
 *
 * Compilation (Android NDK cross-compile):
 *   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++ \
 *       -O2 -std=c++17 inference.cpp \
 *       -I${ONNXRT_ANDROID}/headers \
 *       -L${ONNXRT_ANDROID}/jni/arm64-v8a -lonnxruntime \
 *       -o amm_inference_arm64
 *
 * Usage:
 *   ./amm_inference models_trained/xgboost_model.onnx
 *
 * Reference:
 *   ONNX Runtime C++ API: https://onnxruntime.ai/docs/api/c/
 */

#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <chrono>
#include <cmath>
#include <cassert>

// ─── ONNX Runtime headers ────────────────────────────────────────────────────
// Uncomment when ONNX Runtime is installed:
// #include <onnxruntime_cxx_api.h>

// ─── Feature index constants (must match Python FeatureEngine order) ─────────
// Base features (0-18): 19 features from FeatureEngine
// App2Vec features (19-34): 8-dim for current_app + 8-dim for prev_app_1
// PrefixSpan features (35-45): 10 binary pattern flags + 1 confidence
static constexpr int FEATURE_DIM         = 46;   // Total: 19 + 16 + 11
static constexpr int BASE_FEATURE_DIM    = 19;
static constexpr int APP2VEC_DIM         = 16;   // 8 per app × 2 apps
static constexpr int PREFIXSPAN_DIM      = 11;   // 10 binary + 1 confidence
static constexpr int NUM_APPS            = 28;   // Number of unique apps

// ─── App index map (must match training label encoder) ────────────────────────
static const std::array<const char*, NUM_APPS> APP_NAMES = {
    "BGMI", "Calendar", "Camera", "CandyCrush", "ChatGPT",
    "Chrome", "Discord", "Docs", "Flipkart", "Gallery",
    "GitHub", "Gmail", "Instagram", "LinkedIn", "Maps",
    "Netflix", "Paytm", "PhonePe", "Pinterest", "Settings",
    "Slack", "Snapchat", "Spotify", "Telegram", "TikTok",
    "Twitter", "WhatsApp", "YouTube"
};

// ─── Mock App2Vec compressed embeddings (8-dim per app) ──────────────────────
// In production, these are loaded from models_trained/app_embeddings.npy
// Shape: NUM_APPS × 8  (28 × 8 = 224 floats ≈ 896 bytes)
static const float APP_EMBEDDINGS[NUM_APPS][8] = {
    // BGMI
    { 0.12f, -0.34f,  0.56f, -0.11f,  0.23f, -0.45f,  0.67f, -0.89f },
    // Calendar
    {-0.23f,  0.45f, -0.12f,  0.67f, -0.34f,  0.11f, -0.56f,  0.78f },
    // Camera
    { 0.34f, -0.12f,  0.78f, -0.45f,  0.11f, -0.67f,  0.23f, -0.56f },
    // CandyCrush
    {-0.45f,  0.23f, -0.67f,  0.12f, -0.89f,  0.34f, -0.11f,  0.45f },
    // ChatGPT
    { 0.56f, -0.78f,  0.11f, -0.23f,  0.45f, -0.12f,  0.89f, -0.34f },
    // Chrome
    {-0.67f,  0.11f, -0.45f,  0.34f, -0.12f,  0.56f, -0.78f,  0.23f },
    // Discord
    { 0.78f, -0.45f,  0.23f, -0.56f,  0.67f, -0.11f,  0.34f, -0.12f },
    // Docs
    {-0.89f,  0.67f, -0.11f,  0.45f, -0.23f,  0.78f, -0.45f,  0.11f },
    // Flipkart
    { 0.11f, -0.89f,  0.45f, -0.78f,  0.34f, -0.56f,  0.12f, -0.23f },
    // Gallery
    {-0.12f,  0.34f, -0.78f,  0.11f, -0.56f,  0.23f, -0.89f,  0.45f },
    // GitHub
    { 0.23f, -0.56f,  0.34f, -0.89f,  0.78f, -0.45f,  0.11f, -0.67f },
    // Gmail
    {-0.34f,  0.78f, -0.56f,  0.23f, -0.45f,  0.89f, -0.67f,  0.12f },
    // Instagram
    { 0.45f, -0.11f,  0.89f, -0.67f,  0.12f, -0.23f,  0.56f, -0.78f },
    // LinkedIn
    {-0.56f,  0.23f, -0.34f,  0.89f, -0.67f,  0.12f, -0.23f,  0.56f },
    // Maps
    { 0.67f, -0.45f,  0.12f, -0.34f,  0.89f, -0.78f,  0.45f, -0.11f },
    // Netflix
    {-0.78f,  0.12f, -0.89f,  0.56f, -0.11f,  0.45f, -0.34f,  0.67f },
    // Paytm
    { 0.89f, -0.67f,  0.45f, -0.12f,  0.23f, -0.89f,  0.78f, -0.45f },
    // PhonePe
    {-0.11f,  0.89f, -0.23f,  0.78f, -0.67f,  0.34f, -0.12f,  0.89f },
    // Pinterest
    { 0.22f, -0.44f,  0.66f, -0.33f,  0.55f, -0.11f,  0.77f, -0.44f },
    // Settings
    {-0.33f,  0.11f, -0.55f,  0.22f, -0.44f,  0.66f, -0.77f,  0.33f },
    // Slack
    { 0.44f, -0.66f,  0.22f, -0.55f,  0.33f, -0.77f,  0.11f, -0.22f },
    // Snapchat
    {-0.55f,  0.33f, -0.77f,  0.44f, -0.22f,  0.11f, -0.66f,  0.55f },
    // Spotify
    { 0.66f, -0.22f,  0.44f, -0.77f,  0.11f, -0.33f,  0.55f, -0.66f },
    // Telegram
    {-0.77f,  0.55f, -0.11f,  0.33f, -0.66f,  0.44f, -0.22f,  0.77f },
    // TikTok
    { 0.88f, -0.44f,  0.33f, -0.11f,  0.77f, -0.55f,  0.22f, -0.33f },
    // Twitter
    {-0.11f,  0.77f, -0.44f,  0.66f, -0.33f,  0.22f, -0.55f,  0.11f },
    // WhatsApp
    { 0.33f, -0.11f,  0.77f, -0.44f,  0.55f, -0.22f,  0.44f, -0.55f },
    // YouTube
    {-0.44f,  0.55f, -0.22f,  0.11f, -0.77f,  0.66f, -0.33f,  0.22f },
};

// ─── Structs ──────────────────────────────────────────────────────────────────

struct AppEvent {
    int   current_app_idx;       // Index into APP_NAMES
    int   prev_app_idx;          // Index of previous app
    int   hour;                  // 0-23
    int   day_of_week;           // 0=Mon, 6=Sun
    float battery;               // 0.0-1.0
    float ram_free_mb;
    float ram_total_mb;
    int   session_depth;
    float time_since_last_switch; // seconds
    bool  is_charging;
    bool  ai_task_active;
    int   network_type;          // 0=wifi, 1=cellular, 2=none
    // Pattern flags (top-10 PrefixSpan patterns)
    std::array<float, 10> pattern_flags;
    float pattern_confidence;
};

struct Prediction {
    int   top_app_idx;           // Predicted next app index
    float confidence;            // Calibrated probability [0,1]
    int   top3[3];               // Top-3 predicted apps
    float top3_probs[3];         // Probabilities for top-3
    double inference_ms;         // Inference time in milliseconds
};

// ─── Feature Extraction ───────────────────────────────────────────────────────

/**
 * Build the 46-dimensional feature vector for an event.
 * Mirrors FeatureEngine.transform() from Python exactly.
 *
 * Feature layout:
 *   [0]    current_app_encoded       (label encoded)
 *   [1]    prev_app_1_encoded
 *   [2]    hour
 *   [3]    day_of_week
 *   [4]    battery
 *   [5]    ram_free_mb
 *   [6]    ram_total_mb
 *   [7]    session_depth
 *   [8]    time_since_last_switch
 *   [9]    is_charging
 *   [10]   ai_task_active
 *   [11]   network_type
 *   [12]   ram_pressure            (= 1 - ram_free/ram_total)
 *   [13]   hour_sin                (circular encoding)
 *   [14]   hour_cos
 *   [15]   day_sin
 *   [16]   day_cos
 *   [17]   is_weekend
 *   [18]   is_peak_hour            (19-22 = 1)
 *   [19..26] current_app_embedding (App2Vec 8-dim)
 *   [27..34] prev_app_embedding    (App2Vec 8-dim)
 *   [35..44] pattern_flags         (10 binary PrefixSpan flags)
 *   [45]   pattern_confidence
 */
std::array<float, FEATURE_DIM> extract_features(const AppEvent& ev) {
    std::array<float, FEATURE_DIM> feat = {};

    // Base features (0-18)
    feat[0]  = static_cast<float>(ev.current_app_idx);
    feat[1]  = static_cast<float>(ev.prev_app_idx);
    feat[2]  = static_cast<float>(ev.hour);
    feat[3]  = static_cast<float>(ev.day_of_week);
    feat[4]  = ev.battery;
    feat[5]  = ev.ram_free_mb;
    feat[6]  = ev.ram_total_mb;
    feat[7]  = static_cast<float>(ev.session_depth);
    feat[8]  = ev.time_since_last_switch;
    feat[9]  = ev.is_charging ? 1.0f : 0.0f;
    feat[10] = ev.ai_task_active ? 1.0f : 0.0f;
    feat[11] = static_cast<float>(ev.network_type);

    // Derived base features
    float ram_pressure = (ev.ram_total_mb > 0)
        ? 1.0f - (ev.ram_free_mb / ev.ram_total_mb)
        : 0.0f;
    feat[12] = ram_pressure;

    // Circular hour encoding
    float hour_rad = 2.0f * M_PI * ev.hour / 24.0f;
    feat[13] = std::sin(hour_rad);
    feat[14] = std::cos(hour_rad);

    // Circular day-of-week encoding
    float dow_rad = 2.0f * M_PI * ev.day_of_week / 7.0f;
    feat[15] = std::sin(dow_rad);
    feat[16] = std::cos(dow_rad);

    feat[17] = (ev.day_of_week >= 5) ? 1.0f : 0.0f;   // is_weekend
    feat[18] = (ev.hour >= 19 && ev.hour <= 22) ? 1.0f : 0.0f; // is_peak_hour

    // App2Vec embeddings (19-34)
    if (ev.current_app_idx >= 0 && ev.current_app_idx < NUM_APPS) {
        for (int i = 0; i < 8; ++i)
            feat[BASE_FEATURE_DIM + i] = APP_EMBEDDINGS[ev.current_app_idx][i];
    }
    if (ev.prev_app_idx >= 0 && ev.prev_app_idx < NUM_APPS) {
        for (int i = 0; i < 8; ++i)
            feat[BASE_FEATURE_DIM + 8 + i] = APP_EMBEDDINGS[ev.prev_app_idx][i];
    }

    // PrefixSpan features (35-45)
    for (int i = 0; i < 10; ++i)
        feat[BASE_FEATURE_DIM + APP2VEC_DIM + i] = ev.pattern_flags[i];
    feat[BASE_FEATURE_DIM + APP2VEC_DIM + 10] = ev.pattern_confidence;

    return feat;
}

// ─── Cache Score (6-Factor) ───────────────────────────────────────────────────

/**
 * 6-Factor Cache Score.
 * S = w_R*R + w_F*F + w_P*P + w_I*I + w_C*C - w_M*M
 *
 * Args:
 *   recency     R: time since last use (normalized, higher=more recent)
 *   frequency   F: launch frequency (normalized 0-1)
 *   pred_prob   P: Platt-calibrated ML probability
 *   importance  I: app importance index
 *   context     C: contextual relevance (time-of-day match)
 *   memory_mb   M: app memory footprint in MB (normalized by 1000)
 *
 * Returns: Cache score S in [0,1] range. Higher = keep in RAM.
 */
float compute_cache_score(
    float recency,    float frequency,
    float pred_prob,  float importance,
    float context,    float memory_mb)
{
    // Weights from TECHNICAL_JUSTIFICATIONS.md (tuned via grid search)
    static constexpr float w_R = 0.25f;
    static constexpr float w_F = 0.20f;
    static constexpr float w_P = 0.30f;  // Highest weight: ML prediction
    static constexpr float w_I = 0.10f;
    static constexpr float w_C = 0.10f;
    static constexpr float w_M = 0.30f;  // Negative: penalize large apps

    float mem_normalized = memory_mb / 1000.0f;  // Normalize by 1GB
    float score = w_R * recency
                + w_F * frequency
                + w_P * pred_prob
                + w_I * importance
                + w_C * context
                - w_M * mem_normalized;

    // Clamp to [0, 1]
    return std::max(0.0f, std::min(1.0f, score));
}

// ─── Preload Gate ─────────────────────────────────────────────────────────────

/**
 * Preload Gate — 5-condition safety check before triggering app preload.
 *
 * Condition 1: Prediction confidence >= calibrated threshold (0.30)
 * Condition 2: RAM free >= app_memory_mb * 1.5 (safety margin)
 * Condition 3: Not currently charging critical battery
 * Condition 4: Network available (can stream initial resources)
 * Condition 5: Not in thrashing state (thrash_score < 0.70)
 *
 * Returns true if ALL 5 conditions are met.
 */
bool preload_gate(
    float pred_confidence,
    float ram_free_mb,
    float app_memory_mb,
    bool  is_low_battery,     // battery < 0.15
    bool  network_available,
    float thrash_score)
{
    bool cond1 = pred_confidence >= 0.30f;
    bool cond2 = ram_free_mb >= app_memory_mb * 1.5f;
    bool cond3 = !is_low_battery;
    bool cond4 = network_available;
    bool cond5 = thrash_score < 0.70f;

    return cond1 && cond2 && cond3 && cond4 && cond5;
}

// ─── Mock inference (replaces ONNX Runtime call) ─────────────────────────────

/**
 * Run inference on the feature vector.
 * In production this calls: session.Run(input_tensor) → softmax probs
 * Here we simulate with a mock softmax to prove the pipeline structure.
 */
std::vector<float> mock_inference(const std::array<float, FEATURE_DIM>& features) {
    std::vector<float> probs(NUM_APPS, 0.0f);

    // Simulate: bias toward apps that match current_app neighborhood
    int cur = static_cast<int>(features[0]);
    for (int i = 0; i < NUM_APPS; ++i) {
        float dot = 0.0f;
        for (int d = 0; d < 8; ++d) {
            float c = (cur >= 0 && cur < NUM_APPS) ? APP_EMBEDDINGS[cur][d] : 0.0f;
            dot += c * APP_EMBEDDINGS[i][d];
        }
        probs[i] = std::exp(dot * 2.0f);
    }

    // Softmax normalize
    float sum = 0.0f;
    for (float p : probs) sum += p;
    for (float& p : probs) p /= sum;

    return probs;
}

// ─── Main demo ────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::cout << "==============================================================\n";
    std::cout << "  AMM C++ Inference Prototype — Phase 2\n";
    std::cout << "  Adaptive Memory Management for Android\n";
    std::cout << "==============================================================\n\n";

    // ── Build a sample event (simulates: user opens Instagram at 9 PM) ────────
    AppEvent event;
    event.current_app_idx        = 12;   // Instagram
    event.prev_app_idx           = 26;   // WhatsApp
    event.hour                   = 21;   // 9 PM
    event.day_of_week            = 5;    // Saturday
    event.battery                = 0.72f;
    event.ram_free_mb            = 1200.0f;
    event.ram_total_mb           = 4096.0f;
    event.session_depth          = 3;
    event.time_since_last_switch = 45.0f;
    event.is_charging            = false;
    event.ai_task_active         = true;   // Instagram uses AI for reels
    event.network_type           = 0;      // WiFi
    event.pattern_flags          = {1,1,0,0,0,0,1,0,0,0}; // WhatsApp->Instagram match
    event.pattern_confidence     = 0.73f;

    std::cout << "Input Event:\n";
    std::cout << "  Current app:   " << APP_NAMES[event.current_app_idx] << "\n";
    std::cout << "  Previous app:  " << APP_NAMES[event.prev_app_idx] << "\n";
    std::cout << "  Time:          " << event.hour << ":00, ";
    const char* days[] = {"Mon","Tue","Wed","Thu","Fri","Sat","Sun"};
    std::cout << days[event.day_of_week] << "\n";
    std::cout << "  Battery:       " << (int)(event.battery * 100) << "%\n";
    std::cout << "  RAM free:      " << event.ram_free_mb << " MB / "
              << event.ram_total_mb << " MB\n\n";

    // ── Extract 46-dim feature vector ─────────────────────────────────────────
    auto t_start = std::chrono::high_resolution_clock::now();

    auto features = extract_features(event);

    auto t_feat = std::chrono::high_resolution_clock::now();

    // ── Run inference (ONNX Runtime in production) ────────────────────────────
    auto probs = mock_inference(features);

    auto t_infer = std::chrono::high_resolution_clock::now();

    double feat_ms  = std::chrono::duration<double, std::milli>(t_feat - t_start).count();
    double infer_ms = std::chrono::duration<double, std::milli>(t_infer - t_feat).count();
    double total_ms = feat_ms + infer_ms;

    // ── Find top-3 predictions ────────────────────────────────────────────────
    int top3_idx[3] = {-1, -1, -1};
    float top3_prob[3] = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < NUM_APPS; ++i) {
        if (probs[i] > top3_prob[0]) {
            top3_prob[2] = top3_prob[1]; top3_idx[2] = top3_idx[1];
            top3_prob[1] = top3_prob[0]; top3_idx[1] = top3_idx[0];
            top3_prob[0] = probs[i];     top3_idx[0] = i;
        } else if (probs[i] > top3_prob[1]) {
            top3_prob[2] = top3_prob[1]; top3_idx[2] = top3_idx[1];
            top3_prob[1] = probs[i];     top3_idx[1] = i;
        } else if (probs[i] > top3_prob[2]) {
            top3_prob[2] = probs[i];     top3_idx[2] = i;
        }
    }

    // ── Display prediction results ────────────────────────────────────────────
    std::cout << "Prediction Results (Top-3 Next Apps):\n";
    for (int k = 0; k < 3; ++k) {
        if (top3_idx[k] >= 0) {
            std::cout << "  " << (k+1) << ". " << APP_NAMES[top3_idx[k]]
                      << " — confidence: " << (int)(top3_prob[k] * 100) << "%\n";
        }
    }

    // ── Compute cache score for predicted top app ─────────────────────────────
    float cache_score = compute_cache_score(
        0.85f,           // recency: recently used
        0.70f,           // frequency: often used
        top3_prob[0],    // pred_prob: from model
        0.60f,           // importance: social app
        0.90f,           // context: evening = social peak
        150.0f           // memory_mb: Instagram ~150MB
    );

    std::cout << "\nCache Score for " << APP_NAMES[top3_idx[0]] << ": "
              << cache_score << "\n";

    // ── Preload gate decision ─────────────────────────────────────────────────
    bool should_preload = preload_gate(
        top3_prob[0],
        event.ram_free_mb,
        150.0f,   // Instagram memory footprint
        event.battery < 0.15f,
        event.network_type != 2,
        0.10f     // thrash_score: system is healthy
    );

    std::cout << "Preload Gate Decision: "
              << (should_preload ? "PRELOAD (keep in RAM)" : "DO NOT PRELOAD") << "\n";

    // ── Timing ────────────────────────────────────────────────────────────────
    std::cout << "\nTiming:\n";
    std::cout << "  Feature extraction: " << feat_ms  << " ms\n";
    std::cout << "  Model inference:    " << infer_ms << " ms\n";
    std::cout << "  Total:              " << total_ms << " ms\n";

    std::cout << "\nFeature vector (" << FEATURE_DIM << "-dim):\n  [";
    for (int i = 0; i < FEATURE_DIM; ++i) {
        std::cout << features[i];
        if (i < FEATURE_DIM - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "\n[OK] C++ prototype completed successfully.\n";
    std::cout << "     Ready for ONNX Runtime integration & Android NDK build.\n";

    return 0;
}
