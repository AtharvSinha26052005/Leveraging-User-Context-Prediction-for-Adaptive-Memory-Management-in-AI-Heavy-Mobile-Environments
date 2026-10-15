# Bug Trail — Every Bug from Start to Finish

> A log anyone can pick up cold and understand what broke and why.

---

## BUG-001: MarkovPredictor `order` parameter
**Date:** 2026-09-14  
**File:** `src/prediction/model_comparison.py:136`  
**Symptom:** `TypeError: MarkovPredictor.__init__() got an unexpected keyword argument 'order'`  
**Cause:** `model_comparison.py` was calling `MarkovPredictor(order=1)` but `MarkovPredictor.__init__()` takes no arguments.  
**Fix:** Changed to `MarkovPredictor()`  
**Status:** ✅ Fixed

---

## BUG-002: MarkovPredictor `.fit()` signature mismatch
**Date:** 2026-09-14  
**File:** `src/prediction/model_comparison.py:139`  
**Symptom:** `TypeError: MarkovPredictor.fit() missing 1 required positional argument: 'next_apps'`  
**Cause:** `model_comparison.py` was calling `markov.fit(train_df)` but the real API is `markov.fit(current_apps: List[str], next_apps: List[str])`  
**Fix:** Changed to `markov.fit(train_df["current_app"].tolist(), train_df["next_app"].tolist())`  
**Status:** ✅ Fixed

---

## BUG-003: MarkovPredictor `.transitions` attribute
**Date:** 2026-09-14  
**File:** `src/prediction/model_comparison.py:167`  
**Symptom:** `AttributeError: 'MarkovPredictor' object has no attribute 'transitions'`  
**Cause:** The actual attribute is `transition_counts` not `transitions`.  
**Fix:** Changed to `markov.transition_counts`  
**Status:** ✅ Fixed

---

## BUG-004: Android 16 `mResumedActivity` not found
**Date:** 2026-09-14  
**File:** `scripts/collect_phone_data.py:100`  
**Symptom:** `get_foreground_app()` returned `None` on Realme 13 5G (Android 16).  
**Cause:** Android 16 renamed `mResumedActivity` to `topResumedActivity` in `dumpsys activity` output.  
**Fix:** Changed search pattern from `"mResumedActivity"` to `"ResumedActivity"` (matches both old and new).  
**Root cause:** Google refactored `ActivityTaskManagerService` in Android 14+ and renamed internal fields.  
**Status:** ✅ Fixed

---

## BUG-005: ADB not in PATH
**Date:** 2026-09-14  
**File:** `scripts/collect_phone_data.py`  
**Symptom:** `FileNotFoundError: 'adb' not found`  
**Cause:** ADB was downloaded to `Desktop\platform-tools\` but not added to system PATH.  
**Fix:** Used absolute path `os.path.join(os.environ["USERPROFILE"], "Desktop", "platform-tools", "adb.exe")` in all subprocess calls.  
**Status:** ✅ Fixed

---

## BUG-006: Laptop collector `GetForegroundWindow()` returns 0 in headless mode
**Date:** 2026-09-14  
**File:** `scripts/collect_laptop_data.py`  
**Symptom:** Collector reports 0 events when run from an IDE background process.  
**Cause:** `ctypes.windll.user32.GetForegroundWindow()` returns `NULL` when the calling process has no interactive window.  
**Fix:** Switched to PowerShell helper script (`get_foreground.ps1`) that uses `Add-Type` with P/Invoke — works in all contexts. Added fallback to `Get-Process` with MainWindowTitle filter.  
**Workaround:** User runs the script from a visible terminal window.  
**Status:** ✅ Fixed (with known limitation in fully headless environments)

---

## BUG-007: ADB `unauthorized` on first connect
**Date:** 2026-09-14  
**Symptom:** `adb devices` shows `unauthorized` instead of `device`.  
**Cause:** Phone had not yet accepted the RSA key prompt for USB debugging.  
**Fix:** User toggled USB debugging off/on, re-plugged cable, and accepted the "Allow USB debugging?" prompt with "Always allow from this computer" checked.  
**Status:** ✅ Fixed (user action required)
