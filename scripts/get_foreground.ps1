Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WinAPI {
    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")]
    public static extern int GetWindowThreadProcessId(IntPtr hWnd, out int lpdwProcessId);
}
"@

$hwnd = [WinAPI]::GetForegroundWindow()
$pid_val = 0
[void][WinAPI]::GetWindowThreadProcessId($hwnd, [ref]$pid_val)

if ($pid_val -gt 0) {
    $proc = Get-Process -Id $pid_val -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Output "$($proc.ProcessName)|$($proc.MainWindowTitle)"
    }
}
