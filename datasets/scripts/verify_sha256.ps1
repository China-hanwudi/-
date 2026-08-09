[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,

    [Parameter(Mandatory = $true)]
    [string]$Root
)

$ErrorActionPreference = 'Stop'

$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$rootPath = (Resolve-Path -LiteralPath $Root).Path
$rootPrefix = $rootPath.TrimEnd(
    [IO.Path]::DirectorySeparatorChar,
    [IO.Path]::AltDirectorySeparatorChar
) + [IO.Path]::DirectorySeparatorChar
$failures = 0
$checked = 0

foreach ($line in Get-Content -LiteralPath $manifestPath -Encoding UTF8) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith('#')) {
        continue
    }
    if ($trimmed -notmatch '^([0-9a-fA-F]{64})\s{2}(.+)$') {
        throw "Malformed SHA-256 manifest line: $line"
    }

    $expected = $Matches[1].ToLowerInvariant()
    $relative = $Matches[2].Replace('/', [IO.Path]::DirectorySeparatorChar)
    $candidate = [IO.Path]::GetFullPath((Join-Path $rootPath $relative))
    if (-not $candidate.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest path escapes the data root: $relative"
    }

    $checked += 1
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        Write-Error "MISSING  $relative" -ErrorAction Continue
        $failures += 1
        continue
    }

    $actual = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        Write-Error "MISMATCH $relative expected=$expected actual=$actual" -ErrorAction Continue
        $failures += 1
        continue
    }
    Write-Output "VERIFIED $relative"
}

if ($checked -eq 0) {
    throw 'The manifest contained no checksum entries.'
}
if ($failures -gt 0) {
    throw "SHA-256 verification failed for $failures of $checked files."
}

Write-Output "All $checked files passed SHA-256 verification."
