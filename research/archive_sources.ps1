param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$archiveRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'original'))
if (-not $archiveRoot.StartsWith($repoRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Archive target is outside the clean repository.'
}

$entries = @()
foreach ($group in @('Evidence', 'Experiment', 'VAE', 'YOLOv26')) {
    $sourceDir = Join-Path $SourceRoot $group
    if (-not (Test-Path -LiteralPath $sourceDir -PathType Container)) {
        throw "Missing original research directory: $sourceDir"
    }
    $targetDir = Join-Path $archiveRoot $group
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    foreach ($source in Get-ChildItem -LiteralPath $sourceDir -File -Filter '*.py') {
        $target = Join-Path $targetDir $source.Name
        $sourceHash = (Get-FileHash -LiteralPath $source.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if (Test-Path -LiteralPath $target) {
            $targetHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($sourceHash -ne $targetHash) { throw "Archive collision: $target" }
        } else {
            Copy-Item -LiteralPath $source.FullName -Destination $target
        }
        $copiedHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($copiedHash -ne $sourceHash) { throw "Archive hash mismatch: $target" }
        $entries += [pscustomobject]@{
            original = "$group/$($source.Name)"
            archive = "research/original/$group/$($source.Name)"
            bytes = $source.Length
            sha256 = $sourceHash
            status = 'historical-source'
        }
    }
}
$manifestPath = Join-Path $PSScriptRoot 'source_manifest.json'
$entries | Sort-Object original | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $manifestPath -Encoding utf8
"Archived $($entries.Count) original research scripts with matching SHA-256 hashes."
