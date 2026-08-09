[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('MELD', 'EmotionTalk')]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [string]$Destination,

    [switch]$IncludeMedia,
    [switch]$IncludeTest,
    [switch]$ListOnly
)

$ErrorActionPreference = 'Stop'
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $scriptDirectory '..\..'))
$destinationPath = [IO.Path]::GetFullPath($Destination)

function Assert-Sha256 {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected
    )
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for ${Path}: expected $Expected, got $actual"
    }
    Write-Output "VERIFIED $Path"
}

function Download-CurlResume {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Output
    )
    $outputDirectory = Split-Path -Parent $Output
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    & curl.exe `
        --fail `
        --location `
        --continue-at - `
        --retry 10 `
        --retry-all-errors `
        --retry-delay 2 `
        --connect-timeout 20 `
        --output $Output `
        $Url
    if ($LASTEXITCODE -ne 0) {
        throw "curl download failed with exit code $LASTEXITCODE`: $Url"
    }
}

function Download-SmallFile {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Output
    )
    $outputDirectory = Split-Path -Parent $Output
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    for ($attempt = 1; $attempt -le 5; $attempt += 1) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Output
            return
        }
        catch {
            if ($attempt -eq 5) {
                throw
            }
            Write-Warning "Small-file download attempt $attempt failed; retrying: $Url"
            Start-Sleep -Seconds (2 * $attempt)
        }
    }
}

if ($Dataset -eq 'MELD') {
    $revision = 'e8cedf27b5d2877e198332c957127e16eb214afe'
    $annotationRoot = Join-Path $destinationPath 'annotations'
    $annotationSpecs = @(
        [pscustomobject]@{ Name = 'train_sent_emo.csv'; Sha256 = 'd2fa2d6529cf03cac2989efec05c9b27d8fd2f4c8fc5974c7ae88aa537fa02db'; Role = 'development' },
        [pscustomobject]@{ Name = 'dev_sent_emo.csv'; Sha256 = '2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c'; Role = 'development' },
        [pscustomobject]@{ Name = 'test_sent_emo.csv'; Sha256 = '8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888'; Role = 'evaluator-only' }
    )

    foreach ($spec in $annotationSpecs) {
        if ($spec.Role -eq 'evaluator-only' -and -not $IncludeTest) {
            continue
        }
        $url = "https://raw.githubusercontent.com/declare-lab/MELD/$revision/data/MELD/$($spec.Name)"
        $output = Join-Path $annotationRoot $spec.Name
        if ($ListOnly) {
            Write-Output "$($spec.Role) $url -> $output"
            continue
        }
        Download-SmallFile -Url $url -Output $output
        Assert-Sha256 -Path $output -Expected $spec.Sha256
    }

    if ($IncludeMedia) {
        $mediaUrl = 'https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz'
        $mediaOutput = Join-Path $destinationPath 'MELD.Raw.tar.gz'
        if ($ListOnly) {
            Write-Output "media $mediaUrl -> $mediaOutput"
        }
        else {
            Download-CurlResume -Url $mediaUrl -Output $mediaOutput
            $expectedBytes = 10878146150
            $actualBytes = (Get-Item -LiteralPath $mediaOutput).Length
            if ($actualBytes -ne $expectedBytes) {
                throw "Unexpected MELD.Raw.tar.gz size: expected $expectedBytes, got $actualBytes"
            }
            Write-Warning 'MELD media size passed, but the official endpoint has no frozen SHA-256 in this repository. Record an internal SHA-256 before confirmation use.'
        }
    }
    exit 0
}

if ($IncludeTest) {
    Write-Warning 'EmotionTalk archives bundle data roles; -IncludeTest has no separate effect. Keep test labels evaluator-only after download.'
}

$python = (Get-Command python -ErrorAction Stop).Source
$downloader = Join-Path $repositoryRoot 'experiment\scripts\download_hf_gated_resume.py'
$revision = 'adbc17fc944e8cf2873643906160c6ca0259ab61'
$specs = @(
    [pscustomobject]@{ Name = 'Text.tar'; Bytes = 1477776; Sha256 = 'facf5cbf3edb4104f5e1e8c450730af5524fe6170865a6930773e7fb6f82a603'; Media = $false },
    [pscustomobject]@{ Name = 'Video.tar'; Bytes = 526286; Sha256 = '01f54fb23fa0838ddbfccadbf1f883e0cc9bf6cf80c3b8d2959bf23a0e127996'; Media = $false },
    [pscustomobject]@{ Name = 'Audio.tar'; Bytes = 14811722752; Sha256 = 'f8599a70f55f489c948cd079c712acf340cf496c5267b14fb34e985539d57e41'; Media = $true },
    [pscustomobject]@{ Name = 'Multimodal.tar'; Bytes = 21294498304; Sha256 = '905af99e277658eedceefcf7ee283e8041ef62a0335245e140ea953fba45777e'; Media = $true }
)

foreach ($spec in $specs) {
    if ($spec.Media -and -not $IncludeMedia) {
        continue
    }
    $output = Join-Path $destinationPath $spec.Name
    if ($ListOnly) {
        Write-Output "gated BAAI/Emotiontalk@$revision/$($spec.Name) bytes=$($spec.Bytes) -> $output"
        continue
    }
    New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null
    & $python $downloader `
        --repo 'BAAI/Emotiontalk' `
        --filename $spec.Name `
        --revision $revision `
        --output $output `
        --size $spec.Bytes `
        --sha256 $spec.Sha256
    if ($LASTEXITCODE -ne 0) {
        throw "EmotionTalk download failed for $($spec.Name) with exit code $LASTEXITCODE"
    }
}

$cardUrl = "https://huggingface.co/datasets/BAAI/Emotiontalk/resolve/$revision/README.md"
$cardOutput = Join-Path $destinationPath 'README.md'
$attributesUrl = "https://huggingface.co/datasets/BAAI/Emotiontalk/resolve/$revision/.gitattributes"
$attributesOutput = Join-Path $destinationPath '.gitattributes'
if ($ListOnly) {
    Write-Output "dataset-card $cardUrl -> $cardOutput"
    Write-Output "metadata $attributesUrl -> $attributesOutput"
}
else {
    Download-SmallFile -Url $cardUrl -Output $cardOutput
    Assert-Sha256 -Path $cardOutput -Expected 'b66e67db8c703700c6174bd542732cc20499217295386e7c5d124d8264033e49'
    Download-SmallFile -Url $attributesUrl -Output $attributesOutput
    Assert-Sha256 -Path $attributesOutput -Expected 'e7a120ab07b1bc5b486be249e9fc6c83d59448d0093e1dfebe95d1566d9cafc0'
}
