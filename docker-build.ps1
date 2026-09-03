<#
.SYNOPSIS
    Build a release container image and PROVE it contains the requested commit
    before any release tag is allowed to point at it.

.DESCRIPTION
    On 2026-09-03 a published image shipped a commit nobody asked for: both
    variants were built with the same --build-arg GIT_SHA, one contained it and
    the other contained its predecessor -- which is why a crash "survived" its
    fix, the fix was simply never in the image under test.

    The Dockerfile already asserts `git rev-parse HEAD == $GIT_SHA`, and that
    assertion cannot catch this: it lives INSIDE the RUN, and a cache hit never
    re-runs the RUN. Nor does the exit code catch it -- the build that shipped
    the wrong commit exited 0, printed "writing image", and named its tags.

    So the only thing that proves an artifact is reading the artifact. This
    wrapper builds to a SCRATCH tag, reads HEAD out of the built image, and
    applies the release tags only if it matches. A mismatch therefore cannot
    move :latest -- the previous good image keeps it.

    The root cause of that build is still unnamed and this guard deliberately
    does not depend on it: it verifies the outcome, not the mechanism.

.EXAMPLE
    .\docker-build.ps1 -GitSha b987410c93b50e0e2cd678da981646a8edfc242d `
        -Variant cu128 -Version 0.8.0-beta.1 -TokenPath D:\docker\tmp\gh_token

.NOTES
    Never pushes. Publishing to the registry stays a separate, deliberate act.
#>
[CmdletBinding()]
param(
    # The FULL 40-character commit sha. Not a branch: a branch moves, and an
    # image that cannot name its commit is the defect this script exists for.
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$GitSha,

    [Parameter(Mandatory = $true)]
    [ValidateSet('cu128', 'cu126')]
    [string]$Variant,

    [Parameter(Mandatory = $true)]
    [string]$Version,

    [Parameter(Mandatory = $true)]
    [string]$TokenPath,

    # The build context. A worktree checked out at $GitSha by preference, so
    # the context matches the commit being built.
    [string]$Context = '.',

    [string]$Repository = 'mastermerlin/mrln-arcane-tuner',

    # Pinned Ollama release. Both or neither, enforced by the Dockerfile.
    [string]$OllamaVersion = '',
    [string]$OllamaSha256 = '',

    # Bypass the layer cache for the whole build. The clone layer is the one
    # that went wrong; --no-cache removes the ambiguity for a real release cut
    # at the cost of a long build.
    [switch]$NoCache,

    # Where the build log and argument vector are written. Overridable so the
    # guard's own tests can run without depositing artifacts in the repo.
    [string]$LogDir = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not (Test-Path -LiteralPath $TokenPath)) {
    throw "Git token file not found: $TokenPath"
}

$cudaBase = '12.8.1'
if ($Variant -eq 'cu126') { $cudaBase = '12.6.3' }

# Deliberately NOT a release tag. Nothing that :latest or the version tag
# points at may be touched until the artifact has proven its commit, so the
# build writes here and the release tags are applied afterwards or never.
$scratchTag = "mrln-build-scratch:$Variant-$($GitSha.Substring(0, 12))"

$buildArgs = @(
    'build',
    '--progress=plain',
    '--secret', "id=git_token,src=$TokenPath",
    '--build-arg', "GIT_SHA=$GitSha",
    '--build-arg', "CUDA_BASE=$cudaBase",
    '--build-arg', "TORCH_CUDA=$Variant"
)
if ($OllamaVersion -ne '' -or $OllamaSha256 -ne '') {
    if ($OllamaVersion -eq '' -or $OllamaSha256 -eq '') {
        throw 'OllamaVersion and OllamaSha256 must be given together or not at all.'
    }
    $buildArgs += @('--build-arg', "OLLAMA_VERSION=$OllamaVersion",
                    '--build-arg', "OLLAMA_SHA256=$OllamaSha256")
}
if ($NoCache) { $buildArgs += '--no-cache' }
$buildArgs += @('-t', $scratchTag, $Context)

# -- Persist the evidence BEFORE building -------------------------------------
# The whole LANE-82 investigation existed because one build left no log: four
# mechanisms were proposed and not one could be tested against the actual
# event, because "which sha was passed" rested on a hand-written note rather
# than on a record. A wrapper that detects a bad artifact without recording how
# it was produced makes the next anomaly detectable but still undiagnosable, so
# the argument vector and the full --progress=plain output are written to disk
# as a matter of course, not on demand.
$logDir = $LogDir
if ($logDir -eq '') { $logDir = Join-Path $PSScriptRoot '.agent\workdir' }
if (-not (Test-Path -LiteralPath $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$slug = "$Variant-$($GitSha.Substring(0, 12))-$stamp"
$argvPath = Join-Path $logDir "build-$slug.argv.txt"
$logPath = Join-Path $logDir "build-$slug.log"

# The secret is a PATH, never a value, so the argv record carries no token.
@(
    "timestamp   : $(Get-Date -Format 'o')",
    "git_sha     : $GitSha",
    "variant     : $Variant",
    "version     : $Version",
    "context     : $((Resolve-Path -LiteralPath $Context).Path)",
    "scratch_tag : $scratchTag",
    "no_cache    : $([bool]$NoCache)",
    '',
    'docker ' + ($buildArgs -join ' ')
) | Set-Content -LiteralPath $argvPath -Encoding utf8

Write-Host "[build] $Variant from $GitSha -> $scratchTag"
Write-Host "[build] argv  -> $argvPath"
Write-Host "[build] log   -> $logPath"

# Windows PowerShell turns a native command's stderr into ErrorRecords, which
# under `$ErrorActionPreference = 'Stop'` aborts the script on output docker
# writes routinely. The exit code stays truthful, so relax the preference for
# the build alone and decide on $LASTEXITCODE.
$previousEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & docker @buildArgs 2>&1 | Tee-Object -FilePath $logPath
    $buildExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousEap
}

if ($buildExit -ne 0) {
    # A cheap extra check, NOT the guard: the build that shipped the wrong
    # commit exited 0. The verification below is what actually decides.
    throw "docker build failed with exit code $buildExit. No release tag was applied. Log: $logPath"
}

# -- The guard: read the commit out of the ARTIFACT, not out of the build log --
# Delegated to docker-verify-commit.ps1 rather than written inline, so the
# assertion can be exercised in seconds against any image on the machine
# instead of only at the end of a 40-minute build. Its three exit codes are
# kept apart here too: 2 ("could not check") must never be handled as 0.
$verifier = Join-Path $PSScriptRoot 'docker-verify-commit.ps1'
if (-not (Test-Path -LiteralPath $verifier)) {
    throw "Verifier not found at $verifier. Refusing to tag an unverified image."
}
& $verifier -Image $scratchTag -ExpectedSha $GitSha
$verifyExit = $LASTEXITCODE

if ($verifyExit -eq 2) {
    throw ("Could not read a commit out of $scratchTag -- the image is UNVERIFIED, " +
           "which is not the same as correct. No release tag was applied. Log: $logPath")
}
if ($verifyExit -ne 0) {
    Write-Host ''
    Write-Host '[build] REFUSING TO TAG -- the image does not contain the requested commit.'
    Write-Host '[build] The release tags are untouched and still point at the previous image.'
    Write-Host "[build] Re-run with -NoCache, or inspect $scratchTag before deleting it."
    Write-Host "[build] Build log: $logPath"
    throw 'Artifact commit mismatch.'
}

# Second, independent check on the same artifact: the version the image will
# claim. Defence in depth against a DIFFERENT failure -- an image built from
# another release line -- and explicitly NOT a second opinion on the defect
# above: `app.__version__` was identical in the wrong commit and the right one,
# so this check would have passed that build in silence. The HEAD assertion is
# the guard; this one must never be mistaken for it.
$imgVersion = (& docker run --rm --entrypoint python $scratchTag `
    -c "import sys; sys.path.insert(0, '/app/backend'); import app; print(app.__version__)" `
    2>&1 | Select-Object -Last 1)
$imgVersion = "$imgVersion".Trim()
if ($imgVersion -ne $Version) {
    throw ("Version mismatch: image reports '$imgVersion', expected '$Version'. " +
           'No release tag was applied.')
}
Write-Host "[build] verified: image version == $imgVersion"

# -- Only now may a release tag move ------------------------------------------
$tags = @("${Repository}:$Version-$Variant")
if ($Variant -eq 'cu128') {
    # cu128 is the default variant: it owns the bare version tag and :latest.
    $tags = @("${Repository}:$Version", "${Repository}:latest")
}
foreach ($t in $tags) {
    & docker tag $scratchTag $t
    if ($LASTEXITCODE -ne 0) { throw "docker tag failed for $t" }
    Write-Host "[build] tagged $t"
}

& docker rmi $scratchTag | Out-Null

Write-Host ''
Write-Host "[build] DONE. $Variant verified at $($GitSha.ToLower()), version $imgVersion."
Write-Host '[build] Nothing was pushed. Publish deliberately, after the user says so.'
