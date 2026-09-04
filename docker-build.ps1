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
    [string]$LogDir = '',

    # Claim the release tags (:<version>, :latest, :<version>-cuNNN). OFF by
    # default: a validation build must not name a release. See the tagging
    # section below for what this cost when it defaulted the other way.
    [switch]$ReleaseTags
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

# Second look at the same artifact, and it is WEAK BY CONSTRUCTION -- read the
# next paragraph before trusting it for anything.
#
# It reads `app.__version__` out of the image and compares it to the version
# being applied. Both come from the same tree, so within a bump window it
# matches for EVERY build and can only ever pass: the image is compared against
# itself. On 2026-09-04 that self-reference actively licensed a wrong tag --
# `0.8.0-beta.1` was applied to an image 40 commits past the public git tag of
# that name, and this line printed "verified" while it happened.
# It is retained only to catch an image from a DIFFERENT release line (a stale
# base, a wrong scratch tag). The HEAD assertion above and the git-tag
# agreement check below are the real guards; this one must never be mistaken
# for either.
$imgVersion = (& docker run --rm --entrypoint python $scratchTag `
    -c "import sys; sys.path.insert(0, '/app/backend'); import app; print(app.__version__)" `
    2>&1 | Select-Object -Last 1)
$imgVersion = "$imgVersion".Trim()
if ($imgVersion -ne $Version) {
    throw ("Version mismatch: image reports '$imgVersion', expected '$Version'. " +
           'No release tag was applied.')
}
Write-Host "[build] verified: image version == $imgVersion"

# -- Release tags: opt IN, never by default -----------------------------------
# Most builds are validation builds. They get inspected, smoke-tested and then
# thrown away, and they have no business naming a release. Defaulting to ON
# meant a validation build silently claimed `:latest` and a version tag, which
# is a loaded gun sitting next to `docker push`. Naming a release is a release
# action and now requires saying so.
if (-not $ReleaseTags) {
    Write-Host ''
    Write-Host "[build] DONE (validation). $Variant verified at $($GitSha.ToLower())."
    Write-Host "[build] Image is $scratchTag -- NO release tag was applied."
    Write-Host '[build] Re-run with -ReleaseTags to claim the release tags.'
    exit 0
}

# The invariant the version check above was reaching for and missing: a release
# tag `X` may only name an image whose commit agrees with the git tag `vX`.
# `v$Version` is public once pushed, so it already denotes a specific tree;
# labelling a different tree with the same name produces two commits under one
# version, which is the exact defect this wrapper exists to prevent. Checked
# against $GitSha, which the artifact has already been proven to contain.
# Where no `v$Version` exists the name is unclaimed and applying it is fine --
# this fails closed without depending on anyone's bump discipline.
# Asked of ORIGIN, not of this checkout, for two reasons. On the merits: what
# makes `v$Version` binding is that it is PUBLIC, so the remote is the
# authority and whatever tags this clone happens to hold is not. And
# defensively: a local `git rev-list -n 1 v$Version` returns empty with exit
# 128 both when the tag does not exist AND when it exists but was never
# fetched -- measured, the two are byte-identical -- so a fresh clone or a
# shallow CI checkout (actions/checkout fetches no tags by default) would take
# the "unclaimed" branch and tag anyway. That is "could not check" reported as
# "check passed", the failure this whole wrapper exists to refuse.
# Three outcomes, kept apart exactly as docker-verify-commit.ps1 keeps its own:
# a line -> compare; empty with exit 0 -> genuinely unclaimed; non-zero exit ->
# UNDETERMINED, which must refuse rather than proceed.
$previousEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    $lsOut = & git -C $PSScriptRoot ls-remote --tags origin "refs/tags/v$Version" 2>&1
    $lsExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousEap
}
if ($lsExit -ne 0) {
    Write-Host ''
    Write-Host "[build] REFUSING TO TAG -- could not ask origin whether v$Version exists."
    Write-Host "[build]   git ls-remote exit $lsExit : $(($lsOut | Out-String).Trim())"
    Write-Host '[build] UNDETERMINED is not the same as unclaimed. No release tag was applied.'
    throw "Could not determine whether the release tag v$Version is already claimed."
}

# Annotated tags report the tag OBJECT on `refs/tags/vX` and the commit it
# points at on `refs/tags/vX^{}`; lightweight tags report only the first, which
# is already the commit. Prefer the dereferenced line when present, or an
# annotated tag would compare a tag-object sha against a commit sha and refuse
# every time.
$tagCommit = ''
foreach ($line in (($lsOut | Out-String) -split "`n")) {
    if ($line -match '^([0-9a-f]{40})\s+refs/tags/\S+\^\{\}\s*$') { $tagCommit = $matches[1]; break }
    if ($line -match '^([0-9a-f]{40})\s+refs/tags/\S+\s*$') { $tagCommit = $matches[1] }
}

if ($tagCommit -ne '') {
    if ($tagCommit -ne $GitSha.ToLower()) {
        Write-Host ''
        Write-Host "[build] REFUSING TO TAG -- git tag v$Version does not name this commit."
        Write-Host "[build]   git tag v$Version -> $tagCommit"
        Write-Host "[build]   image contains     -> $($GitSha.ToLower())"
        Write-Host '[build] Tagging this image would put two different commits under one'
        Write-Host '[build] version name. Bump the version, or build the tagged commit.'
        throw "Release tag v$Version already denotes a different commit."
    }
    Write-Host "[build] verified: origin's tag v$Version agrees with the image's commit"
} else {
    Write-Host "[build] note: origin has no tag v$Version; the name is unclaimed"
}

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
