<#
.SYNOPSIS
    Read the commit out of a built container image and compare it to the one
    that was asked for.

.DESCRIPTION
    This is the guard behind docker-build.ps1, kept separate and callable on
    its own for one reason: an assertion you can only exercise by running a
    40-minute build is an assertion that will not be exercised. Given an image
    reference and an expected sha it can be run in seconds against any image on
    the machine, which is what makes it testable.

    It checks ONE condition -- "the artifact's HEAD is not the sha I asked for"
    -- and deliberately does not care how that came about. The defect it exists
    for was never diagnosed; a guard defined by a mechanism would only cover
    the mechanism someone guessed.

    Three outcomes, kept distinct on purpose:

        0  verified    a 40-hex sha was read and it matches
        1  mismatch    a 40-hex sha was read and it does NOT match
        2  unverified  the check could not be performed at all

    2 must never collapse into 0. "I could not check" silently becoming "the
    check passed" is the failure this whole lane is about, so the success path
    asserts a SUCCESSFUL READ of a 40-hex sha rather than merely the absence of
    a mismatch -- an empty result, a missing git, a refused safe.directory or an
    image that will not start are all failures, not clean bills of health.

.EXAMPLE
    .\docker-verify-commit.ps1 -Image mastermerlin/mrln-arcane-tuner:0.8.0-beta.1 `
        -ExpectedSha f1cbbbcfcab038cbdb559bc15278ca68e6f2a0ae
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Image,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$ExpectedSha,

    # Print nothing; communicate through the exit code alone.
    [switch]$Quiet
)

Set-StrictMode -Version Latest
# NOT 'Stop': docker writes to stderr routinely and Windows PowerShell turns a
# native command's stderr into ErrorRecords. Terminating here would surface as
# an exception rather than as the "unverified" exit code that callers rely on.
$ErrorActionPreference = 'Continue'

function Write-Note([string]$text) {
    if (-not $Quiet) { Write-Host $text }
}

$expected = $ExpectedSha.ToLower()

# safe.directory because /app belongs to the unprivileged app user while this
# runs as root; without it git refuses the checkout as "dubious ownership" and
# the read fails for a reason that has nothing to do with the commit.
$raw = & docker run --rm --entrypoint git $Image `
    -c safe.directory=/app -C /app rev-parse HEAD 2>&1
$dockerExit = $LASTEXITCODE

$actual = (($raw | Out-String) -split "`n" |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -match '^[0-9a-f]{40}$' } |
    Select-Object -First 1)

if ($dockerExit -ne 0 -or [string]::IsNullOrWhiteSpace($actual)) {
    # Deliberately NOT treated as a mismatch, and above all not as a pass:
    # the state is "unknown", and a caller must be able to tell it apart.
    Write-Note "[verify] UNVERIFIED -- could not read a commit out of $Image."
    Write-Note "[verify] docker exit $dockerExit; output: $(($raw | Out-String).Trim())"
    exit 2
}

if ($actual -ne $expected) {
    Write-Note "[verify] MISMATCH -- $Image does not contain the requested commit."
    Write-Note "[verify]   asked for: $expected"
    Write-Note "[verify]   image has: $actual"
    exit 1
}

Write-Note "[verify] OK -- $Image contains $actual"
exit 0
