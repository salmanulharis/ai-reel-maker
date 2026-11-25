<#
.SYNOPSIS
  Build the Windows installer: dist\AIReelMaker-Setup-<version>.exe

.DESCRIPTION
  1. Checks: clean git tree (unless -AllowDirty), Python 3.12 on PATH, Inno Setup 6.
  2. Builds the UI and runs the tests (unless -SkipTests).
  3. Stages the program folder in build\stage: embeddable Python with the packages from
     requirements-lock.txt, the app, the built UI, the Remotion source, Node and FFmpeg.
  4. Runs "run.py --self-test" with the staged Python.
  5. Compiles the installer with Inno Setup.
  6. Installs it silently into a temporary folder, runs the self-test there, and uninstalls it
     (skipped when AI Reel Maker is already installed, so a real install is never touched).

  Downloads (embeddable Python, Node) are cached in packaging\.cache and checked against the
  sha256 in packaging\tools.json; an empty sha256 is filled in by the first build.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\build.ps1
#>
param(
    [switch]$AllowDirty,
    [switch]$SkipTests,
    [switch]$SkipInstallTest
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $PSScriptRoot
$Packaging = Join-Path $Root "packaging"
$Cache = Join-Path $Packaging ".cache"
$Build = Join-Path $Root "build"
$Stage = Join-Path $Build "stage"
$Dist = Join-Path $Root "dist"
$ToolsFile = Join-Path $Packaging "tools.json"
$AppId = "{9546E71F-2919-4D6B-B414-DA011F8C960F}"

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "Build failed: $text" -ForegroundColor Red; exit 1 }
function Check-Exit($what) { if ($LASTEXITCODE -ne 0) { Fail "$what (exit code $LASTEXITCODE)" } }

function Get-Sha256($path) { (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower() }

function Set-ToolValue($tool, $key, $value) {
    # Keep tools.json readable (PowerShell 5.1's ConvertTo-Json formats it oddly).
    python -c "import json,sys; p=sys.argv[1]; d=json.load(open(p,encoding='utf-8')); d[sys.argv[2]][sys.argv[3]]=sys.argv[4]; open(p,'w',encoding='utf-8',newline='\n').write(json.dumps(d,indent=2)+'\n')" $ToolsFile $tool $key $value
    Check-Exit "updating tools.json"
    Write-Host "   pinned $tool.$key = $value (commit packaging\tools.json)" -ForegroundColor Yellow
}

function Confirm-Hash($tool, $key, $path, $expected) {
    $actual = Get-Sha256 $path
    if (-not $expected) { Set-ToolValue $tool $key $actual; return }
    if ($actual -ne $expected.ToLower()) { Fail "$path does not match the sha256 in tools.json ($actual)" }
}

function Get-Download($url, $name) {
    $target = Join-Path $Cache $name
    if (-not (Test-Path $target)) {
        Write-Host "   downloading $url"
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile "$target.part"
        Move-Item "$target.part" $target -Force
    }
    return $target
}

function Copy-Tracked($paths, $destination) {
    # Files git knows about (tracked or new, never ignored), so caches and local files stay out.
    $files = git -C $Root ls-files --cached --others --exclude-standard -- $paths
    Check-Exit "git ls-files"
    foreach ($file in $files) {
        $source = Join-Path $Root $file
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
        $target = Join-Path $destination $file
        New-Item -ItemType Directory -Force (Split-Path -Parent $target) | Out-Null
        Copy-Item -LiteralPath $source $target -Force
    }
}

# 1. Checks
Step "Checking the build computer"
New-Item -ItemType Directory -Force $Cache, $Build, $Dist | Out-Null
$tools = Get-Content $ToolsFile -Raw | ConvertFrom-Json
$version = (Select-String -Path (Join-Path $Root "app\version.py") -Pattern '__version__ = "([^"]+)"').Matches[0].Groups[1].Value
if (-not $version) { Fail "no version in app\version.py" }
Write-Host "   version $version"
if (-not $AllowDirty) {
    $dirty = git -C $Root status --porcelain
    if ($dirty) { Fail "the git tree has uncommitted changes (commit them, or pass -AllowDirty for a test build)" }
}
$pyVersion = (python -c "import platform; print(platform.python_version())").Trim()
Check-Exit "python"
if ($pyVersion -notlike "3.12.*") { Fail "Python 3.12 is needed on PATH to install the packages (found $pyVersion)" }
$iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { Fail "Inno Setup 6 was not found (winget install JRSoftware.InnoSetup)" }
$ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
$ffprobe = (Get-Command ffprobe -ErrorAction SilentlyContinue).Source
if (-not $ffmpeg -or -not $ffprobe) { Fail "ffmpeg and ffprobe must be on PATH (winget install Gyan.FFmpeg)" }
$ffVersion = (& $ffmpeg -hide_banner -version | Select-Object -First 1)
if ($ffVersion -notmatch "ffmpeg version $([regex]::Escape($tools.ffmpeg.version))") { Fail "FFmpeg $($tools.ffmpeg.version) is pinned in tools.json, found: $ffVersion" }

# 2. UI and tests
Step "Building the UI"
Push-Location (Join-Path $Root "frontend")
npm ci --no-audit --no-fund; Check-Exit "npm ci"
npm run build; Check-Exit "npm run build"
Pop-Location
if (-not $SkipTests) {
    Step "Running the tests"
    Push-Location $Root
    python -m pytest -q -p no:cacheprovider; Check-Exit "pytest"
    Pop-Location
    Push-Location (Join-Path $Root "frontend")
    npx tsc --noEmit -p .; Check-Exit "frontend type check"
    Pop-Location
}

# 3. Stage the program folder
Step "Staging the program folder"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force $Stage | Out-Null
Copy-Tracked @("app", "run.py", "launcher.pyw", "LICENSE", "remotion/src", "remotion/scripts", "remotion/package.json", "remotion/package-lock.json", "remotion/tsconfig.json") $Stage
Copy-Item -Recurse (Join-Path $Root "frontend\dist") (Join-Path $Stage "frontend\dist")
Copy-Item (Join-Path $Packaging "THIRD-PARTY-NOTICES.txt") $Stage
Set-Content -Path (Join-Path $Stage "VERSION") -Value $version -Encoding ascii
Set-Content -Path (Join-Path $Stage "INSTALLED") -Value "Installed with Setup.exe: data lives in %LOCALAPPDATA%\AI Reel Maker (see app/paths.py)." -Encoding ascii

Write-Host "   Python $($tools.python.version) (embeddable)"
$pyZip = Get-Download $tools.python.url "python-$($tools.python.version)-embed-amd64.zip"
Confirm-Hash "python" "sha256" $pyZip $tools.python.sha256
$pyDir = Join-Path $Stage "runtime\python"
Expand-Archive -LiteralPath $pyZip -DestinationPath $pyDir -Force
$pth = Get-ChildItem $pyDir -Filter "python*._pth" | Select-Object -First 1
# Packages, and the program folder two levels up (for "import app"); "import site" reads .pth files.
$pthLines = @((Get-Content $pth.FullName | Where-Object { $_ -match '\.zip$' -or $_ -eq '.' }) + @('Lib\site-packages', '..\..', 'import site'))
[IO.File]::WriteAllLines($pth.FullName, [string[]]$pthLines)

Write-Host "   packages from requirements-lock.txt"
$sitePackages = Join-Path $pyDir "Lib\site-packages"
python -m pip install --disable-pip-version-check --no-warn-script-location --no-deps --only-binary=:all: --target $sitePackages -r (Join-Path $Root "requirements-lock.txt")
Check-Exit "pip install"
# Remove what the app never uses: launchers made for the build computer's Python, test suites.
$binDir = Join-Path $sitePackages "bin"
if (Test-Path $binDir) { Remove-Item -Recurse -Force $binDir }
foreach ($package in @("numpy", "scipy", "sklearn", "pandas", "av")) {
    $packageDir = Join-Path $sitePackages $package
    if (Test-Path $packageDir) {
        Get-ChildItem $packageDir -Recurse -Directory -Filter "tests" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "   Node $($tools.node.version)"
$nodeZipName = "node-v$($tools.node.version)-win-x64.zip"
$nodeZip = Get-Download $tools.node.url $nodeZipName
$sums = Get-Download $tools.node.shasums "node-v$($tools.node.version)-SHASUMS256.txt"
$official = ((Get-Content $sums | Where-Object { $_ -match "  $([regex]::Escape($nodeZipName))$" }) -split "\s+")[0]
if (-not $official) { Fail "no checksum for $nodeZipName on nodejs.org" }
if ((Get-Sha256 $nodeZip) -ne $official.ToLower()) { Fail "$nodeZipName does not match nodejs.org's checksum" }
Confirm-Hash "node" "sha256" $nodeZip $tools.node.sha256
$nodeTemp = Join-Path $Build "node-unpacked"
if (Test-Path $nodeTemp) { Remove-Item -Recurse -Force $nodeTemp }
Expand-Archive -LiteralPath $nodeZip -DestinationPath $nodeTemp -Force
New-Item -ItemType Directory -Force (Join-Path $Stage "tools") | Out-Null
Move-Item (Join-Path $nodeTemp "node-v$($tools.node.version)-win-x64") (Join-Path $Stage "tools\node")
Remove-Item -Recurse -Force $nodeTemp

Write-Host "   FFmpeg $($tools.ffmpeg.version) from $(Split-Path -Parent $ffmpeg)"
$ffDir = Join-Path $Stage "tools\ffmpeg"
New-Item -ItemType Directory -Force $ffDir | Out-Null
Confirm-Hash "ffmpeg" "ffmpeg_sha256" $ffmpeg $tools.ffmpeg.ffmpeg_sha256
Confirm-Hash "ffmpeg" "ffprobe_sha256" $ffprobe $tools.ffmpeg.ffprobe_sha256
Copy-Item $ffmpeg, $ffprobe $ffDir
$ffLicense = Join-Path (Split-Path -Parent (Split-Path -Parent $ffmpeg)) "LICENSE"
if (Test-Path $ffLicense) { Copy-Item $ffLicense (Join-Path $ffDir "LICENSE.txt") }

Write-Host "   compiling Python files"
$stagedPython = Join-Path $pyDir "python.exe"
& $stagedPython -m compileall -q -j 0 (Join-Path $Stage "app") $sitePackages | Out-Null
Get-ChildItem $Stage -Recurse -Directory -Filter "__pycache__" | Where-Object { $_.FullName -like "*\remotion\*" } | Remove-Item -Recurse -Force

# 4. Self-test of the staged folder (its data goes to a throwaway folder)
Step "Self-test of the staged program"
$testHome = Join-Path $Build "selftest-home"
if (Test-Path $testHome) { Remove-Item -Recurse -Force $testHome }
$env:AIREEL_HOME = $testHome
& $stagedPython (Join-Path $Stage "run.py") --self-test
$selfTest = $LASTEXITCODE
Remove-Item Env:\AIREEL_HOME
if ($selfTest -ne 0) { Fail "the staged program's self-test failed" }
$stageMb = [math]::Round(((Get-ChildItem $Stage -Recurse -File | Measure-Object Length -Sum).Sum / 1MB))
Write-Host "   staged program: $stageMb MB"

# 5. Installer
Step "Compiling the installer"
& $iscc "/Q" "/DAppVersion=$version" "/DStageDir=$Stage" "/DOutputDir=$Dist" (Join-Path $Packaging "installer.iss")
Check-Exit "Inno Setup"
$setup = Join-Path $Dist "AIReelMaker-Setup-$version.exe"
if (-not (Test-Path $setup)) { Fail "the installer was not made" }
Write-Host "   $setup ($([math]::Round((Get-Item $setup).Length / 1MB)) MB)"

# 6. Real install test
if (-not $SkipInstallTest) {
    Step "Install test"
    $existing = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$($AppId)_is1"
    if (Test-Path $existing) {
        Write-Host "   skipped: AI Reel Maker is installed on this computer, and the test must not replace it." -ForegroundColor Yellow
    }
    else {
        $testDir = Join-Path $Build "install-test"
        if (Test-Path $testDir) { Remove-Item -Recurse -Force $testDir }
        $process = Start-Process -FilePath $setup -ArgumentList @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/NOICONS", "/CURRENTUSER", "/DIR=`"$testDir`"", "/LOG=`"$Build\install-test.log`"") -Wait -PassThru
        if ($process.ExitCode -ne 0) { Fail "silent install failed (exit code $($process.ExitCode), see build\install-test.log)" }
        $env:AIREEL_HOME = $testHome
        & (Join-Path $testDir "runtime\python\python.exe") (Join-Path $testDir "run.py") --self-test
        $installed = $LASTEXITCODE
        Remove-Item Env:\AIREEL_HOME
        $uninstaller = Join-Path $testDir "unins000.exe"
        if (Test-Path $uninstaller) {
            Start-Process -FilePath $uninstaller -ArgumentList @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART") -Wait | Out-Null
        }
        if ($installed -ne 0) { Fail "the installed program's self-test failed" }
        Write-Host "   installed, self-tested and uninstalled"
    }
}

Step "Done"
Write-Host "   $setup"
Write-Host "   sha256 $(Get-Sha256 $setup)"
