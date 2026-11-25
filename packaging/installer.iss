; AI Reel Maker installer (Inno Setup 6). Built by packaging\build.ps1, which passes:
;   /DAppVersion=1.0.0 /DStageDir=<staged program folder> /DOutputDir=<dist folder>
; Per-user install (no administrator prompt) to %LOCALAPPDATA%\Programs\AI Reel Maker.
; Data stays in %LOCALAPPDATA%\AI Reel Maker and survives updates; uninstalling asks about it.

#define AppName "AI Reel Maker"
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef StageDir
  #define StageDir "..\build\stage"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif

[Setup]
AppId={{9546E71F-2919-4D6B-B414-DA011F8C960F}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=salmanulharis
AppPublisherURL=https://github.com/salmanulharis/ai-reel-maker
AppSupportURL=https://github.com/salmanulharis/ai-reel-maker/issues
AppUpdatesURL=https://github.com/salmanulharis/ai-reel-maker/releases
VersionInfoVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename=AIReelMaker-Setup-{#AppVersion}
SetupIconFile=app.ico
UninstallDisplayIcon={app}\app.ico
UninstallDisplayName={#AppName}
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; An update replaces the program files completely, so nothing from an older version is left.
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\tools"
Type: filesandordirs; Name: "{app}\frontend"
Type: filesandordirs; Name: "{app}\remotion"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "app.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\runtime\python\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\app.ico"; Comment: "Turn long videos into short highlights"
Name: "{autoprograms}\Stop {#AppName}"; Filename: "{app}\runtime\python\pythonw.exe"; Parameters: """{app}\launcher.pyw"" --stop"; WorkingDir: "{app}"; IconFilename: "{app}\app.ico"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\runtime\python\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\app.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\runtime\python\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"; Description: "Start {#AppName}"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
; Compiled Python files and anything else written next to the program.
Type: filesandordirs; Name: "{app}"

[Code]
procedure StopRunningApp();
var
  Launcher: String;
  Code: Integer;
begin
  Launcher := ExpandConstant('{app}\launcher.pyw');
  if FileExists(Launcher) then
    Exec(ExpandConstant('{app}\runtime\python\pythonw.exe'), '"' + Launcher + '" --stop --quiet',
      ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Code);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  StopRunningApp();
  Result := '';
end;

function InitializeUninstall(): Boolean;
begin
  StopRunningApp();
  Result := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\{#AppName}');
    if DirExists(DataDir) and not UninstallSilent() then
      if MsgBox('Also delete your projects, finished videos, settings and the prepared video tools?' + #13#10 + #13#10 +
                DataDir + #13#10 + #13#10 +
                'Choose No to keep them for a later install. Downloaded AI models (in your user folder) are kept either way.',
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
