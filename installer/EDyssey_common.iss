; Shared Inno Setup config for both EDyssey installer variants - included by
; EDyssey_online.iss (small, downloads model weights on first use) and
; EDyssey_offline.iss (bundles them, no internet needed after install).
; Not meant to be compiled directly - `#define Variant` (and, for a variant
; that builds from a different PyInstaller output folder, `#define DistDir`
; too) then `#include` this from one of those two files instead.

#define AppName "EDyssey"
#define AppVersion "2.1.20260909.1300"
#define AppPublisher "Saleh Gholam"
#define AppURL "https://github.com/SalehGholam/EDyssey"
; Single-braced form for use everywhere except the [Setup] AppId= directive
; itself below, which - per Inno Setup's own constant-escaping rules for
; that directive's value - needs a DOUBLED leading brace ({{#AppGuid) to
; produce this same single-braced value; see the comment there.
#define AppGuid "{B8B1B6DA-4B7E-4C8B-9F52-EDY55EE00001}"
; Only set the default if the includer didn't already #define one -
; EDyssey_offline.iss builds from a separate PyInstaller output
; (dist_offline/, via EDYSSEY_OFFLINE_BUILD=1) and overrides this.
#ifndef DistDir
  #define DistDir "..\dist\EDyssey"
#endif

[Setup]
; {{#AppGuid rather than {#AppGuid: this directive's value parser treats a
; leading "{" as the start of a Inno constant reference (like {app}) unless
; doubled - see AppGuid's own comment above.
AppId={{#AppGuid}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
; VersionInfo* stamps setup.exe's/the uninstaller's OWN Explorer Properties
; > Details tab - separate from AppVersion above (which only drives the
; wizard's own UI text and the Apps & Features "Version" column, both
; already correct). Left unset, Inno Setup defaults this to 0.0.0.0 - a
; Win32 version resource caps each of the 4 numeric fields at 65535, which
; AppVersion's YYYYMMDD segment doesn't fit, so this re-encodes the same
; build timestamp as MMDD.HHMM instead (drops the year - purely decorative
; file-properties metadata, not used anywhere the app itself checks its own
; version). The major.minor prefix is taken from AppVersion itself rather
; than hardcoded, so bumping AppVersion (e.g. 2.1 -> 2.2) can't leave this
; silently stamping the old one.
#define VersionInfoVersionValue Copy(AppVersion, 1, 4) + Copy(AppVersion, 9, 4) + "." + Copy(AppVersion, 14, 4)
VersionInfoVersion={#VersionInfoVersionValue}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} Setup
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#VersionInfoVersionValue}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
; Lets a non-admin install per-user (into {localappdata}\Programs) instead
; of requiring elevation - relevant because the documented post-install
; "pip install --target ...\_internal torch" step (see INSTALL.md) is much
; simpler without an elevated shell for a Program Files install. `commandline`
; (in addition to `dialog`) lets a silent/unattended install (/VERYSILENT)
; pick the per-user path too via /CURRENTUSER - without it, silent installs
; fall back to Inno Setup's own default (PrivilegesRequired=admin) and hang
; forever waiting on a UAC consent prompt nothing can click in a
; non-interactive session (confirmed the hard way testing this installer).
PrivilegesRequiredOverridesAllowed=dialog commandline
OutputDir=Output
OutputBaseFilename=EDyssey_Setup_{#Variant}_{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
LicenseFile=..\LICENSE
InfoAfterFile=..\THIRD_PARTY_NOTICES.md
; PyQt5/hyperspy GPLv3 dependencies mean the app itself ships under
; GPL-3.0 too (see LICENSE) - Inno Setup's own generated installer
; executable is a separate, unrelated program.
UninstallDisplayIcon={app}\EDyssey.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
; Runtime-writable paths (logs/, temp/, EDyssey/io_utils/temp/, on-demand-
; downloaded model weights) are deliberately NOT listed here - the app
; creates them itself on first use, and Inno Setup's uninstaller only
; removes files IT installed, so anything the app wrote (logs a user may
; want to keep, downloaded models they don't want to re-fetch) survives an
; uninstall with zero extra config.

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\EDyssey.exe"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\EDyssey.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
Filename: "{app}\EDyssey.exe"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[Code]
// Same AppId across versions means Inno Setup would otherwise just install
// straight over whatever's already there - files a newer version removed
// (e.g. worker_*.py's move into EDyssey/workers/ this same release) would
// linger as stale leftovers instead of getting cleaned up, and the
// destination-folder picker gets skipped entirely (Inno Setup's own
// upgrade-detection). Asking to uninstall first avoids both.
function InitializeSetup(): Boolean;
var
  UninstallString: String;
  ResultCode: Integer;
begin
  Result := True;
  if RegQueryStringValue(HKA, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{#AppGuid}_is1',
       'UninstallString', UninstallString) then
  begin
    if MsgBox('A previous installation of {#AppName} was found. It needs to be removed ' +
         'before installing this version - uninstall it now?', mbConfirmation, MB_YESNO) = IDYES then
    begin
      UninstallString := RemoveQuotes(UninstallString);
      if not Exec(UninstallString, '/VERYSILENT /NORESTART /SUPPRESSMSGBOXES', '',
           SW_SHOW, ewWaitUntilTerminated, ResultCode) then
      begin
        MsgBox('Could not run the previous version''s uninstaller automatically - ' +
             'please uninstall {#AppName} manually via Settings > Apps, then run this ' +
             'installer again.', mbError, MB_OK);
        Result := False;
      end;
    end
    else
      Result := False;
  end;
end;
