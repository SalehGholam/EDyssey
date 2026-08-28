; Shared Inno Setup config for both EDyssey installer variants - included by
; EDyssey_online.iss (small, downloads model weights on first use) and
; EDyssey_offline.iss (bundles them, no internet needed after install).
; Not meant to be compiled directly - `#define Variant` (and, for a variant
; that builds from a different PyInstaller output folder, `#define DistDir`
; too) then `#include` this from one of those two files instead.

#define AppName "EDyssey"
#define AppVersion "1.0.0"
#define AppPublisher "SalehG"
#define AppURL "https://github.com/SalehGholam/EDyssey"
; Only set the default if the includer didn't already #define one -
; EDyssey_offline.iss builds from a separate PyInstaller output
; (dist_offline/, via EDYSSEY_OFFLINE_BUILD=1) and overrides this.
#ifndef DistDir
  #define DistDir "..\dist\EDyssey"
#endif

[Setup]
AppId={{B8B1B6DA-4B7E-4C8B-9F52-EDY55EE00001}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
; Lets a non-admin install per-user (into {localappdata}\Programs) instead
; of requiring elevation - relevant because the documented post-install
; "pip install --target ...\_internal torch" step (see INSTALL.md) is much
; simpler without an elevated shell for a Program Files install.
PrivilegesRequiredOverridesAllowed=dialog
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
