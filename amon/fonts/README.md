# Glyph-matching fonts

TrueType / OpenType fonts placed here are used by the HUD text recognizer
(`amon.textocr`) to build letter templates.

## Bundled default

`VCR_OSD_MONO_1.001.ttf` — **VCR OSD Mono** by Riciery Leal (MrManet),
version 1.001 (2015). Distributed as **100% free** on [DaFont](https://www.dafont.com/vcr-osd-mono.font)
(personal and commercial use). Dropped in as the default OSD-style match
for VHS / CRT-style overlays.

## Custom fonts

1. Copy your `.ttf` / `.otf` into this folder (or keep it elsewhere).
2. Point the HUD detector at it in the YAML config:

```yaml
- class: amon.detectors.hud.HudDetector
  config:
    glyph_font: MyFont.ttf          # resolved under amon/fonts/
    # glyph_font: /abs/path/Font.ttf
```

Relative paths are tried against this folder first, then the process
working directory.
