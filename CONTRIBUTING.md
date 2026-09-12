# Contributing

1. Create a feature branch.
2. Keep the two core modes focused: person-shot selection and black-section removal.
3. Do not add re-encoding or creative video editing to the default export path.
4. Add or update tests for timeline/segment behavior.
5. Run `pytest` before opening a pull request.

Bug reports should include:

- OS and Python version
- FFmpeg version
- input codec/container information (`ffprobe` output if useful)
- the relevant `config.toml` values
- `manifest.json` when the issue concerns detection boundaries

Do not attach private reference photos or private video footage to public issues unless you have explicitly chosen to publish them.
