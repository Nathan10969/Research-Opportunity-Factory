# DINOv3 authorized full-download design

Status: approved by the user's explicit instruction to download every unique
file in the supplied official Meta URL list.

## Scope

Download all 17 unique official files represented by the 42 repeated URL
occurrences in the supplied attachment: 12 pretrained backbones and 5 official
task heads. Do not silently reduce the set to the models used by the PNNL pilot.

## Storage and secrecy contract

- Root: `D:/rookie/rookie/model_weights/dinov3_official_full_20260817`.
- Store backbones and heads in separate subdirectories.
- Read URLs only from the user attachment at runtime.
- Never write full URLs to Git, logs, manifests, process summaries, or evidence.
- Accept only HTTPS URLs on `dinov3.llamameta.net` with a `.pth` basename.

## Transfer and validation contract

- Deduplicate exact URLs before scheduling.
- Use at most two concurrent downloads.
- Resume an existing `.part` file with an HTTP Range request.
- If a server ignores Range, restart that one file rather than append corruption.
- Move `.part` to the final filename only after the response completes.
- Compute full SHA-256 and verify that it begins with the eight-hex suffix in
  the official filename.
- Persist a redacted progress file and final checksum manifest containing no URL.
- Keep at least 200 GiB free on D; stop scheduling new files below that floor.

## Failure behavior

HTTP failures retain the `.part` file for later resume. A failed file remains
`partial` or `failed`; it is never counted as downloaded. Completion requires
17 final files, 17 prefix checks, 17 full hashes, and zero missing files.
