---
id: ADR-0009
title: "RFC 8785 JCS Header Digest, Prefix-Composable Signatures, RFC 9180 HPKE & Rank-r Preview Containment"
status: accepted
supersedes: []
superseded_by: null
specs:
  - docs/specs/containers/dnam_binary_format.md
  - docs/specs/security/provenance_and_enclave_drm.md
  - docs/specs/security/watermarking_and_marketplace.md
requirements:
  - REQ-5.6
  - REQ-8.1
  - REQ-8.2
  - REQ-8.3
  - REQ-8.4
  - REQ-8.5
  - REQ-8.6
  - REQ-8.7
  - REQ-8.8
---

# ADR-0009: RFC 8785 JCS Header Digest, Prefix-Composable Signatures, RFC 9180 HPKE & Rank-`r` Preview Containment

This record justifies the container framing, cryptographic provenance, HPKE DRM, and storefront preview containment in [`../specs/containers/dnam_binary_format.md`](../specs/containers/dnam_binary_format.md), [`../specs/security/provenance_and_enclave_drm.md`](../specs/security/provenance_and_enclave_drm.md), and [`../specs/security/watermarking_and_marketplace.md`](../specs/security/watermarking_and_marketplace.md) fulfilling [`REQ-5.6`, `REQ-8.1`–`REQ-8.8`](../requirements/system_goals_and_requirements.md).

## 1. Decision & Counterexample Matrix

| Mechanism | Canonical Selection | Counterexample to Naive Formulation |
| :--- | :--- | :--- |
| **Growable Presets & Container Framing (`REQ-8.1`, `ref[container.dnam.mutable-frame]`)** | Open containers place a growable `DMU1` mutable JSON frame and `DMUE` footer after the retained tier boundary; protected containers use an HPKE-wrapped `DME1` frame. Partial presets restore signed defaults for omitted controls before atomic clamping. | Fixed 4 KB JSON header slack requires re-signing when user presets outgrow the initial page. |
| **Authenticated `2.0` Containers & RFC 9180 HPKE (`ref[sec.jcs.immutable-header-digest]`, `ref[sec.sig.prefix-delegation]`, `ref[sec.drm.hpke-record-encryption]`)** | 64B preamble + 128B directory, RFC 8785 JCS immutable-header projection, length-framed `F`/`L` `DAU1`/`DAI1`/`DSA1` authentication envelopes, and RFC 9180 HPKE recipient slots (rejecting non-contributory low-order X25519 keys). | Custom non-HPKE X25519 wrapping allowed low-order public keys to leak the content key; delimiter-joined signing strings collided when vendor/model IDs contained colons. |
| **Offline Protected Authorization (`REQ-8.4`, `REQ-8.6`, `ref[sec.sig.offline-auth]`)** | Issued protected models remain usable offline indefinitely while local keys and authorization are intact; revocation applies only to future licensing and requires independent device-source evidence beyond a buyer watermark or transferred file. | Periodic online lease renewal or automatic hardware revocation from a leaked watermarked file alone breaks live stage reliability. |
| **Storefront Previews (`REQ-8.7`, `ref[sec.preview.containment]`)** | `.dnam-prev` physically omits `Full/Pro/Ultra` arenas, preserves natural musical playing, exposes a bounded responsive window on single-control models (`P = 1`), and enforces fixed-anchor rank-`r` (`r < D_cont`) containment across all published previews for multi-control models (`P ≥ 2`). | Claiming unconditional replay/cloning impossibility or applying continuous rank deficiency to 0-D / 1-D control spaces. |
