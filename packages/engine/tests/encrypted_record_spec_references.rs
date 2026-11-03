#![cfg(feature = "crypto")]
//! Record framing references, not tests of an implemented container protocol.

use chacha20poly1305::aead::{AeadInPlace, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Nonce, Tag};
use sha2::{Digest, Sha256};

const MAX_PLAINTEXT_LEN: u64 = ((1u64 << 32) - 1) * 64;

fn field(label: &str, value: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    output.extend_from_slice(&u32::try_from(label.len()).unwrap().to_le_bytes());
    output.extend_from_slice(label.as_bytes());
    output.extend_from_slice(&u64::try_from(value.len()).unwrap().to_le_bytes());
    output.extend_from_slice(value);
    output
}

fn structural_metadata_digest(layer: u8, preamble: &[u8], directory: &[u8]) -> [u8; 32] {
    Sha256::digest(
        [
            field("domain", b"DNAM-ENCRYPTED-STRUCTURE-V1"),
            field("layer", &[layer]),
            field("preamble", preamble),
            field("directory", directory),
        ]
        .concat(),
    )
    .into()
}

fn mutable_wrap_digest(context: &[u8; 32], header: &[u8; 20], recipients: &[u8]) -> [u8; 32] {
    Sha256::digest(
        [
            field("domain", b"DNAM-MUTABLE-WRAP-V1"),
            field("record_context_sha256", context),
            field("header", header),
            field("recipient_key_ids", recipients),
        ]
        .concat(),
    )
    .into()
}

fn mutable_content_aad(digest: &[u8; 32], slots: &[u8]) -> Vec<u8> {
    [
        field("domain", b"DNAM-MUTABLE-CONTENT-V1"),
        field("mutable_wrap_sha256", digest),
        field("slots", slots),
    ]
    .concat()
}

fn record_nonce(index: u32) -> Result<[u8; 12], &'static str> {
    if index > 4 {
        return Err("unsupported record index");
    }
    let mut nonce = [0; 12];
    nonce[8..].copy_from_slice(&index.to_le_bytes());
    Ok(nonce)
}

fn record_aad(
    context: &[u8; 32],
    index: u32,
    ciphertext_len: u64,
) -> Result<Vec<u8>, &'static str> {
    if index > 4 {
        return Err("unsupported record index");
    }
    if !(16..=MAX_PLAINTEXT_LEN + 16).contains(&ciphertext_len) {
        return Err("unsupported ciphertext length");
    }
    let kind = u8::from(index != 0);
    let tier = if index == 0 {
        u8::MAX
    } else {
        u8::try_from(index - 1).unwrap()
    };
    Ok([
        field("domain", b"DNAM-RECORD-AAD-V1"),
        field("record_context_sha256", context),
        field("record_index", &index.to_le_bytes()),
        field("record_kind", &[kind]),
        field("tier", &[tier]),
        field("ciphertext_length", &ciphertext_len.to_le_bytes()),
    ]
    .concat())
}

fn seal(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], plaintext: &[u8]) -> Vec<u8> {
    let cipher = ChaCha20Poly1305::new(key.into());
    let mut ciphertext = plaintext.to_vec();
    let tag = cipher
        .encrypt_in_place_detached(Nonce::from_slice(nonce), aad, &mut ciphertext)
        .unwrap();
    ciphertext.extend_from_slice(&tag);
    ciphertext
}

fn open(
    key: &[u8; 32],
    nonce: &[u8; 12],
    aad: &[u8],
    ciphertext: &[u8],
) -> Result<Vec<u8>, chacha20poly1305::aead::Error> {
    let (ciphertext, tag) = ciphertext.split_at(ciphertext.len() - 16);
    let mut plaintext = ciphertext.to_vec();
    ChaCha20Poly1305::new(key.into()).decrypt_in_place_detached(
        Nonce::from_slice(nonce),
        aad,
        &mut plaintext,
        Tag::from_slice(tag),
    )?;
    Ok(plaintext)
}

#[test]
fn encrypted_structure_framing_has_fixed_bytes_and_distinct_layer_binding() {
    let expected = [
        b"\x06\0\0\0domain\x1b\0\0\0\0\0\0\0DNAM-ENCRYPTED-STRUCTURE-V1".as_slice(),
        b"\x05\0\0\0layer\x01\0\0\0\0\0\0\0\0",
        b"\x08\0\0\0preamble\x01\0\0\0\0\0\0\0p",
        b"\x09\0\0\0directory\x02\0\0\0\0\0\0\0dd",
    ]
    .concat();
    let original = structural_metadata_digest(0, b"p", b"dd");
    let expected_digest: [u8; 32] = Sha256::digest(expected).into();
    assert_eq!(original, expected_digest);
    assert_ne!(original, structural_metadata_digest(1, b"p", b"dd"));
    assert_ne!(original, structural_metadata_digest(0, b"q", b"dd"));
    assert_ne!(original, structural_metadata_digest(0, b"p", b"de"));
    assert_ne!(original, structural_metadata_digest(0, b"pd", b"d"));
}

#[test]
fn record_nonce_and_aad_have_fixed_little_endian_bytes() {
    let context = [0x11; 32];
    let expected = [
        b"\x06\0\0\0domain\x12\0\0\0\0\0\0\0DNAM-RECORD-AAD-V1".as_slice(),
        b"\x15\0\0\0record_context_sha256\x20\0\0\0\0\0\0\0",
        &context,
        b"\x0c\0\0\0record_index\x04\0\0\0\0\0\0\0\x01\0\0\0",
        b"\x0b\0\0\0record_kind\x01\0\0\0\0\0\0\0\x01",
        b"\x04\0\0\0tier\x01\0\0\0\0\0\0\0\0",
        b"\x11\0\0\0ciphertext_length\x08\0\0\0\0\0\0\0\x30\0\0\0\0\0\0\0",
    ]
    .concat();
    assert_eq!(record_aad(&context, 1, 48).unwrap(), expected);
    for index in 0..=4 {
        let nonce = record_nonce(index).unwrap();
        assert_eq!(&nonce[..8], &[0; 8]);
        assert_eq!(&nonce[8..], &index.to_le_bytes());
    }
    let shared_aad = record_aad(&context, 0, 16).unwrap();
    assert!(
        shared_aad
            .windows(24)
            .any(|bytes| bytes == b"\x0b\0\0\0record_kind\x01\0\0\0\0\0\0\0\0")
    );
    assert!(
        shared_aad
            .windows(17)
            .any(|bytes| bytes == b"\x04\0\0\0tier\x01\0\0\0\0\0\0\0\xff")
    );
    assert_eq!(record_nonce(5), Err("unsupported record index"));
    assert_eq!(record_aad(&context, 5, 16), Err("unsupported record index"));
    assert_eq!(
        record_aad(&context, 0, 15),
        Err("unsupported ciphertext length")
    );
    assert!(record_aad(&context, 0, MAX_PLAINTEXT_LEN + 16).is_ok());
    assert_eq!(
        record_aad(&context, 0, MAX_PLAINTEXT_LEN + 17),
        Err("unsupported ciphertext length")
    );
}

#[test]
fn aead_binds_context_record_identity_length_and_ciphertext() {
    let key = [0x42; 32];
    let nonce = record_nonce(1).unwrap();
    let context = [0x11; 32];
    let plaintext = b"example tier record";
    let ciphertext_len = u64::try_from(plaintext.len()).unwrap() + 16;
    let aad = record_aad(&context, 1, ciphertext_len).unwrap();
    let ciphertext = seal(&key, &nonce, &aad, plaintext);

    assert_eq!(open(&key, &nonce, &aad, &ciphertext).unwrap(), plaintext);
    // AEAD alone cannot reject an unchanged valid replay.
    assert_eq!(open(&key, &nonce, &aad, &ciphertext).unwrap(), plaintext);
    assert!(
        open(
            &key,
            &nonce,
            &record_aad(&[0x22; 32], 1, ciphertext_len).unwrap(),
            &ciphertext
        )
        .is_err()
    );
    assert!(
        open(
            &key,
            &nonce,
            &record_aad(&context, 2, ciphertext_len).unwrap(),
            &ciphertext
        )
        .is_err()
    );
    assert!(
        open(
            &key,
            &nonce,
            &record_aad(&context, 1, ciphertext_len + 1).unwrap(),
            &ciphertext
        )
        .is_err()
    );
    assert!(open(&key, &record_nonce(2).unwrap(), &aad, &ciphertext).is_err());
    for (label, old, new) in [("record_kind", 1, 0), ("tier", 0, 1)] {
        let encoded = field(label, &[old]);
        let mut changed_aad = aad.clone();
        let position = aad
            .windows(encoded.len())
            .position(|window| window == encoded)
            .unwrap();
        changed_aad[position + encoded.len() - 1] = new;
        assert!(open(&key, &nonce, &changed_aad, &ciphertext).is_err());
    }
    let mut altered = ciphertext.clone();
    altered[0] ^= 1;
    assert!(open(&key, &nonce, &aad, &altered).is_err());
    let mut altered_tag = ciphertext.clone();
    *altered_tag.last_mut().unwrap() ^= 1;
    assert!(open(&key, &nonce, &aad, &altered_tag).is_err());
}

#[test]
fn changing_associated_data_does_not_make_nonce_reuse_safe() {
    let key = [0x42; 32];
    let nonce = record_nonce(1).unwrap();
    let left = b"first secret";
    let right = b"other secret";
    assert_eq!(left.len(), right.len());
    let length = u64::try_from(left.len()).unwrap() + 16;
    let left_ct = seal(
        &key,
        &nonce,
        &record_aad(&[0x11; 32], 1, length).unwrap(),
        left,
    );
    let right_ct = seal(
        &key,
        &nonce,
        &record_aad(&[0x22; 32], 1, length).unwrap(),
        right,
    );

    for ((a, b), (x, y)) in left_ct.iter().zip(&right_ct).zip(left.iter().zip(right)) {
        assert_eq!(a ^ b, x ^ y);
    }
}

#[test]
fn encrypted_mutable_content_binds_issuance_header_recipients_and_slots() {
    let context = [0x11; 32];
    let mut recipients = 1u32.to_le_bytes().to_vec();
    recipients.extend_from_slice(&[0x33; 32]);
    let mut slots = 1u32.to_le_bytes().to_vec();
    slots.extend_from_slice(&[0x44; 80]);
    let mut plaintext = 1u32.to_le_bytes().to_vec();
    plaintext.extend_from_slice(&[0x55; 32]);
    plaintext.extend_from_slice(b"DMU1\x8f\xb4\xcc\xa1\x02\x00\x00\x00\x00\x00\x00\x00{}");
    let mut header = [0u8; 20];
    header[..4].copy_from_slice(b"DME1");
    header[4] = 2;
    header[8..12].copy_from_slice(&1u32.to_le_bytes());
    header[12..20].copy_from_slice(&(u64::try_from(plaintext.len()).unwrap() + 16).to_le_bytes());

    let digest = mutable_wrap_digest(&context, &header, &recipients);
    let aad = mutable_content_aad(&digest, &slots);
    let key = [0x42; 32];
    let nonce = [0; 12];
    let ciphertext = seal(&key, &nonce, &aad, &plaintext);
    assert_eq!(open(&key, &nonce, &aad, &ciphertext).unwrap(), plaintext);

    let other_context = [0x12; 32];
    let mut other_header = header;
    other_header[4] = 1;
    let mut other_recipients = recipients.clone();
    other_recipients[4] ^= 1;
    for changed in [
        mutable_wrap_digest(&other_context, &header, &recipients),
        mutable_wrap_digest(&context, &other_header, &recipients),
        mutable_wrap_digest(&context, &header, &other_recipients),
    ] {
        assert!(
            open(
                &key,
                &nonce,
                &mutable_content_aad(&changed, &slots),
                &ciphertext
            )
            .is_err()
        );
    }
    let mut other_slots = slots.clone();
    other_slots[4] ^= 1;
    assert!(
        open(
            &key,
            &nonce,
            &mutable_content_aad(&digest, &other_slots),
            &ciphertext
        )
        .is_err()
    );
    let mut altered = ciphertext.clone();
    altered[0] ^= 1;
    assert!(open(&key, &nonce, &aad, &altered).is_err());
    assert!(open(&[0x43; 32], &nonce, &aad, &ciphertext).is_err());
}
