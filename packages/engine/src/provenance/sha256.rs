//! Pure `#![no_std]` FIPS 180-4 SHA-256 & Hexadecimal Utilities.

use alloc::string::String;

use crate::schema::DnamError;

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[inline(always)]
const fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    let mut i = 0;
    while i < 16 {
        let j = i * 4;
        w[i] = u32::from_be_bytes([block[j], block[j + 1], block[j + 2], block[j + 3]]);
        i += 1;
    }
    while i < 64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
        i += 1;
    }

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    let mut t = 0;
    while t < 64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[t])
            .wrapping_add(w[t]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
        t += 1;
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// Computes the FIPS 180-4 SHA-256 digest of `data` in `#![no_std]`.
pub fn sha256_digest(data: &[u8]) -> [u8; 32] {
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let (chunks, rem) = data.as_chunks::<64>();
    for blk in chunks {
        sha256_compress(&mut state, blk);
    }
    let mut pad = [0u8; 128];
    pad[..rem.len()].copy_from_slice(rem);
    pad[rem.len()] = 0x80;
    let bit_len = (data.len() as u64) * 8;
    if rem.len() < 56 {
        pad[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let mut blk = [0u8; 64];
        blk.copy_from_slice(&pad[0..64]);
        sha256_compress(&mut state, &blk);
    } else {
        pad[120..128].copy_from_slice(&bit_len.to_be_bytes());
        let mut blk0 = [0u8; 64];
        let mut blk1 = [0u8; 64];
        blk0.copy_from_slice(&pad[0..64]);
        blk1.copy_from_slice(&pad[64..128]);
        sha256_compress(&mut state, &blk0);
        sha256_compress(&mut state, &blk1);
    }

    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 8 {
        let b = state[i].to_be_bytes();
        out[i * 4] = b[0];
        out[i * 4 + 1] = b[1];
        out[i * 4 + 2] = b[2];
        out[i * 4 + 3] = b[3];
        i += 1;
    }
    out
}

/// Encodes a byte slice as lowercase hexadecimal.
pub fn encode_hex(bytes: &[u8]) -> String {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX_CHARS[(b >> 4) as usize] as char);
        out.push(HEX_CHARS[(b & 0x0F) as usize] as char);
    }
    out
}

/// Decodes a fixed-size hexadecimal string into `[u8; N]`.
pub fn decode_hex_array<const N: usize>(hex: &str) -> Result<[u8; N], DnamError> {
    if hex.len() != N * 2 {
        return Err(DnamError::CryptoVerificationFailed(
            "invalid hex string length",
        ));
    }
    let bytes = hex.as_bytes();
    let mut out = [0u8; N];
    let mut i = 0;
    while i < N {
        let hi = hex_nibble(bytes[i * 2])?;
        let lo = hex_nibble(bytes[i * 2 + 1])?;
        out[i] = (hi << 4) | lo;
        i += 1;
    }
    Ok(out)
}

#[inline(always)]
const fn hex_nibble(b: u8) -> Result<u8, DnamError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(DnamError::CryptoVerificationFailed(
            "invalid hex character in cryptographic field",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_sha256_nist_vectors_and_hex_errors() {
        // Empty string SHA-256
        let empty_sha = encode_hex(&sha256_digest(b""));
        assert_eq!(
            empty_sha,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        // 60-byte payload (> 55 byte padding boundary)
        let sha60 = sha256_digest(&[0x61u8; 60]);
        assert_eq!(decode_hex_array::<32>(&encode_hex(&sha60)).unwrap(), sha60);
        // Uppercase hex decoding and error paths
        assert_eq!(
            decode_hex_array::<4>("DEADBEEF").unwrap(),
            [0xDE, 0xAD, 0xBE, 0xEF]
        );
        assert!(decode_hex_array::<4>("deadbe").is_err());
        assert!(decode_hex_array::<4>("deadbezz").is_err());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_sha256_and_hex_roundtrip(
            data in prop::collection::vec(any::<u8>(), 0..512),
        ) {
            let digest = sha256_digest(&data);
            let hex = encode_hex(&digest);
            let decoded = decode_hex_array::<32>(&hex).unwrap();
            prop_assert_eq!(digest, decoded);
        }
    }
}
