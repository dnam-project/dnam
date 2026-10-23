//! RFC 8439 `ChaCha20` 64-Byte Cache-Line Arena Stream Encryption.

use zeroize::Zeroizing;

use crate::schema::DnamError;

#[inline(always)]
const fn chacha20_quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(16);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(12);

    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(8);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(7);
}

/// Generates a single 64-byte RFC 8439 `ChaCha20` keystream block for `(cek, nonce_96, block_counter)`.
#[inline]
pub const fn chacha20_block_64b(
    cek: &[u8; 32],
    nonce_96: &[u8; 12],
    block_counter: u32,
) -> [u8; 64] {
    let mut initial = [0u32; 16];
    // "expand 32-byte k"
    initial[0] = 0x6170_7865;
    initial[1] = 0x3320_646e;
    initial[2] = 0x7962_2d32;
    initial[3] = 0x6b20_6574;

    let mut i = 0;
    while i < 8 {
        let j = i * 4;
        initial[4 + i] = u32::from_le_bytes([cek[j], cek[j + 1], cek[j + 2], cek[j + 3]]);
        i += 1;
    }
    initial[12] = block_counter;
    initial[13] = u32::from_le_bytes([nonce_96[0], nonce_96[1], nonce_96[2], nonce_96[3]]);
    initial[14] = u32::from_le_bytes([nonce_96[4], nonce_96[5], nonce_96[6], nonce_96[7]]);
    initial[15] = u32::from_le_bytes([nonce_96[8], nonce_96[9], nonce_96[10], nonce_96[11]]);

    let mut working = initial;
    let mut r = 0;
    while r < 10 {
        // Column rounds
        chacha20_quarter_round(&mut working, 0, 4, 8, 12);
        chacha20_quarter_round(&mut working, 1, 5, 9, 13);
        chacha20_quarter_round(&mut working, 2, 6, 10, 14);
        chacha20_quarter_round(&mut working, 3, 7, 11, 15);
        // Diagonal rounds
        chacha20_quarter_round(&mut working, 0, 5, 10, 15);
        chacha20_quarter_round(&mut working, 1, 6, 11, 12);
        chacha20_quarter_round(&mut working, 2, 7, 8, 13);
        chacha20_quarter_round(&mut working, 3, 4, 9, 14);
        r += 1;
    }

    let mut out = [0u8; 64];
    let mut k = 0;
    while k < 16 {
        let word = working[k].wrapping_add(initial[k]).to_le_bytes();
        let base = k * 4;
        out[base] = word[0];
        out[base + 1] = word[1];
        out[base + 2] = word[2];
        out[base + 3] = word[3];
        k += 1;
    }
    out
}

/// Encrypts or decrypts a 64-byte aligned tensor arena prefix in-place using RFC 8439 `ChaCha20`
/// (`Lite=168`, `Full=349`, `Ultra=566` blocks, `0` bytes padding overhead).
#[inline]
pub fn chacha20_xor_arena_blocks(
    arena: &mut [u8],
    cek: &[u8; 32],
    nonce_96: &[u8; 12],
    start_block: u32,
) -> Result<(), DnamError> {
    if (arena.len() & 0x3F) != 0 {
        return Err(DnamError::UnalignedOffset {
            field: "chacha20_xor_arena_blocks.len",
            offset: arena.len() as u32,
        });
    }
    for (j, chunk) in arena.as_chunks_mut::<64>().0.iter_mut().enumerate() {
        let ks = Zeroizing::new(chacha20_block_64b(
            cek,
            nonce_96,
            start_block.wrapping_add(j as u32),
        ));
        for i in 0..64 {
            chunk[i] ^= ks[i];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_chacha20_unaligned_error() {
        let mut bad = [0u8; 63];
        assert_eq!(
            chacha20_xor_arena_blocks(&mut bad, &[0u8; 32], &[0u8; 12], 0),
            Err(DnamError::UnalignedOffset {
                field: "chacha20_xor_arena_blocks.len",
                offset: 63,
            })
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_chacha20_xor_involution(
            cek in prop::array::uniform32(any::<u8>()),
            nonce in prop::array::uniform12(any::<u8>()),
            start_block in 0u32..1000u32,
            blocks in prop::collection::vec(prop::collection::vec(any::<u8>(), 64..=64), 1..=8),
        ) {
            let mut plaintext = alloc::vec::Vec::with_capacity(blocks.len() * 64);
            for b in &blocks {
                plaintext.extend_from_slice(b);
            }
            let mut buf = plaintext.clone();
            chacha20_xor_arena_blocks(&mut buf, &cek, &nonce, start_block).unwrap();
            prop_assert_ne!(&buf, &plaintext);
            chacha20_xor_arena_blocks(&mut buf, &cek, &nonce, start_block).unwrap();
            prop_assert_eq!(&buf, &plaintext);
        }
    }
}
