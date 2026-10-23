//! Castagnoli `CRC32C` (`0x1EDC6F41` / reflected `0x82F63B78`) lookup table and software fallback.

/// Slicing-by-8 lookup tables (`8 × 256 × u32 = 8 KB`) for Castagnoli `CRC32C`
/// (reflected polynomial `0x82F63B78`).
const CRC32C_TABLES: [[u32; 256]; 8] = {
    let mut tables = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0x82F6_3B78;
            } else {
                crc >>= 1;
            }
            bit += 1;
        }
        tables[0][i] = crc;
        i += 1;
    }
    let mut t = 1;
    while t < 8 {
        let mut j = 0;
        while j < 256 {
            let prev = tables[t - 1][j];
            tables[t][j] = (prev >> 8) ^ tables[0][(prev & 0xFF) as usize];
            j += 1;
        }
        t += 1;
    }
    tables
};

/// Updates a running Castagnoli `CRC32C` checksum over `data` using Slicing-by-8.
#[inline]
pub fn crc32c_slice(crc_init: u32, data: &[u8]) -> u32 {
    let mut crc = !crc_init;
    let (chunks, remainder) = data.as_chunks::<8>();
    for chunk in chunks {
        let c0 = crc ^ u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let c1 = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        crc = CRC32C_TABLES[7][(c0 & 0xFF) as usize]
            ^ CRC32C_TABLES[6][((c0 >> 8) & 0xFF) as usize]
            ^ CRC32C_TABLES[5][((c0 >> 16) & 0xFF) as usize]
            ^ CRC32C_TABLES[4][(c0 >> 24) as usize]
            ^ CRC32C_TABLES[3][(c1 & 0xFF) as usize]
            ^ CRC32C_TABLES[2][((c1 >> 8) & 0xFF) as usize]
            ^ CRC32C_TABLES[1][((c1 >> 16) & 0xFF) as usize]
            ^ CRC32C_TABLES[0][(c1 >> 24) as usize];
    }
    for &byte in remainder {
        let idx = ((crc ^ (byte as u32)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32C_TABLES[0][idx];
    }
    !crc
}

/// Computes Castagnoli `CRC32C` over `data` starting from `0`.
#[inline]
pub fn crc32c(data: &[u8]) -> u32 {
    crc32c_slice(0, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DefaultMathBackend, MathBackend};
    use proptest::prelude::*;

    #[test]
    fn test_crc32c_known_vector() {
        // Standard RFC 3720 / iSCSI CRC32C check value for "123456789" is 0xE3069283
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_hardware_crc32c_matches_software_table_and_incremental(
            payload in proptest::collection::vec(any::<u8>(), 0..512),
            split_ratio in 0.0_f64..1.0_f64,
        ) {
            let sw = crc32c(&payload);
            let hw = DefaultMathBackend::crc32c(&payload);
            prop_assert_eq!(hw, sw);

            let split_idx = ((payload.len() as f64) * split_ratio) as usize;
            let (left, right) = payload.split_at(split_idx.min(payload.len()));
            let inc = crc32c_slice(crc32c_slice(0, left), right);
            prop_assert_eq!(inc, sw);
        }
    }
}
