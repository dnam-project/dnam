//! Byte-level mutable-frame examples, not a production container parser.

use dnam_engine::math::crc32c;

#[test]
fn open_mutable_empty_object_has_a_fixed_crc32c_and_wire_frame() {
    let payload = b"{}";
    let length = u64::try_from(payload.len()).unwrap().to_le_bytes();
    let checksum_input = [b"DMU1".as_slice(), &length, payload].concat();
    assert_eq!(crc32c(b"123456789"), 0xe306_9283);
    assert_eq!(crc32c(&checksum_input), 0xa1cc_b48f);

    let frame = [
        b"DMU1".as_slice(),
        &crc32c(&checksum_input).to_le_bytes(),
        &length,
        payload,
    ]
    .concat();
    assert_eq!(frame, b"DMU1\x8f\xb4\xcc\xa1\x02\0\0\0\0\0\0\0{}");
}

#[test]
fn recomputing_mutable_crc_does_not_authenticate_the_content() {
    let first = b"DMU1\x07\0\0\0\0\0\0\0{\"x\":0}";
    let mut frame = [
        b"DMU1".as_slice(),
        &crc32c(first).to_le_bytes(),
        &first[4..],
    ]
    .concat();
    frame[21] = b'1';
    let altered_checksum = crc32c(&[&frame[..4], &frame[8..16], &frame[16..]].concat());
    let stored_checksum = u32::from_le_bytes(frame[4..8].try_into().unwrap());
    assert_ne!(stored_checksum, altered_checksum);
    frame[4..8].copy_from_slice(&altered_checksum.to_le_bytes());
    assert_eq!(
        u32::from_le_bytes(frame[4..8].try_into().unwrap()),
        crc32c(&[&frame[..4], &frame[8..16], &frame[16..]].concat())
    );
    assert_eq!(&frame[16..], br#"{"x":1}"#);
}
