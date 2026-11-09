//! Canonical v2.0 `.dnam` (`b"DNAM"`) Hybrid Binary Container Wire Framing,
//! RFC 8785 JCS Header Projection, `DMU1`/`DMUE` Mutable Suffix, `CP1` Control Projections,
//! Structured Conditional Selectors, and Delay History Declarations.
//!
//! Implements:
//! - `ref[container.dnam.formats]`, `ref[container.dnam.wire-framing]`,
//!   `ref[container.dnam.preamble-directory]`, `ref[container.dnam.open-layout]`
//! - `ref[container.dnam.mutable-frame]`, `ref[container.dnam.mutable-json]`
//! - `ref[container.dnam.tier-prefix]`, `ref[container.dnam.delay-histories]`,
//!   `ref[container.dnam.load-validation]`, `ref[container.dnam.stage3-slices]`,
//!   `ref[container.dnam.cp1-profile]`, `ref[container.dnam.conditional-rules]`
//! - `ref[container.dnam.metadata-ui-hints]`, `ref[container.dnam.sample-rate-anchor]`,
//!   `ref[container.dnam.jcs-header-schema]`, `ref[container.dnam.security-crossref]`

pub mod conditional;
pub mod jcs;
pub mod layout;
pub mod slices_and_cp1;

pub use conditional::*;
pub use jcs::*;
pub use layout::*;
pub use slices_and_cp1::*;
