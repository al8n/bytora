use super::*;

// Growth doubles the capacity but stops at the length limit, so a length that
// fits never overflows. A large element brings the limit within reach.
#[test]
fn growth_stops_at_the_length_limit() {
  type Large = [u8; 1 << 20];
  let max = max_len::<Large>();
  let capacity = max / 2 + 1;
  assert!(capacity.saturating_mul(2) > max);
  assert_eq!(grown_capacity::<Large>(capacity, capacity + 1), max);
  assert_eq!(grown_capacity::<u8>(16, 17), 32);
  assert_eq!(grown_capacity::<u8>(16, 40), 40);
  assert_eq!(grown_capacity::<u8>(0, 1), 1);
  // An allocation grows from Vec's smallest non-zero capacity, as Vec does.
  assert_eq!(Allocation::<u8>::empty().grown_capacity(0, 1), 8);
  assert_eq!(Allocation::<u8>::empty().grown_capacity(0, 20), 20);
  assert_eq!(Allocation::<Large>::empty().grown_capacity(0, 1), 1);
}
