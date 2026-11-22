use super::*;

#[test]
fn every_page_boundary_and_the_last_index_fit_the_directory() {
  let mut next = 0;
  for page in 0..PAGES {
    let count = 1 << (FIRST_PAGE_BITS + page.saturating_sub(1) as u32);
    for local in [next, next + count - 1] {
      for shard in [0, SHARDS as u32 - 1] {
        let key = Key(local << SHARD_BITS | shard);
        assert_eq!(key.location(), (page, (local - next) as usize));
        assert!(core::ptr::eq(key.shard(), &DIRECTORY[shard as usize]));
      }
    }
    next += count;
  }
  assert_eq!(next, PER_SHARD);
  assert_eq!(next as usize * SHARDS, 1 << INDEX_BITS);
  assert_eq!(size_of::<Slot>(), 32);
  assert_eq!(size_of::<[Shard; SHARDS]>(), 4096);
  #[cfg(all(feature = "std", unix))]
  assert_eq!(size_of::<Cache>(), 272);
}
