use crate::hydration::fallback_cache::FallbackCache;
use crate::models::{AuthorId, PureCore, TweetId};
use xai_core_entities::entities::PureCoreData;

pub(crate) type PureCoreFallbackCache = FallbackCache<u64, PureCore>;

pub(crate) fn pure_core_fallback_cache(capacity: usize) -> PureCoreFallbackCache {
    FallbackCache::new("author_id", capacity)
}

pub(crate) fn pure_core(core: &PureCoreData) -> PureCore {
    PureCore {
        author_id: AuthorId(core.author_id),
        source_tweet_id: core.source_tweet_id.map(TweetId),
        direct_reply_root_author_id: direct_reply_root_author(core),
    }
}

fn direct_reply_root_author(core: &PureCoreData) -> Option<AuthorId> {
    core.in_reply_to_tweet_id
        .filter(|&replied_to| core.conversation_id == Some(replied_to))
        .and(core.in_reply_to_user_id)
        .map(AuthorId)
}
