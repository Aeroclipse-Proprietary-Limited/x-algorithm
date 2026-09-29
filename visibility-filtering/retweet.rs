use crate::filter::{
    EvaluationStatus, FilterOutcome, FilterRequest, FilterTweets, HydratedRequest,
};
use crate::hydration::{HydrationOutput, Hydrators};
use crate::models::{Decided, RawCandidate, TweetFeatures, TweetId, Verdict, Withholding};
use crate::rules::metrics as ft_metrics;
use std::collections::{HashMap, HashSet};

const RETWEET_SOURCES: &str = "evaluate_tweets_retweet_sources";

pub(crate) async fn evaluate_merging_sources(
    filter_tweets: &FilterTweets,
    request: FilterRequest,
) -> Vec<FilterOutcome> {
    let sources_request = FilterRequest {
        country_code: request.country_code.clone(),
        candidates: Vec::new(),
        ..request
    };
    let mut hydrated = filter_tweets.hydrate(request).await;
    let (in_batch, fetched) = source_ids(&hydrated);
    if in_batch.is_empty() && fetched.is_empty() {
        return filter_tweets.evaluate(hydrated).outcomes;
    }
    ft_metrics::incr_nonzero(
        RETWEET_SOURCES,
        &[("outcome", "in_batch")],
        in_batch.len() as u64,
    );
    ft_metrics::incr_nonzero(
        RETWEET_SOURCES,
        &[("outcome", "fetched")],
        fetched.len() as u64,
    );
    let fetched = if fetched.is_empty() {
        None
    } else {
        let candidates = fetched
            .into_iter()
            .map(|tweet_id| RawCandidate {
                tweet_id,
                request_author_id: None,
            })
            .collect();
        Some(
            filter_tweets
                .hydrate(FilterRequest {
                    candidates,
                    ..sources_request
                })
                .await,
        )
    };
    copy_from_sources(&mut hydrated, fetched.as_ref());
    let outcomes = filter_tweets.evaluate(hydrated).outcomes;
    let fetched_outcomes = fetched
        .map(|fetched| filter_tweets.evaluate(fetched).outcomes)
        .unwrap_or_default();
    let is_evaluated_retweet = |outcome: &FilterOutcome| {
        outcome.status == EvaluationStatus::Evaluated && outcome.source_tweet_id.is_some()
    };
    let sources: HashMap<TweetId, (EvaluationStatus, Verdict, Hydrators)> = outcomes
        .iter()
        .filter(|outcome| in_batch.contains(&outcome.tweet_id))
        .map(|outcome| {
            (
                outcome.tweet_id,
                (outcome.status, outcome.verdict.clone(), outcome.rested_on),
            )
        })
        .chain(fetched_outcomes.into_iter().map(|outcome| {
            (
                outcome.tweet_id,
                (outcome.status, outcome.verdict, outcome.rested_on),
            )
        }))
        .collect();
    outcomes
        .into_iter()
        .map(|mut outcome| {
            if is_evaluated_retweet(&outcome) {
                match outcome.source_tweet_id.and_then(|id| sources.get(&id)) {
                    Some((EvaluationStatus::Evaluated, source, source_rested_on)) => {
                        outcome.verdict = merge_verdict(outcome.verdict, source);
                        outcome.rested_on = outcome.rested_on.union(*source_rested_on);
                    }
                    _ => outcome.status = EvaluationStatus::Failed,
                }
            }
            outcome
        })
        .collect()
}

fn source_ids(hydrated: &HydratedRequest) -> (HashSet<TweetId>, HashSet<TweetId>) {
    let HydrationOutput {
        candidates,
        failed_ids,
        pure_cores,
        ..
    } = &hydrated.hydration;
    let sources: HashSet<TweetId> = candidates
        .iter()
        .map(|candidate| TweetId(candidate.tweet_id))
        .filter(|tweet_id| !failed_ids.contains(tweet_id))
        .filter_map(|tweet_id| pure_cores.get(&tweet_id)?.source_tweet_id)
        .collect();
    if sources.is_empty() {
        return Default::default();
    }
    let requested: HashSet<TweetId> = hydrated
        .candidates
        .iter()
        .map(|candidate| candidate.tweet_id)
        .collect();
    sources
        .into_iter()
        .partition(|source_id| requested.contains(source_id))
}

#[derive(Clone, Copy)]
struct CopiedFromSource {
    has_media: bool,
}

impl CopiedFromSource {
    fn of(source: &TweetFeatures) -> Self {
        Self {
            has_media: source.media.has_uploaded_media,
        }
    }

    fn copy_onto(self, retweet: &mut TweetFeatures) {
        retweet.media.has_media = self.has_media;
    }
}

fn copy_from_sources(hydrated: &mut HydratedRequest, fetched: Option<&HydratedRequest>) {
    let copied: HashMap<TweetId, CopiedFromSource> = hydrated
        .hydration
        .candidates
        .iter()
        .chain(
            fetched
                .into_iter()
                .flat_map(|fetched| &fetched.hydration.candidates),
        )
        .map(|candidate| {
            (
                TweetId(candidate.tweet_id),
                CopiedFromSource::of(&candidate.tweet_features),
            )
        })
        .collect();
    let HydrationOutput {
        candidates,
        pure_cores,
        ..
    } = &mut hydrated.hydration;
    for candidate in candidates {
        let source = pure_cores
            .get(&TweetId(candidate.tweet_id))
            .and_then(|core| core.source_tweet_id);
        if let Some(copied) = source.and_then(|source| copied.get(&source)) {
            copied.copy_onto(&mut candidate.tweet_features);
        }
    }
}

fn merge_verdict(retweet: Verdict, source: &Verdict) -> Verdict {
    match (&retweet, source) {
        (
            Verdict::Withheld(Decided {
                value: Withholding::Drop(_),
                ..
            }),
            Verdict::Withheld(Decided {
                value: Withholding::Tombstone(_),
                ..
            }),
        ) => retweet,
        (_, Verdict::Withheld(_)) => source.clone(),
        (Verdict::Withheld(_), _) => retweet,
        (
            Verdict::Shown {
                media: None,
                engagement: None,
            },
            _,
        ) => source.clone(),
        _ => retweet,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::sources::InMemorySources;
    use crate::models::{
        ClientCapability, DropReason, LimitedEngagement, LimitedEngagementReason, MediaFeature,
        MediaInterstitial, MediaRestriction, NsfwFeature, TombstoneReason,
    };
    use crate::rules::fixtures::{allow, legacy_interstitial};
    use crate::rules::metrics::Rpc;
    use crate::rules::{RuleEngine, SafetyLevel};
    use std::sync::Arc;
    use xai_core_entities::entities::PureCoreData;
    use xai_visibility_filtering::models::FilteredReason;
    use xai_x_thrift::action::InterstitialReason;

    #[test]
    fn merge_verdict_prefers_the_more_severe_withholding_then_restricted_retweet() {
        let dropped = Verdict::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(FilteredReason::AuthorIsSuspended)),
            by: "suspended_author/drop",
        });
        let tombstoned = Verdict::Withheld(Decided {
            value: Withholding::Tombstone(TombstoneReason::LocalRegulations),
            by: "nsfw_high_precision/tombstone/local_regulations",
        });
        let blurred = Verdict::Shown {
            media: Some(Decided {
                value: MediaRestriction::MediaInterstitial(MediaInterstitial {
                    legacy: FilteredReason::ContainNsfwMedia,
                    reason: InterstitialReason::Sensitive(true),
                    prompt: None,
                }),
                by: "nsfw_user/blur/sensitive_user",
            }),
            engagement: None,
        };
        let limited = Verdict::Shown {
            media: None,
            engagement: Some(Decided {
                value: LimitedEngagement(LimitedEngagementReason::ConversationControl),
                by: "limit_replies_by_invitation/limited_engagement/conversation_control",
            }),
        };
        let unrestricted = Verdict::Shown {
            media: None,
            engagement: None,
        };
        for (retweet, source, merged) in [
            (&Verdict::unresolved_author(), &dropped, &dropped),
            (&dropped, &tombstoned, &dropped),
            (&tombstoned, &dropped, &dropped),
            (&unrestricted, &tombstoned, &tombstoned),
            (&dropped, &limited, &dropped),
            (&unrestricted, &limited, &limited),
            (&blurred, &limited, &blurred),
        ] {
            assert_eq!(merge_verdict(retweet.clone(), source), *merged);
        }
    }

    #[tokio::test]
    async fn a_retweet_has_media_exactly_when_its_source_uploaded_some_in_the_batch_or_fetched() {
        let features = |source_tweet_id, nsfw_admin, media| TweetFeatures {
            source_tweet_id,
            nsfw: NsfwFeature {
                user: false,
                admin: nsfw_admin,
            },
            media,
            ..Default::default()
        };
        let media = |has_media, has_uploaded_media| MediaFeature {
            has_media,
            has_uploaded_media,
            ..Default::default()
        };
        let retweet = |author_id, source_tweet_id| PureCoreData {
            author_id,
            source_tweet_id: Some(source_tweet_id),
            ..Default::default()
        };
        let sources = Arc::new(
            InMemorySources::default()
                .pure_core(4, retweet(40, 6))
                .pure_core(5, retweet(50, 7))
                .tweet(6, 60)
                .tweet(7, 70)
                .tweet_features(4, features(Some(6), true, media(false, false)))
                .tweet_features(5, features(Some(7), true, media(true, true)))
                .tweet_features(6, features(None, false, media(true, true)))
                .tweet_features(7, features(None, false, media(true, false))),
        );
        let filter_tweets = FilterTweets::new(sources, RuleEngine::for_tests());
        let blurred = (
            EvaluationStatus::Evaluated,
            legacy_interstitial("nsfw_account/legacy_interstitial"),
        );
        let allowed = (EvaluationStatus::Evaluated, allow());
        for tweet_ids in [vec![4, 5, 6, 7], vec![4, 5]] {
            let outcomes = evaluate_merging_sources(
                &filter_tweets,
                FilterRequest {
                    viewer_id: Some(1),
                    country_code: None,
                    client_capability: ClientCapability::default(),
                    safety_level: SafetyLevel::TimelineHomeHydration,
                    candidates: tweet_ids
                        .iter()
                        .map(|&tweet_id| RawCandidate {
                            tweet_id: TweetId(tweet_id),
                            request_author_id: None,
                        })
                        .collect(),
                    rpc: Rpc::EvaluateTweets,
                },
            )
            .await;
            let verdict = |i: usize| (outcomes[i].status, outcomes[i].verdict.clone());
            assert_eq!(verdict(0), blurred, "{tweet_ids:?}");
            assert_eq!(verdict(1), allowed, "{tweet_ids:?}");
        }
    }
}
