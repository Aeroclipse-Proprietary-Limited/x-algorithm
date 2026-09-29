use super::builders::labeled;
use super::{Role, Row};
use crate::models::{ClientCapability, SafetyLabelType, ViewerFeatures};
use crate::rules::fixtures::{allow, dropped, viewer, AUTHOR_ID, VIEWER_ID};
use crate::rules::SafetyLevel::{
    ImmersiveExpandedRecommendations, TimelineHome, TimelineHomeHydration,
    TimelineHomeRecommendations,
};
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "malicious_url_label",
            post: labeled(SafetyLabelType::MALICIOUS_URL),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "malicious_url/drop/undesirable",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "malicious_url/drop/undesirable",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
                (TimelineHome, Role::NonFollower, allow()),
            ],
        },
        Row {
            name: "nsfw_high_recall_label",
            post: labeled(SafetyLabelType::NSFW_HIGH_RECALL),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_high_recall/drop/nsfw_media",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_high_recall/drop/nsfw_media",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
            ],
        },
        Row {
            name: "do_not_amplify_label",
            post: labeled(SafetyLabelType::DO_NOT_AMPLIFY),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "do_not_amplify/drop/undesirable",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "do_not_amplify/drop/undesirable",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
            ],
        },
        Row {
            name: "spam_high_recall_label",
            post: labeled(SafetyLabelType::SPAM_HIGH_RECALL),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "spam_high_recall/drop/undesirable",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "spam_high_recall/drop/undesirable",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
                (TimelineHome, Role::NonFollower, allow()),
                (
                    ImmersiveExpandedRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "spam_high_recall/drop/undesirable",
                    ),
                ),
            ],
        },
        Row {
            name: "fosnr_abuse_insults_label",
            post: labeled(SafetyLabelType::FOSNR_ABUSE_INSULTS),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_abuse_insults/drop/undesirable",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_abuse_insults/drop/undesirable",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
                (TimelineHome, Role::NonFollower, allow()),
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_abuse_insults_non_follower/drop/undesirable",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    Role::LoggedOut,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_abuse_insults_non_follower/drop/undesirable",
                    ),
                ),
                (TimelineHomeHydration, Role::Follower, allow()),
                (
                    TimelineHomeHydration,
                    Role::As("client_without_fosnr", client_without_fosnr(VIEWER_ID)),
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_fallback/drop/undesirable",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    Role::As(
                        "author_on_client_without_fosnr",
                        client_without_fosnr(AUTHOR_ID),
                    ),
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_fallback/drop/undesirable",
                    ),
                ),
            ],
        },
    ]
}

fn client_without_fosnr(viewer_id: u64) -> ViewerFeatures {
    ViewerFeatures {
        client_capability: ClientCapability {
            fosnr_rules: false,
            fosnr_fallback_drops: true,
            ..ClientCapability::default()
        },
        ..viewer(viewer_id)
    }
}
