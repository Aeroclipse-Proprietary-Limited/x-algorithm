use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::action::{InterstitialAction, InterstitialReason};

#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    Withheld(Decided<Withholding>),
    Shown {
        media: Option<Decided<MediaRestriction>>,
        engagement: Option<Decided<LimitedEngagement>>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Decided<T> {
    pub value: T,
    pub by: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Withholding {
    Drop(DropReason),
    Tombstone(TombstoneReason),
}

#[derive(Clone, Debug, PartialEq)]
pub enum DropReason {
    Legacy(FilteredReason),
    NsfwViewer(NsfwViewerDropReason),
}

impl DropReason {
    pub fn legacy(&self) -> &FilteredReason {
        match self {
            Self::Legacy(reason) => reason,
            Self::NsfwViewer(_) => &FilteredReason::ContainNsfwMedia,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum NsfwViewerDropReason {
    IsUnderage,
    HasNoStatedAge,
    LoggedOut,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MediaRestriction {
    MediaInterstitial(MediaInterstitial),
    NsfwInterstitial,
}

impl MediaRestriction {
    pub fn legacy(&self) -> &FilteredReason {
        match self {
            Self::MediaInterstitial(blur) => &blur.legacy,
            Self::NsfwInterstitial => &FilteredReason::ContainNsfwMedia,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MediaInterstitial {
    pub legacy: FilteredReason,
    pub reason: InterstitialReason,
    pub prompt: Option<InterstitialAction>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LimitedEngagement(pub LimitedEngagementReason);

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum TombstoneReason {
    SensitiveViewerAgeVerification,
    UpdateAppIos,
    UpdateAppAndroid,
    LocalRegulations,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum LimitedEngagementReason {
    ConversationControl,
    ReadonlyViewer,
    BlockedViewer,
    RootAuthorBlockedViewer,
    StaleTweet,
}

impl Verdict {
    pub fn unresolved_author() -> Self {
        Self::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(FilteredReason::UnspecifiedReason)),
            by: "unresolved_author_id",
        })
    }
}
