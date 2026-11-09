"""
config.py
=========
Central configuration for the Polisis pipeline.
All hyper-parameters, taxonomy definitions, and path settings live here
so every other module imports from a single source of truth.
"""

import os

# ─────────────────────────────────────────────
# Directory layout
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
OPP115_DIR      = os.path.join(DATA_DIR, "opp115")          # OPP-115 corpus
POLICIES_DIR    = os.path.join(DATA_DIR, "policies_corpus") # 130K policies
MODELS_DIR      = os.path.join(BASE_DIR, "models")
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "policies_embeddings.bin")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

for _d in [DATA_DIR, OPP115_DIR, POLICIES_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────
# FastText / Embedding hyper-parameters
# ─────────────────────────────────────────────
FASTTEXT_DIM        = 300
FASTTEXT_MIN_COUNT  = 5
FASTTEXT_WINDOW     = 5
FASTTEXT_EPOCHS     = 5
FASTTEXT_MIN_N      = 3   # min character n-gram
FASTTEXT_MAX_N      = 6   # max character n-gram
FASTTEXT_MODEL      = "skipgram"

# ─────────────────────────────────────────────
# CNN classifier hyper-parameters (Table 1 of paper)
# ─────────────────────────────────────────────
CNN_EMBED_DIM       = 300
CNN_NUM_FILTERS     = 200
CNN_FILTER_SIZE     = 3
CNN_DENSE_SIZE      = 100
CNN_BATCH_SIZE      = 40
CNN_EPOCHS          = 30
CNN_DROPOUT         = 0.5
CNN_LEARNING_RATE   = 1e-3
CNN_THRESHOLD       = 0.5   # Multi-label binarisation threshold

# Train/test split (65 train / 50 test as in the paper)
OPP115_TRAIN_POLICIES = 65
OPP115_TEST_POLICIES  = 50

# ─────────────────────────────────────────────
# Segmenter
# ─────────────────────────────────────────────
MAX_LIST_ITEM_WORDS   = 20   # items ≤ this → merge with intro
GRAPHSEG_THRESHOLD    = 0.25 # cosine-similarity threshold for GraphSeg

# ─────────────────────────────────────────────
# OPP-115 Taxonomy  (Wilson et al., 2016)
# ─────────────────────────────────────────────

# 10 high-level categories (shaded blocks in Fig. 3)
HIGH_LEVEL_CATEGORIES = [
    "First Party Collection/Use",
    "Third Party Sharing/Collection",
    "User Choice/Control",
    "Data Retention",
    "Data Security",
    "International and Specific Audiences",
    "Do Not Track",
    "Policy Change",
    "Access, Edit and Deletion",
    "Other",
]

# ── Attribute → allowed values mapping ──────────────────────────────────────
ATTRIBUTES = {
    # First Party Collection/Use
    "collection_mode": [
        "explicit", "implicit", "unspecified",
    ],
    "action_first_party": [
        "collect-on-website", "collect-in-mobile-app",
        "collect-on-mobile-website", "unspecified",
    ],

    # Third Party Sharing/Collection
    "action_third_party": [
        "receive-shared-with", "collect-on-first-party-website-app",
        "track-on-first-party-website-app", "see", "unspecified",
    ],

    # Shared (1st & 3rd party)
    "personal_information_type": [
        "computer-information", "contact", "cookies-and-tracking-elements",
        "demographic", "financial", "generic-personal-information",
        "health", "ip-address-and-device-ids", "location",
        "personal-identifier", "social-media-data", "survey-data",
        "unspecified", "user-online-activities", "user-profile",
    ],
    "purpose": [
        "additional-service-feature", "advertising", "analytics-research",
        "basic-service-feature", "legal-requirement", "marketing",
        "merger-acquisition", "personalization-customization",
        "service-operation-and-security", "unspecified",
    ],
    "identifiability": [
        "aggregated-or-anonymized", "identifiable", "unspecified",
    ],
    "does_does_not": [
        "does", "does-not",
    ],
    "third_party_entity": [
        "receive-shared-with", "collect-on-first-party-website-app",
        "track-on-first-party-website-app", "see", "unspecified",
    ],

    # User Choice/Control
    "choice_type": [
        "browser-device-privacy-controls", "dont-use-service-feature",
        "first-party-privacy-controls", "opt-in", "opt-out-link",
        "opt-out-via-contacting-company", "third-party-privacy-controls",
        "unspecified",
    ],
    "choice_scope": [
        "both", "collection", "first-party-collection",
        "first-party-use", "third-party-sharing-collection",
        "third-party-use", "unspecified", "use",
    ],

    # Data Retention
    "retention_period": [
        "indefinitely", "limited", "stated-period", "unspecified",
    ],
    "retention_purpose": [
        "advertising", "analytics-research", "legal-requirement",
        "service-operation-and-security", "unspecified",
    ],

    # Data Security
    "security_measure": [
        "data-access-limitation", "generic", "privacy-review-audit",
        "privacy-security-program", "secure-data-storage",
        "secure-data-transfer", "secure-user-authentication",
    ],

    # International and Specific Audiences
    "audience_type": [
        "californians", "children", "europeans",
    ],

    # Do Not Track
    "do_not_track_policy": [
        "honored", "not-honored",
    ],

    # Policy Change
    "change_type": [
        "privacy-relevant-change", "unspecified",
    ],
    "notification_type": [
        "general-notice-in-privacy-policy", "general-notice-on-website",
        "personal-notice", "unspecified",
    ],

    # Access, Edit and Deletion
    "access_scope": [
        "edit-information", "view", "unspecified",
    ],
    "access_type": [
        "edit-information", "view", "unspecified",
    ],
    "access_rights": [
        "yes", "no", "unspecified",
    ],
}

# Map each category to its mandatory attributes
CATEGORY_ATTRIBUTES = {
    "First Party Collection/Use": [
        "action_first_party", "personal_information_type",
        "purpose", "does_does_not", "identifiability",
    ],
    "Third Party Sharing/Collection": [
        "action_third_party", "personal_information_type",
        "purpose", "does_does_not", "third_party_entity",
    ],
    "User Choice/Control": [
        "choice_type", "choice_scope", "does_does_not",
    ],
    "Data Retention": [
        "retention_period", "retention_purpose", "personal_information_type",
    ],
    "Data Security": [
        "security_measure",
    ],
    "International and Specific Audiences": [
        "audience_type",
    ],
    "Do Not Track": [
        "do_not_track_policy",
    ],
    "Policy Change": [
        "change_type", "notification_type",
    ],
    "Access, Edit and Deletion": [
        "access_scope", "access_type", "access_rights",
    ],
    "Other": [],
}

# ─────────────────────────────────────────────
# Disconnect / Privacy icon definitions (Table 2)
# ─────────────────────────────────────────────
DISCONNECT_ICONS = {
    "Expected Use": {
        "description": "Discloses whether data collected is used unexpectedly.",
        "segment_filter": {
            "category": "First Party Collection/Use",
            "purpose": "advertising",
        },
        "yellow_condition": {
            "category": "User Choice/Control",
            "choice_type": ["opt-in", "opt-out-link", "opt-out-via-contacting-company"],
        },
    },
    "Expected Collection": {
        "description": "Discloses whether 3rd parties track users.",
        "segment_filter": {
            "category": "Third Party Sharing/Collection",
            "purpose": ["advertising", "analytics-research"],
            "action_third_party": [
                "track-on-first-party-website-app",
                "collect-on-first-party-website-app",
            ],
        },
        "yellow_condition": {
            "category": "User Choice/Control",
            "choice_type": ["opt-in", "opt-out-link", "opt-out-via-contacting-company"],
        },
    },
    "Precise Location": {
        "description": "Discloses whether geolocation is tracked.",
        "segment_filter": {
            "personal_information_type": "location",
        },
    },
    "Data Retention": {
        "description": "Discloses data retention period.",
        "segment_filter": {
            "category": "Data Retention",
        },
        "green_condition": {
            "retention_period": ["stated-period", "limited"],
        },
    },
    "Children Privacy": {
        "description": "TrustArc Children's Privacy Certification.",
        "segment_filter": {
            "category": "International and Specific Audiences",
            "audience_type": "children",
        },
    },
}
